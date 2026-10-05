import numpy as np
import pytest
from photutils.segmentation import SegmentationImage, SourceCatalog

from neo.data.dataset import SR_HST_HSC_Dataset
from neo.eval import catalogs, metrics, postprocess


def gaussian(shape, cy, cx, sigma, amp, q=1.0, theta_deg=0.0):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    t = np.radians(theta_deg)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    return amp * np.exp(-0.5 * ((u / sigma) ** 2 + (v / (q * sigma)) ** 2))


def block_sum(img, f):
    ny, nx = img.shape
    return img.reshape(ny // f, f, nx // f, f).sum(axis=(1, 3))


def test_to_physical_inverts_the_training_scaling_and_padding():
    rng = np.random.default_rng(0)
    hr = rng.uniform(0, 50, size=(600, 600))
    scaled = SR_HST_HSC_Dataset.ds9_scaling(hr, offset=1)
    padded = np.pad(scaled, 84, mode="reflect")
    assert padded.shape == (768, 768)
    np.testing.assert_allclose(postprocess.to_physical(padded), hr, rtol=1e-10, atol=1e-10)


def test_center_crop_matches_the_dataset_crop_offsets():
    a = np.arange(852 * 852).reshape(852, 852)
    assert postprocess.center_crop(a, 600)[0, 0] == a[126, 126]
    b = np.arange(142 * 142).reshape(142, 142)
    assert postprocess.center_crop(b, 100)[0, 0] == b[21, 21]


def test_balance_noise_refills_zeros_with_negative_noise_only():
    rng = np.random.default_rng(1)
    img = np.abs(rng.normal(0, 1, size=(80, 80)))
    img[rng.random(img.shape) < 0.3] = 0
    out = postprocess.balance_noise(img, rng=np.random.default_rng(2))
    zeros = img == 0
    assert np.all(out[zeros] < 0) and np.array_equal(out[~zeros], img[~zeros])


def test_paper_segmap_reprojection_lands_on_the_nested_grid():
    seg = np.zeros((600, 600), int)
    seg[60:66, 120:126] = 1
    lr = catalogs.lr_segmap(SegmentationImage(seg), (100, 100), 6).data
    assert list(zip(*np.nonzero(lr), strict=True)) == [(10, 20)]


def test_default_threshold_is_the_paper_surface_brightness_per_0p03_pixel():
    # the paper's value is e-/s per native 0.03" pixel (its 0.028" grid preserved surface
    # brightness): unchanged in e-/s on 0.03" pixels, whatever the grid it was sampled on
    assert catalogs.default_threshold(1.0, 0.03) == pytest.approx(catalogs.PAPER_THRESHOLD_CPS)
    # COSMOS-Web F814W (ZP 25.94: 152.76 nJy per e-/s) on our 0.0333" grid: 1.3036 nJy per pixel
    hrscale = 152.7566058238
    assert catalogs.default_threshold(hrscale, 0.2 / 6) == pytest.approx(1.3036, rel=1e-4)
    # for --units paper pairs it is the paper's number times the cutout's NJYPERPX
    njy_per_px = hrscale * (0.2 / 6 / 0.03) ** 2
    assert catalogs.default_threshold(hrscale, 0.2 / 6) == pytest.approx(
        catalogs.PAPER_THRESHOLD_CPS * njy_per_px, rel=1e-12
    )


def test_paper_minimum_area_and_kernel_match_the_paper_on_the_sky():
    assert catalogs.paper_npixels(0.168 / 6) == 100
    assert catalogs.paper_npixels(0.2 / 6) == 71  # 100 * (0.028 / 0.0333)^2 = 70.56
    assert catalogs.paper_fwhm(0.168 / 6) == pytest.approx(3.0)
    # same smoothing on the sky as the paper's 3 px FWHM on a 3x3 kernel (truncation included)
    hr_fwhm = catalogs.paper_fwhm(0.2 / 6)
    assert hr_fwhm == pytest.approx(1.652, abs=1e-3)
    assert catalogs.kernel_width(hr_fwhm) * 0.2 / 6 == pytest.approx(
        catalogs.kernel_width(3.0) * 0.168 / 6
    )
    assert catalogs.paper_fwhm(0.2, catalogs.PAPER_LR_PIXEL_ARCSEC) == pytest.approx(hr_fwhm)
    assert catalogs.paper_fwhm(0.168, catalogs.PAPER_LR_PIXEL_ARCSEC) == pytest.approx(3.0)


def test_create_convolved_data_default_is_the_papers_3px_kernel():
    rng = np.random.default_rng(4)
    img = rng.normal(size=(40, 40))
    np.testing.assert_array_equal(
        catalogs.create_convolved_data(img), catalogs.create_convolved_data(img, 3.0)
    )
    assert not np.allclose(
        catalogs.create_convolved_data(img), catalogs.create_convolved_data(img, 2.52)
    )


def scene_images():
    """Two noisy Gaussian galaxies (HST), a block-summed LR copy and two SR versions."""
    rng = np.random.default_rng(3)
    hst = gaussian((600, 600), 300, 300, 9.0, 40.0) + gaussian(
        (600, 600), 150, 420, 6.0, 25.0, 0.5, 30
    )
    hst = hst + rng.normal(0, 0.02, hst.shape)
    lr = block_sum(hst, 6)
    blurred = gaussian((600, 600), 300, 300, 11.0, 40.0 * (9 / 11) ** 2) + gaussian(
        (600, 600), 150, 420, 7.5, 25.0 * (6 / 7.5) ** 2, 0.6, 45
    )
    sr = {"perfect": hst.copy(), "blurry": blurred + rng.normal(0, 0.02, hst.shape)}
    return hst, sr, lr


@pytest.fixture(scope="module")
def scene():
    hst, sr, lr = scene_images()
    result = catalogs.catalog_set(
        postprocess.subtract_background(hst),
        {k: postprocess.subtract_background(v) for k, v in sr.items()},
        lr,
        threshold=0.5,
    )
    assert result is not None
    return result


def test_catalog_recovers_gaussian_morphology(scene):
    hst_tbl, _, _ = scene
    assert len(hst_tbl) == 2
    big = hst_tbl[np.argmax(hst_tbl["segment_flux"])]
    assert big["half_light_radius"] == pytest.approx(np.sqrt(2 * np.log(2)) * 9.0, rel=0.05)
    c_gauss = np.sqrt(np.log(4)) / np.sqrt(np.log(4 / 3))
    assert big["flux_concentration_75_25"] == pytest.approx(c_gauss, rel=0.05)


def test_identical_image_has_zero_bias_and_lr_is_scaled_to_hr_pixels(scene):
    hst_tbl, sr_tbls, lr_tbl = scene
    perfect = metrics.per_source(hst_tbl, sr_tbls["perfect"])
    for name, vals in perfect.items():
        assert np.allclose(vals, 0, atol=1e-9), name
    lr = metrics.per_source(hst_tbl, lr_tbl, factor=6)
    assert np.all(np.abs(lr["R_e"]) < 0.15)  # block-summed copy: same size once scaled by 6
    blurry = metrics.per_source(hst_tbl, sr_tbls["blurry"])
    assert np.all(blurry["R_e"] > 0.05)  # a wider galaxy is measured as wider


def test_orientation_similarity():
    assert metrics.orientation_similarity([10, 10, 10], [10, 190, 100]) == pytest.approx([0, 0, 1])


def test_summarize_and_gain():
    s = metrics.summarize(np.r_[np.full(99, 0.1), np.nan])
    assert s["n"] == 99 and s["median"] == pytest.approx(0.1) and s["nmad"] == 0
    g = metrics.gain(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0.1, 0.1, 0.1, 10.0]))
    assert g["frac_improved"] == 0.75 and g["n"] == 4


def test_lr_shapes_from_the_smoothed_copy_when_asked(scene):
    _, _, lr_tbl = scene
    hst, _, lr = scene_images()
    hst = postprocess.subtract_background(hst)
    smoothed = catalogs.catalog_set(hst, {}, lr, threshold=0.5, lr_fwhm=2.52)[2]
    segm, _ = catalogs.detect_hst(hst, 0.5)
    segm_lr = catalogs.lr_segmap(segm, lr.shape, 6)
    expected = catalogs.measure(
        SourceCatalog(lr, segm_lr, convolved_data=catalogs.create_convolved_data(lr, 2.52))
    )
    np.testing.assert_allclose(smoothed["fwhm"], expected["fwhm"])
    np.testing.assert_allclose(smoothed["segment_flux"], lr_tbl["segment_flux"])  # fluxes: data
    assert not np.allclose(smoothed["fwhm"], lr_tbl["fwhm"])  # shapes: the smoothed copy


def test_paper_q_statistic_is_minus_our_q_bias_and_a_68pct_spread(scene):
    hst_tbl, sr_tbls, lr_tbl = scene
    for tbl in (sr_tbls["blurry"], lr_tbl):
        b = metrics.ellipticity_bias(hst_tbl, tbl)
        np.testing.assert_allclose(b, -metrics.per_source(hst_tbl, tbl)["q"], atol=1e-12)
    b = np.r_[np.linspace(-0.2, 0.3, 101), np.nan]
    q = metrics.paper_q_statistic(b)
    finite = b[np.isfinite(b)]
    assert q["n"] == 101 and q["median"] == pytest.approx(np.median(finite))
    assert q["q68"] == pytest.approx(np.quantile(np.abs(finite - np.median(finite)), 0.68))
    assert q["std"] == pytest.approx(np.std(finite))


def test_per_source_rejects_misaligned_catalogs(scene):
    hst_tbl, sr_tbls, _ = scene
    other = sr_tbls["perfect"].copy()
    other["label"] = other["label"][::-1]
    with pytest.raises(ValueError):
        metrics.per_source(hst_tbl, other)
