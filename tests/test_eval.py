import numpy as np
import pytest
from photutils.segmentation import SegmentationImage

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


def test_default_threshold_converts_the_paper_value():
    # same units and pixel scale as the paper -> unchanged; 152.4 nJy per count, 0.0333" pixels
    assert catalogs.default_threshold(1.0, 0.028) == pytest.approx(catalogs.PAPER_THRESHOLD_CPS)
    expected = catalogs.PAPER_THRESHOLD_CPS * 152.4 * (0.2 / 6 / 0.028) ** 2
    assert catalogs.default_threshold(152.4, 0.2 / 6) == pytest.approx(expected)


@pytest.fixture(scope="module")
def scene():
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


def test_per_source_rejects_misaligned_catalogs(scene):
    hst_tbl, sr_tbls, _ = scene
    other = sr_tbls["perfect"].copy()
    other["label"] = other["label"][::-1]
    with pytest.raises(ValueError):
        metrics.per_source(hst_tbl, other)
