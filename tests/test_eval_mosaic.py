import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS

from neo.eval import mosaic
from neo.eval.postprocess import center_crop

PIX = 0.2  # LSST arcsec per pixel
BORDER = 20  # synthetic patches: 120 px outer, 80 px inner


def tract_wcs_header():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [53.1, -27.8]
    wcs.wcs.crpix = [200.5, 200.5]  # tract pixel grid
    wcs.wcs.cd = np.array([[-PIX / 3600, 0], [0, PIX / 3600]])
    return wcs


def write_patch(path, tract, px, py, outer=120, flag=None):
    """A patch whose pixel (y, x) holds 1000 * tract_y + tract_x (nJy), DP2 header layout."""
    wcs = tract.deepcopy()
    wcs.wcs.crpix = wcs.wcs.crpix - np.array([px, py])
    ys, xs = np.mgrid[0:outer, 0:outer]
    image = (1000.0 * (ys + py) + (xs + px)).astype(np.float32)
    header = wcs.to_header()
    header["BUNIT"] = "nJy"
    header["CRVAL1A"], header["CRVAL2A"] = float(px), float(py)
    header["CRPIX1A"], header["CRPIX2A"] = 1.0, 1.0
    mask = np.zeros((outer, outer), np.int32)
    if flag is not None:
        mask[flag] = 1
    mhdu = fits.ImageHDU(mask, name="MASK")
    mhdu.header["MSKN0000"], mhdu.header["MSKM0000"] = "NO_DATA", 1
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(image, header=header, name="IMAGE"), mhdu]
    ).writeto(path)


def test_patch_origin_from_alternate_wcs():
    header = fits.Header({"CRVAL1A": 8850.0, "CRVAL2A": 2850.0, "CRPIX1A": 1.0, "CRPIX2A": 1.0})
    assert mosaic.patch_origin(header) == (8850, 2850)


def test_build_lr_stitches_inner_regions_without_resampling(tmp_path):
    tract = tract_wcs_header()
    paths = []
    # 2 x 2 patches on an 80 px inner grid, each with a 20 px border (overlapping neighbours)
    for j, py in enumerate((-BORDER, 80 - BORDER)):
        for i, px in enumerate((-BORDER, 80 - BORDER)):
            path = tmp_path / f"deep_coadd_1_{j * 10 + i}_i.fits"
            flag = (slice(50, 52), slice(50, 52)) if (i, j) == (0, 0) else None
            write_patch(path, tract, px, py, flag=flag)
            paths.append(str(path))
    ra, dec = tract.pixel_to_world_values(80, 80)
    stored, bad, wcs, cards = mosaic.build_lr(
        paths, float(ra), float(dec), 100 * PIX / 60, units_scale=1.0, border=BORDER
    )
    assert stored.shape == (100, 100)  # 100 px, already a multiple of the stride
    x0, y0 = cards["TRACTX0"], cards["TRACTY0"]
    assert (x0, y0) == (30, 30)
    ys, xs = np.mgrid[0:100, 0:100]
    expected = 1000.0 * (ys + y0) + (xs + x0)
    covered = (ys + y0 < 160) & (xs + x0 < 160)  # the 2 x 2 inner grid spans tract 0..159
    flagged = (ys + y0 >= 30) & (ys + y0 < 32) & (xs + x0 >= 30) & (xs + x0 < 32)
    np.testing.assert_array_equal(bad, ~covered | flagged)
    np.testing.assert_array_equal(stored[~bad], expected[~bad].astype(np.float32))
    assert np.all(stored[bad] == 0)
    # the LR WCS is the tract WCS shifted to the box
    np.testing.assert_allclose(
        wcs.pixel_to_world_values(0, 0), tract.pixel_to_world_values(x0, y0), atol=1e-10
    )


def test_build_lr_rounds_side_to_stride(tmp_path):
    tract = tract_wcs_header()
    path = tmp_path / "deep_coadd_1_0_i.fits"
    write_patch(path, tract, -BORDER, -BORDER, outer=400)
    ra, dec = tract.pixel_to_world_values(150, 150)
    stored, *_ = mosaic.build_lr([str(path)], float(ra), float(dec), 101 * PIX / 60, border=BORDER)
    assert stored.shape == (120, 120)
    mosaic.check_tiling(stored.shape)


def test_build_lr_refuses_mixed_tracts(tmp_path):
    tract = tract_wcs_header()
    write_patch(tmp_path / "a.fits", tract, 0, 0)
    other = tract.deepcopy()
    other.wcs.crval = [60.0, -27.8]
    write_patch(tmp_path / "b.fits", other, 0, 0)
    with pytest.raises(ValueError, match="more than one tract"):
        mosaic.build_lr([str(tmp_path / "a.fits"), str(tmp_path / "b.fits")], 53.1, -27.8, 0.3)


@pytest.mark.parametrize("shape", [(90, 120), (110, 120), (120, 130)])
def test_check_tiling_rejects_untiled_shapes(shape):
    with pytest.raises(ValueError):
        mosaic.check_tiling(shape)


def upsample_generator(x):
    """Rotation-equivariant stand-in for the generator: nearest 6x upsample of the log input."""
    return torch.nn.functional.interpolate(x, scale_factor=6, mode="nearest")


def tied_lr(shape, seed=0):
    """Integer LR values 0..9: the maximum repeats in every window, so the per-window clip at the
    99.999th percentile changes nothing and the model's round trip is exact."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10, size=shape).astype(np.float32)


def test_window_model_returns_hr_units_and_rot4_rotates_back():
    lr = tied_lr((2, 100, 100))
    lr[:, :5, :] = 9  # an asymmetric feature: a wrong rotate-back would move it
    for tta in ("none", "rot4"):
        out = mosaic.make_window_model(upsample_generator, tta=tta)([lr[..., None]])
        assert out.shape == (2, 600, 600, 1)
        expected = np.repeat(np.repeat(lr, 6, axis=1), 6, axis=2)
        np.testing.assert_allclose(out[..., 0], expected, rtol=1e-4, atol=1e-4)


def test_window_model_rot4_reduces_over_rotations():
    # a generator that is NOT equivariant: adds +0.05 (log space) in its own top-left quadrant
    def biased(x):
        y = upsample_generator(x).clone()
        y[..., 84:384, 84:384] += 0.05
        return y

    lr = tied_lr((1, 100, 100))
    single = mosaic.make_window_model(biased, tta="none")([lr[..., None]])[0, ..., 0]
    med = mosaic.make_window_model(biased, tta="rot4", reduce="median")([lr[..., None]])[0, ..., 0]
    up = np.repeat(np.repeat(lr[0], 6, axis=0), 6, axis=1)
    # single pass: the bias sits in the top-left quadrant only
    assert np.all(single[:300, :300] > up[:300, :300] - 1e-6)
    assert np.allclose(single[300:, 300:], up[300:, 300:], atol=1e-4)
    # rotated back, each quadrant is biased in exactly one of the four passes: the median of
    # three unbiased values and one biased one is the unbiased value
    np.testing.assert_allclose(med, up, rtol=1e-4, atol=1e-4)


def write_lr_file(path, lr, bad=None):
    wcs = tract_wcs_header()[0 : lr.shape[0], 0 : lr.shape[1]]
    cards = {"NJYPERPX": 2.0, "BUNIT": "test"}
    bad = np.zeros(lr.shape, bool) if bad is None else bad
    mosaic.write_lr(path, lr, bad, wcs, cards)
    return wcs


def run_morpheus(model, lr, out, batch=4):
    from morpheus_core import morpheus_core

    out.mkdir()
    hduls, _ = morpheus_core.predict(
        model,
        [lr],
        1,
        batch,
        (mosaic.WINDOW, mosaic.WINDOW),
        dilation=mosaic.FACTOR,
        stride=(mosaic.STRIDE, mosaic.STRIDE),
        aggregate_method=morpheus_core.AGGREGATION_METHODS.MEAN_VAR,
        out_dir=str(out),
    )
    for hdul in hduls:
        hdul.close()


def brute_force_flag(bad):
    flag = np.zeros(bad.shape, np.uint8)
    for wy in range(0, bad.shape[0] - 100 + 1, 20):
        for wx in range(0, bad.shape[1] - 100 + 1, 20):
            if bad[wy : wy + 100, wx : wx + 100].any():
                flag[wy : wy + 100, wx : wx + 100] = mosaic.FLAG_NEAR_BAD
    flag[bad] = mosaic.FLAG_BAD
    return flag


@pytest.mark.parametrize(
    "points", [[], [(0, 0)], [(70, 85)], [(159, 179), (100, 3)], [(19, 19), (20, 20)]]
)
def test_lr_flag_matches_window_brute_force(points):
    bad = np.zeros((160, 180), bool)
    for y, x in points:
        bad[y, x] = True
    np.testing.assert_array_equal(mosaic.lr_flag(bad), brute_force_flag(bad))


def test_lr_flag_on_a_bad_strip():
    bad = np.zeros((200, 240), bool)
    bad[:, :7] = True  # an uncovered left edge
    np.testing.assert_array_equal(mosaic.lr_flag(bad), brute_force_flag(bad))


def test_read_bad_without_extension(tmp_path):
    path = tmp_path / "old.fits"
    fits.PrimaryHDU(np.ones((120, 140), np.float32)).writeto(path)
    assert not mosaic.read_bad(path).any()


def test_finalize_masks_bad_pixels(tmp_path):
    lr = tied_lr((140, 160), seed=2)
    bad = np.zeros(lr.shape, bool)
    bad[10:13, 150:153] = True  # a NO_DATA blob near the right edge
    lr[bad] = 0.0
    lr_path = tmp_path / "lr.fits"
    write_lr_file(lr_path, lr, bad)
    out = tmp_path / "masked"
    run_morpheus(mosaic.make_window_model(upsample_generator), lr, out)
    stats = mosaic.finalize(out, lr_path, rows_per_chunk=97)  # rounded down to whole LR rows

    flag_lr = brute_force_flag(bad)
    flag = fits.getdata(out / "sr_flag.fits")
    assert flag.dtype == np.uint8 and flag.shape == (840, 960)
    np.testing.assert_array_equal(flag, np.repeat(np.repeat(flag_lr, 6, axis=0), 6, axis=1))
    hr_bad = flag == mosaic.FLAG_BAD
    for name in ("sr_mean", "sr_std", "sr_paper"):
        data = fits.getdata(out / f"{name}.fits")
        assert np.isnan(data[hr_bad]).all() and np.isfinite(data[~hr_bad]).all(), name
    assert np.isfinite(fits.getdata(out / "sr_n.fits")).all()
    assert stats["coverage"] == pytest.approx(1 - bad.mean())
    assert stats["near_bad_frac"] == pytest.approx((flag_lr == mosaic.FLAG_NEAR_BAD).mean())
    assert np.isfinite(stats["flux_mean_njy"])
    # flags and WCS travel with sr_flag
    header = fits.getheader(out / "sr_flag.fits")
    assert header["FLAGBAD"] == mosaic.FLAG_BAD and header["PRODUCT"] == "sr_flag"
    assert WCS(header).has_celestial
    mosaic.preview(out, lr_path)  # NaNs do not break the preview


def test_finalize_refuses_a_mismatched_lr(tmp_path):
    lr = tied_lr((120, 120), seed=3)
    out = tmp_path / "m"
    run_morpheus(mosaic.make_window_model(upsample_generator), lr, out)
    other = tmp_path / "other.fits"
    write_lr_file(other, np.zeros((140, 140), np.float32))
    with pytest.raises(ValueError, match="not 6x"):
        mosaic.finalize(out, other)


@pytest.mark.parametrize("tta", ["none", "rot4"])
def test_morpheus_mosaic_end_to_end(tmp_path, tta):
    from morpheus_core import morpheus_core

    lr = tied_lr((140, 160), seed=1)
    lr_path = tmp_path / "lr.fits"
    lr_wcs = write_lr_file(lr_path, lr)
    out = tmp_path / tta
    out.mkdir()
    model = mosaic.make_window_model(upsample_generator, tta=tta)
    hduls, _ = morpheus_core.predict(
        model,
        [lr],
        1,
        4,
        (mosaic.WINDOW, mosaic.WINDOW),
        dilation=mosaic.FACTOR,
        stride=(mosaic.STRIDE, mosaic.STRIDE),
        aggregate_method=morpheus_core.AGGREGATION_METHODS.MEAN_VAR,
        out_dir=str(out),
    )
    for hdul in hduls:
        hdul.close()
    stats = mosaic.finalize(out, lr_path, rows_per_chunk=97)
    assert stats["coverage"] == 1.0

    scale = mosaic.hr_njy_per_px(lr_wcs)
    up = np.repeat(np.repeat(lr, 6, axis=0), 6, axis=1)
    mean = fits.getdata(out / "sr_mean.fits")
    n = fits.getdata(out / "sr_n.fits")
    std = fits.getdata(out / "sr_std.fits")
    paper = fits.getdata(out / "sr_paper.fits")
    assert mean.shape == (840, 960)
    np.testing.assert_allclose(mean, up * scale, rtol=1e-4, atol=1e-4 * scale)
    np.testing.assert_allclose(std, 0, atol=1e-3 * scale)
    np.testing.assert_allclose(paper, mean, rtol=1e-3, atol=1e-3 * scale)
    # windows per HR pixel: every 600 px window at a multiple of the 120 px dilated stride
    expected_n = np.zeros(n.shape)
    for wy in range(0, 140 - 100 + 1, 20):
        for wx in range(0, 160 - 100 + 1, 20):
            expected_n[wy * 6 : wy * 6 + 600, wx * 6 : wx * 6 + 600] += 1
    np.testing.assert_array_equal(n, expected_n)
    assert n[0, 0] == 1 and n.max() == 12  # 3 x 4 windows; 25 once both axes hold 5+ windows

    # the HR grid nests in the LR grid: HR pixel (2.5, 2.5) is the centre of LR pixel (0, 0)
    hr_wcs = WCS(fits.getheader(out / "sr_mean.fits"))
    np.testing.assert_allclose(
        hr_wcs.pixel_to_world_values(2.5, 2.5), lr_wcs.pixel_to_world_values(0, 0), atol=1e-10
    )
    assert fits.getheader(out / "sr_mean.fits")["BUNIT"] == "nJy"
    assert (out / "preview.png").exists() is False  # preview is a separate step
    mosaic.preview(out, lr_path)
    assert (out / "preview.png").exists()


def test_hr_njy_per_px_matches_pairs_units():
    from neo.surveys.hst.mosaic import F814W_AB_ZEROPOINT, njy_per_count

    wcs = tract_wcs_header()
    expected = njy_per_count(F814W_AB_ZEROPOINT) * ((PIX / 6) / 0.03) ** 2
    assert mosaic.hr_njy_per_px(wcs) == pytest.approx(expected, rel=1e-9)


def test_center_crop_matches_window_padding():
    # the 14 px LR pad becomes the 84 px HR pad that to_physical removes
    assert mosaic.PAD * mosaic.FACTOR == (768 - 600) // 2
    assert center_crop(np.zeros((1, 768, 768)), 600).shape == (1, 600, 600)


def test_morpheus_variance_is_finalised_everywhere(tmp_path):
    """Window k adds k to its prediction: every HR pixel's mean and variance must be those of the
    window indices covering it (population variance, morpheus's S_n / n), at every pixel."""
    from morpheus_core import morpheus_core

    shape = (160, 180)
    lr = np.zeros(shape, np.float32)
    lr_path = tmp_path / "lr.fits"
    write_lr_file(lr_path, lr)
    counter = iter(range(10_000))

    def model(batch_inputs):
        b = batch_inputs[0].shape[0]
        return np.stack([np.full((600, 600, 1), next(counter), np.float32) for _ in range(b)])

    out = tmp_path / "var"
    out.mkdir()
    hduls, _ = morpheus_core.predict(
        model,
        [lr],
        1,
        3,
        (100, 100),
        dilation=6,
        stride=(20, 20),
        aggregate_method=morpheus_core.AGGREGATION_METHODS.MEAN_VAR,
        out_dir=str(out),
    )
    for hdul in hduls:
        hdul.close()
    mosaic.finalize(out, lr_path)
    scale = mosaic.hr_njy_per_px(WCS(fits.getheader(lr_path)))

    values = [[[] for _ in range(shape[1] * 6)] for _ in range(shape[0] * 6)]
    k = 0
    for wy in range(0, shape[0] - 100 + 1, 20):  # morpheus visits windows in raster order
        for wx in range(0, shape[1] - 100 + 1, 20):
            for y in range(wy * 6, wy * 6 + 600, 60):  # a sparse grid of probe pixels is enough
                for x in range(wx * 6, wx * 6 + 600, 60):
                    values[y][x].append(k)
            k += 1
    mean = fits.getdata(out / "sr_mean.fits")
    std = fits.getdata(out / "sr_std.fits")
    for y in range(0, shape[0] * 6, 60):
        for x in range(0, shape[1] * 6, 60):
            v = np.array(values[y][x], dtype=float)
            assert mean[y, x] == pytest.approx(v.mean() * scale, rel=1e-5, abs=1e-6)
            assert std[y, x] == pytest.approx(v.std() * scale, rel=1e-4, abs=1e-5)


def test_sr_refuses_a_mostly_bad_lr(tmp_path):
    lr = np.zeros((120, 120), np.float32)
    bad = np.zeros(lr.shape, bool)
    bad[:, :10] = True  # 8.3% uncovered
    lr_path = tmp_path / "lr.fits"
    write_lr_file(lr_path, lr, bad)
    argv = ["sr", "--config", "x.ini", "--checkpoint", "x.pt", "--lr", str(lr_path)]
    argv += ["--out", str(tmp_path / "o"), "--no-comet"]
    with pytest.raises(SystemExit, match="max-badfrac"):
        mosaic.main(argv)


def test_sr_refusal_and_overwrite_guard_touch_nothing(tmp_path):
    lr = np.zeros((120, 120), np.float32)
    bad = np.zeros(lr.shape, bool)
    bad[:, :10] = True
    lr_path = tmp_path / "lr.fits"
    write_lr_file(lr_path, lr, bad)
    out = tmp_path / "o"
    out.mkdir()
    old = {name: b"old" for name in ("output.fits", "n.fits", "sr_mean.fits", "mosaic.json")}
    for name, data in old.items():
        (out / name).write_bytes(data)
    base = ["sr", "--config", "x.ini", "--checkpoint", "x.pt", "--lr", str(lr_path)]
    base += ["--out", str(out), "--no-comet"]
    # refused on the bad fraction, even with --overwrite: the old run is untouched
    with pytest.raises(SystemExit, match="max-badfrac"):
        mosaic.main([*base, "--overwrite"])
    # passes the bad-fraction check, but an existing product needs --overwrite
    (out / "output.fits").unlink()
    (out / "n.fits").unlink()
    with pytest.raises(SystemExit, match="--overwrite"):
        mosaic.main([*base, "--max-badfrac", "0.5"])
    assert all((out / name).read_bytes() == b"old" for name in ("sr_mean.fits", "mosaic.json"))
