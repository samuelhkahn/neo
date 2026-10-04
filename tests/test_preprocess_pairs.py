import glob

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from conftest import make_tan_wcs
from test_hst_mosaic import write_mosaic

from neo.preprocess import leakage, pairs
from neo.preprocess.grid import lr_window_mask
from neo.surveys.hst.mosaic import MosaicSet

F = 6
AREA_RATIO = (0.2 / F / 0.03) ** 2  # HR output pixel area / mosaic pixel area


def synthetic_scene():
    """A 24x24 LR frame at 0.2\"/px and a 3\" HR mosaic at 0.03\"/px centered on it."""
    lr_wcs = make_tan_wcs(0.2, (24, 24))
    center = lr_wcs.pixel_to_world(11.5, 11.5)
    hr_wcs = make_tan_wcs(0.03, (100, 100), crval=(center.ra.deg, center.dec.deg))
    hr_data = np.full((100, 100), 2.0, np.float32)
    hr_data[:10, :] = 0
    return lr_wcs, hr_wcs, hr_data


def test_hr_on_lr_grid_reprojects_onto_nested_grid():
    lr_wcs, hr_wcs, hr_data = synthetic_scene()
    hr, valid, out_wcs = pairs.hr_on_lr_grid(hr_data, hr_wcs, lr_wcs, (24, 24), F, block_size=64)
    assert hr.shape == (144, 144) and hr.dtype == np.float32
    assert valid[72, 72] and not valid[0, 0]
    assert np.isclose(hr[72, 72], 2.0 * AREA_RATIO, rtol=1e-3)
    assert np.allclose(proj_plane_pixel_scales(out_wcs) * 3600, 0.2 / F)
    ok = lr_window_mask(valid, F)
    assert ok[12, 12] and not ok[0, 0]
    assert 100 < ok.sum() < 300


def test_hr_on_lr_grid_conserves_source_flux():
    lr_wcs, hr_wcs, _ = synthetic_scene()
    yy, xx = np.mgrid[0:100, 0:100]
    source = np.exp(-((yy - 50) ** 2 + (xx - 50) ** 2) / (2 * 3.0**2)).astype(np.float32)
    hr, valid, _ = pairs.hr_on_lr_grid(source, hr_wcs, lr_wcs, (24, 24), F, block_size=64)
    assert np.isclose(hr.sum(), source.sum(), rtol=0.01)


def test_full_window_corners_matches_brute_force():
    ok = np.random.default_rng(1).random((15, 18)) > 0.3
    size = 4
    got = pairs.full_window_corners(ok, size)
    expected = np.array(
        [
            [ok[y : y + size, x : x + size].all() for x in range(18 - size + 1)]
            for y in range(15 - size + 1)
        ]
    )
    assert got.shape == expected.shape and (got == expected).all()
    assert pairs.full_window_corners(ok, 20).size == 0


def test_sample_windows_is_seeded_distinct_and_valid():
    ok = np.random.default_rng(1).random((30, 30)) > 0.2
    picked, n_valid = pairs.sample_windows(ok, 3, 5, np.random.default_rng(7))
    again, _ = pairs.sample_windows(ok, 3, 5, np.random.default_rng(7))
    assert picked == again and len(set(picked)) == 5 and n_valid >= 5
    assert all(ok[y : y + 3, x : x + 3].all() for y, x in picked)
    assert pairs.sample_windows(np.zeros((5, 5), bool), 3, 2, np.random.default_rng(0)) == ([], 0)
    assert pairs.sample_windows(ok, 3, 0, np.random.default_rng(0))[0] == []


def test_split_masks_never_share_rows():
    ok = np.ones((10, 6), bool)
    masks = pairs.split_masks(ok, 0.3)
    assert masks["train"][:7].all() and not masks["train"][7:].any()
    assert masks["val"][7:].all() and not masks["val"][:7].any()
    assert not (masks["train"] & masks["val"]).any()
    assert pairs.split_masks(ok, 0.0)["train"].all()


def test_split_masks_follows_coverage_not_image_height():
    ok = np.zeros((20, 6), bool)
    ok[2:12] = True
    masks = pairs.split_masks(ok, 0.3)
    assert masks["train"][2:9].all() and not masks["train"][9:].any()
    assert masks["val"][9:12].all() and not masks["val"][:9].any()
    assert not pairs.split_masks(np.zeros((5, 5), bool), 0.3)["val"].any()


def test_cutout_hdu_shifts_wcs_and_adds_cards():
    wcs = make_tan_wcs(0.2, (24, 24))
    hdu = pairs.cutout_hdu(np.zeros((4, 5), np.float32), wcs, 10, 3, {"LRX0": 3})
    assert hdu.header["LRX0"] == 3
    origin = WCS(hdu.header).pixel_to_world(0, 0)
    assert origin.separation(wcs.pixel_to_world(3, 10)).arcsec < 1e-6


def test_overlap_box_encloses_hr_footprint():
    lr_wcs, hr_wcs, hr_data = synthetic_scene()
    y0, y1, x0, x1 = pairs.overlap_box(lr_wcs, (24, 24), hr_wcs, hr_data.shape)
    assert 0 <= y0 < 5 and 19 < y1 <= 24 and 0 <= x0 < 5 and 19 < x1 <= 24
    far = make_tan_wcs(0.03, (100, 100), crval=(151.0, 2.0))
    assert pairs.overlap_box(lr_wcs, (24, 24), far, (100, 100)) is None


def test_process_patch_end_to_end(tmp_path):
    lr_wcs, hr_wcs, hr_data = synthetic_scene()
    lr_data = np.random.default_rng(0).normal(size=(24, 24)).astype(np.float32)
    image = fits.ImageHDU(lr_data, header=lr_wcs.to_header(), name="IMAGE")
    image.header["BUNIT"] = "nJy"
    mask_data = np.zeros((24, 24), np.int32)
    mask_data[20:, :] = 1
    mask = fits.ImageHDU(mask_data, name="MASK")
    mask.header["MSKN0000"], mask.header["MSKM0000"] = "NO_DATA", 1
    lr_path = tmp_path / "deep_coadd_1_2_i.fits"
    fits.HDUList([fits.PrimaryHDU(), image, mask]).writeto(lr_path)
    write_mosaic(tmp_path / "mosaic.fits", hr_data, hr_wcs)
    mosaic = MosaicSet([tmp_path / "mosaic.fits"])
    out = tmp_path / "pairs"
    for split in ("train", "val"):
        for kind in ("lr", "hr"):
            (out / split / kind).mkdir(parents=True)

    counts = pairs.process_patch(
        lr_path,
        mosaic,
        out,
        size=4,
        factor=F,
        per_patch=4,
        rng=np.random.default_rng(0),
        reject=["NO_DATA"],
        block_size=64,
        val_frac=0.5,
    )
    assert counts["train"] == 2 and counts["val"] == 2 and counts["valid"] >= 4

    lr = fits.open(out / "train" / "lr" / "deep_coadd_1_2_i_train_00000.fits")[0]
    hr = fits.open(out / "train" / "hr" / "deep_coadd_1_2_i_train_00000.fits")[0]
    assert lr.data.shape == (4, 4) and hr.data.shape == (24, 24)
    assert lr.header["LRY0"] + 4 <= 20
    assert lr.header["BUNIT"] == "nJy" and hr.header["BUNIT"] == "nJy"
    assert hr.header["HRSCALE"] == mosaic.njy_per_count
    lr_origin = WCS(lr.header).pixel_to_world(0, 0)
    hr_origin = WCS(hr.header).pixel_to_world((F - 1) / 2, (F - 1) / 2)
    assert lr_origin.separation(hr_origin).arcsec < 1e-6
    assert np.allclose(hr.data, 2.0 * AREA_RATIO * mosaic.njy_per_count, rtol=1e-2)
    y, x = lr.header["LRY0"], lr.header["LRX0"]
    assert np.array_equal(lr.data, lr_data[y : y + 4, x : x + 4])

    train_rows = [fits.getheader(f)["LRY0"] for f in glob.glob(str(out / "train" / "lr" / "*"))]
    val_rows = [fits.getheader(f)["LRY0"] for f in glob.glob(str(out / "val" / "lr" / "*"))]
    assert max(train_rows) + 4 <= min(val_rows)


def assert_valid_disjoint_tiling(ok, corners, size):
    occupied = np.zeros_like(ok, dtype=int)
    for y, x in corners:
        assert ok[y : y + size, x : x + size].all()
        occupied[y : y + size, x : x + size] += 1
    assert occupied.max() <= 1
    # maximal: no further window fits in the uncovered valid area
    assert not pairs.full_window_corners(ok & (occupied == 0), size).any()


def test_tile_windows_packs_a_rectangle_exactly():
    ok = np.ones((50, 70), bool)
    corners = pairs.tile_windows(ok, 10)
    assert len(corners) == 35
    assert_valid_disjoint_tiling(ok, corners, 10)
    assert pairs.tile_windows(np.ones((5, 5), bool), 10) == []


def test_tile_windows_on_an_irregular_mask():
    rng = np.random.default_rng(3)
    ok = np.ones((80, 90), bool)
    for _ in range(25):
        y, x = rng.integers(0, 80), rng.integers(0, 90)
        ok[y : y + 3, x : x + 4] = False
    corners = pairs.tile_windows(ok, 7)
    assert len(corners) > 40
    assert_valid_disjoint_tiling(ok, corners, 7)


def dec_of(wcs, shape):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return wcs.pixel_to_world_values(xx, yy)[1]


def test_sky_split_masks_follow_global_dec_stripes_with_a_guard():
    from scipy.ndimage import distance_transform_edt

    shape, scale, pad = (300, 260), 0.2, 20
    wcs = make_tan_wcs(scale, shape, crval=(150.1, 2.2))
    period, frac, guard = 0.5, 0.2, 2.0  # 30" period, 6" val stripes, 2" guard
    masks = pairs.sky_split_masks(np.ones(shape, bool), wcs, frac, period, guard)
    train, val = masks["train"], masks["val"]
    phase = np.mod(dec_of(wcs, shape) * 60 / period, 1.0)
    assert val.any() and train.any() and not (train & val).any()
    assert (val == (phase < frac)).all()
    # val stripes also just outside the frame count, so measure on a padded grid
    yy, xx = np.mgrid[-pad : shape[0] + pad, -pad : shape[1] + pad]
    val_padded = np.mod(wcs.pixel_to_world_values(xx, yy)[1] * 60 / period, 1.0) < frac
    edge_gap = ((distance_transform_edt(~val_padded) - 1) * scale)[pad:-pad, pad:-pad]
    # train pixel edges sit >= guard + 1 px from val pixel edges in the same grid (the extra pixel
    # keeps other grids' val pixels >= guard away), and no more sky than that is given up
    assert edge_gap[train].min() >= guard + scale - 1e-9
    assert edge_gap[~train & ~val].max() < guard + 2 * scale


def test_sky_split_agrees_between_overlapping_grids():
    """Two images of the same sky on different grids (like neighbouring patches or tracts) never
    put the same sky in train in one and val in the other."""
    shape, size, guard = (240, 240), 12, 2.0
    a = make_tan_wcs(0.2, shape, crval=(150.1, 2.2))
    b = make_tan_wcs(0.2, shape, crval=(150.1 + 9 / 3600, 2.2 + 13 / 3600))  # offset ~16"
    b.wcs.pc = [
        [-0.2 / 3600 * np.cos(0.3), 0.2 / 3600 * np.sin(0.3)],
        [0.2 / 3600 * np.sin(0.3), 0.2 / 3600 * np.cos(0.3)],
    ]  # and rotated
    ok = np.ones(shape, bool)
    masks_a = pairs.sky_split_masks(ok, a, 0.25, 0.4, guard)
    masks_b = pairs.sky_split_masks(ok, b, 0.25, 0.4, guard)
    rng = np.random.default_rng(0)
    train, _ = pairs.sample_windows(masks_a["train"], size, 400, rng)
    val = pairs.tile_windows(masks_b["val"], size)
    assert train and val

    def quads(wcs, corners):
        headers = [wcs[y : y + size, x : x + size].to_header() for y, x in corners]
        for h in headers:
            h["NAXIS1"] = h["NAXIS2"] = size
        return np.array([leakage.footprint(h) for h in headers])

    plane = leakage.tangent_plane(np.concatenate([quads(a, train), quads(b, val)]))
    assert leakage.conflicts(plane[: len(train)], plane[len(train) :], guard - 0.01) == []


def test_process_patch_sky_split_tiles_val_and_densely_samples_train(tmp_path):
    lr_wcs = make_tan_wcs(0.2, (60, 60))
    center = lr_wcs.pixel_to_world(29.5, 29.5)
    hr_wcs = make_tan_wcs(0.03, (420, 420), crval=(center.ra.deg, center.dec.deg))
    lr_data = np.random.default_rng(0).normal(size=(60, 60)).astype(np.float32)
    image = fits.ImageHDU(lr_data, header=lr_wcs.to_header(), name="IMAGE")
    image.header["BUNIT"] = "nJy"
    mask = fits.ImageHDU(np.zeros((60, 60), np.int32), name="MASK")
    mask.header["MSKN0000"], mask.header["MSKM0000"] = "NO_DATA", 1
    lr_path = tmp_path / "deep_coadd_1_2_i.fits"
    fits.HDUList([fits.PrimaryHDU(), image, mask]).writeto(lr_path)
    write_mosaic(tmp_path / "mosaic.fits", np.full((420, 420), 2.0, np.float32), hr_wcs)
    out = tmp_path / "pairs"
    for split in ("train", "val"):
        for kind in ("lr", "hr"):
            (out / split / kind).mkdir(parents=True)
    counts = pairs.process_patch(
        lr_path,
        MosaicSet([tmp_path / "mosaic.fits"]),
        out,
        size=4,
        factor=F,
        per_patch=0,
        rng=np.random.default_rng(0),
        reject=["NO_DATA"],
        block_size=128,
        val_frac=0.3,
        split_mode="sky",
        val_period=0.1,  # 6" period: 1.8" val stripes, two per 12" frame
        guard=0.2,
        train_density=2.0,
        val_tile=True,
    )
    assert counts["train"] > 0 and counts["val"] > 0
    corners = {}
    for split in ("train", "val"):
        corners[split] = [
            (fits.getheader(p)["LRY0"], fits.getheader(p)["LRX0"])
            for p in (out / split / "lr").glob("*.fits")
        ]
    occupied = np.zeros((60, 60), int)
    for y, x in corners["val"]:
        occupied[y : y + 4, x : x + 4] += 1
    assert occupied.max() == 1  # val windows tile without overlap
    train_rows = {y + k for y, _ in corners["train"] for k in range(4)}
    val_rows = {y + k for y, _ in corners["val"] for k in range(4)}
    assert min(abs(a - b) for a in train_rows for b in val_rows) >= 2  # guard 1 px + slack


def test_main_refuses_to_mix_with_existing_pairs(tmp_path):
    import pytest

    _, hr_wcs, hr_data = synthetic_scene()
    write_mosaic(tmp_path / "mosaic.fits", hr_data, hr_wcs)
    (tmp_path / "lr").mkdir()
    stale = tmp_path / "pairs" / "train" / "lr"
    stale.mkdir(parents=True)
    fits.PrimaryHDU(np.zeros((2, 2), np.float32)).writeto(stale / "old.fits")
    with pytest.raises(SystemExit, match="already holds pairs"):
        pairs.main(
            [
                "--lr-dir",
                str(tmp_path / "lr"),
                "--hr-mosaic",
                str(tmp_path / "mosaic.fits"),
                "--out",
                str(tmp_path / "pairs"),
            ]
        )


def write_scene(tmp_path, names=("deep_coadd_1_1_i", "deep_coadd_1_2_i")):
    lr_wcs = make_tan_wcs(0.2, (60, 60))
    center = lr_wcs.pixel_to_world(29.5, 29.5)
    hr_wcs = make_tan_wcs(0.03, (420, 420), crval=(center.ra.deg, center.dec.deg))
    (tmp_path / "lr").mkdir()
    for k, name in enumerate(names):
        data = np.random.default_rng(k).normal(size=(60, 60)).astype(np.float32)
        image = fits.ImageHDU(data, header=lr_wcs.to_header(), name="IMAGE")
        image.header["BUNIT"] = "nJy"
        mask = fits.ImageHDU(np.zeros((60, 60), np.int32), name="MASK")
        mask.header["MSKN0000"], mask.header["MSKM0000"] = "NO_DATA", 1
        mask.header["MSKN0001"], mask.header["MSKM0001"] = "SATURATED", 2
        fits.HDUList([fits.PrimaryHDU(), image, mask]).writeto(tmp_path / "lr" / f"{name}.fits")
    write_mosaic(tmp_path / "mosaic.fits", np.full((420, 420), 2.0, np.float32), hr_wcs)


def run_pairs(tmp_path, out, *extra):
    pairs.main(
        [
            "--lr-dir", str(tmp_path / "lr"), "--hr-mosaic", str(tmp_path / "mosaic.fits"),
            "--out", str(out), "--lr-size", "4", "--block-size", "128", "--split", "sky",
            "--val-frac", "0.3", "--val-period", "0.1", "--guard", "0.2",
            "--train-density", "2", "--val-tile", *extra,
        ]
    )  # fmt: skip


def snapshot(out):
    return {
        str(p.relative_to(out)): fits.getdata(p).tobytes() for p in sorted(out.glob("*/*/*.fits"))
    }


def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    write_scene(tmp_path)
    run_pairs(tmp_path, tmp_path / "full")
    full = snapshot(tmp_path / "full")
    assert any("1_1" in k for k in full) and any("1_2" in k for k in full)

    # interrupted run: the second image never finished and left a stray file behind
    run_pairs(tmp_path, tmp_path / "cut")
    cut = tmp_path / "cut"
    (cut / ".done" / "deep_coadd_1_2_i.json").unlink()
    for p in cut.glob("*/*/deep_coadd_1_2_i_*.fits"):
        p.unlink()
    stray = cut / "train" / "lr" / "deep_coadd_1_2_i_train_99999.fits"
    fits.PrimaryHDU(np.zeros((4, 4), np.float32)).writeto(stray)
    with pytest.raises(SystemExit, match="--resume"):
        run_pairs(tmp_path, cut)
    run_pairs(tmp_path, cut, "--resume")
    assert snapshot(cut) == full and not stray.exists()


def test_resume_refuses_changed_settings(tmp_path):
    write_scene(tmp_path, names=("deep_coadd_1_1_i",))
    run_pairs(tmp_path, tmp_path / "out")
    with pytest.raises(SystemExit, match="guard"):
        run_pairs(tmp_path, tmp_path / "out", "--resume", "--guard", "0.4")
