import glob

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from conftest import make_tan_wcs
from test_hst_mosaic import write_mosaic

from neo.preprocess import pairs
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
