import numpy as np
import pytest
from astropy.io import fits
from conftest import make_tan_wcs

from neo.surveys.hst.mosaic import (
    F814W_AB_ZEROPOINT,
    HstMosaic,
    MosaicSet,
    ab_zeropoint,
    njy_per_count,
)

# Values from the CANDELS COSMOS F814W mosaic header.
PHOTFLAM, PHOTPLAM = 7.0331885e-20, 8056.9424


def test_f814w_zeropoint_and_flux_scale():
    zp = ab_zeropoint(PHOTFLAM, PHOTPLAM)
    assert zp == pytest.approx(25.94, abs=0.01)
    assert njy_per_count(zp) == pytest.approx(152, abs=1)
    assert njy_per_count(31.4) == pytest.approx(1.0)


def write_mosaic(path, data, wcs, photflam=PHOTFLAM):
    header = wcs.to_header()
    header["PHOTFLAM"], header["PHOTPLAM"] = photflam, PHOTPLAM
    fits.PrimaryHDU(data=data, header=header).writeto(path)


def test_region_reads_only_the_sky_box(tmp_path):
    ny, nx = 200, 300
    data = (np.arange(ny)[:, None] * 1000 + np.arange(nx)[None, :]).astype(np.float32)
    write_mosaic(tmp_path / "m.fits", data, make_tan_wcs(0.03, (ny, nx)))
    mosaic = HstMosaic(tmp_path / "m.fits")
    assert mosaic.njy_per_count == pytest.approx(152, abs=1)

    sky = mosaic.wcs.pixel_to_world([50, 120], [30, 90])
    ra, dec = sky.ra.deg, sky.dec.deg
    sub, sub_wcs = mosaic.region(ra.min(), ra.max(), dec.min(), dec.max())

    y0, x0 = divmod(int(sub[0, 0]), 1000)
    assert 28 <= y0 <= 30 and 48 <= x0 <= 50
    assert 61 <= sub.shape[0] <= 65 and 71 <= sub.shape[1] <= 75
    full = mosaic.wcs.pixel_to_world(x0 + 5, y0 + 7)
    assert full.separation(sub_wcs.pixel_to_world(5, 7)).arcsec < 1e-6

    assert mosaic.region(ra.min() + 10, ra.max() + 10, dec.min(), dec.max()) is None
    mosaic.close()


def test_valid_excludes_zeros_and_nans():
    data = np.array([[1.0, 0.0], [np.nan, -2.0]], np.float32)
    assert HstMosaic.valid(data).tolist() == [[True, False], [False, True]]


def test_zeropoint_override_for_headers_without_photflam(tmp_path):
    """COSMOS-Web tiles have no PHOTFLAM/PHOTPLAM (units e/s); an explicit zeropoint is required."""
    wcs = make_tan_wcs(0.03, (50, 50))
    fits.PrimaryHDU(np.ones((50, 50), np.float32), header=wcs.to_header()).writeto(
        tmp_path / "cw.fits"
    )

    with pytest.raises(ValueError, match="PHOTFLAM"):
        HstMosaic(tmp_path / "cw.fits")

    mosaic = HstMosaic(tmp_path / "cw.fits", zeropoint=F814W_AB_ZEROPOINT)
    assert mosaic.zeropoint == F814W_AB_ZEROPOINT
    assert mosaic.njy_per_count == pytest.approx(152, abs=1)
    mosaic.close()

    tiles = MosaicSet([tmp_path / "cw.fits"], zeropoint=F814W_AB_ZEROPOINT)
    assert tiles.njy_per_count == pytest.approx(152, abs=1)
    tiles.close()


def two_tiles(tmp_path, photflam_right=PHOTFLAM):
    """Two 100x100 tiles at 0.03\"/px side by side in RA, sharing a zeropoint unless told not to."""
    left = make_tan_wcs(0.03, (100, 100), crval=(150.0, 2.0))
    right_center = left.pixel_to_world(-50.5, 49.5)  # 100 px west... i.e. adjacent in x
    right = make_tan_wcs(0.03, (100, 100), crval=(right_center.ra.deg, right_center.dec.deg))
    write_mosaic(tmp_path / "left.fits", np.full((100, 100), 1.0, np.float32), left)
    write_mosaic(
        tmp_path / "right.fits", np.full((100, 100), 2.0, np.float32), right, photflam_right
    )
    return left, right


def test_mosaic_set_finds_overlapping_tiles(tmp_path):
    left, right = two_tiles(tmp_path)
    tiles = MosaicSet([tmp_path / "left.fits", tmp_path / "right.fits"])
    assert tiles.njy_per_count == pytest.approx(152, abs=1)

    sky = left.pixel_to_world([10, 40], [10, 40])
    box = (sky.ra.deg.min(), sky.ra.deg.max(), sky.dec.deg.min(), sky.dec.deg.max())
    assert [t.path.name for t in tiles.overlapping(*box)] == ["left.fits"]
    assert len(tiles.regions(*box)) == 1

    sky = left.pixel_to_world([-30, 30], [10, 40])
    box = (sky.ra.deg.min(), sky.ra.deg.max(), sky.dec.deg.min(), sky.dec.deg.max())
    assert len(tiles.regions(*box)) == 2
    tiles.close()


def test_mosaic_set_rejects_mixed_zeropoints(tmp_path):
    two_tiles(tmp_path, photflam_right=PHOTFLAM * 2)
    with pytest.raises(ValueError):
        MosaicSet([tmp_path / "left.fits", tmp_path / "right.fits"])
