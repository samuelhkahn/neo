import numpy as np
import pytest
from astropy.io import fits
from conftest import make_tan_wcs

from neo.surveys.hst.candels import CandelsMosaic, ab_zeropoint, njy_per_count

# Values from the CANDELS COSMOS F814W mosaic header.
PHOTFLAM, PHOTPLAM = 7.0331885e-20, 8056.9424


def test_f814w_zeropoint_and_flux_scale():
    zp = ab_zeropoint(PHOTFLAM, PHOTPLAM)
    assert zp == pytest.approx(25.94, abs=0.01)
    assert njy_per_count(zp) == pytest.approx(152, abs=1)
    assert njy_per_count(31.4) == pytest.approx(1.0)


def write_mosaic(path, data, wcs):
    header = wcs.to_header()
    header["PHOTFLAM"], header["PHOTPLAM"] = PHOTFLAM, PHOTPLAM
    fits.PrimaryHDU(data=data, header=header).writeto(path)


def test_region_reads_only_the_sky_box(tmp_path):
    ny, nx = 200, 300
    data = (np.arange(ny)[:, None] * 1000 + np.arange(nx)[None, :]).astype(np.float32)
    write_mosaic(tmp_path / "m.fits", data, make_tan_wcs(0.03, (ny, nx)))
    mosaic = CandelsMosaic(tmp_path / "m.fits")
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
    assert CandelsMosaic.valid(data).tolist() == [[True, False], [False, True]]
