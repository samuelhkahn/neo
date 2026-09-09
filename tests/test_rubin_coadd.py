import numpy as np
import pytest
from astropy.io import fits
from conftest import make_tan_wcs

from neo.surveys.rubin.coadd import Coadd, flagged_pixels, load_coadd, mask_bits_from_header


def test_mask_bits_from_header():
    header = fits.Header()
    header["MSKN0000"], header["MSKM0000"], header["MSKD0000"] = "NO_DATA", 1, "desc"
    header["MSKN0003"], header["MSKM0003"] = "SATURATED", 8
    assert mask_bits_from_header(header) == {"NO_DATA": 1, "SATURATED": 8}


def test_flagged_pixels_combines_planes():
    coadd = Coadd(
        image=np.zeros((2, 2), np.float32),
        mask=np.array([[0, 1], [8, 9]], np.int32),
        wcs=None,
        mask_bits={"NO_DATA": 1, "SATURATED": 8},
        bunit="nJy",
    )
    assert flagged_pixels(coadd, ["NO_DATA"]).tolist() == [[False, True], [False, True]]
    assert flagged_pixels(coadd, ["NO_DATA", "SATURATED"]).tolist() == [[False, True], [True, True]]
    with pytest.raises(KeyError):
        flagged_pixels(coadd, ["BOGUS"])


def test_load_coadd(tmp_path):
    wcs = make_tan_wcs(0.2, (10, 12))
    image = fits.ImageHDU(np.ones((10, 12), np.float32), header=wcs.to_header(), name="IMAGE")
    image.header["BUNIT"] = "nJy"
    mask = fits.ImageHDU(np.zeros((10, 12), np.int32), name="MASK")
    mask.header["MSKN0000"], mask.header["MSKM0000"] = "NO_DATA", 1
    fits.HDUList([fits.PrimaryHDU(), image, mask]).writeto(tmp_path / "c.fits")

    coadd = load_coadd(tmp_path / "c.fits")
    assert coadd.image.shape == (10, 12) and coadd.mask.dtype == np.int32
    assert coadd.mask_bits == {"NO_DATA": 1} and coadd.bunit == "nJy"
    assert coadd.wcs.pixel_to_world(0, 0).separation(wcs.pixel_to_world(0, 0)).arcsec < 1e-6
