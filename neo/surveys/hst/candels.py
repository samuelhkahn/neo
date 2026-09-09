"""The CANDELS COSMOS F814W mosaic: one large TAN-projected FITS image in counts/s, 0 = no data."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def ab_zeropoint(photflam: float, photplam: float) -> float:
    """AB magnitude zeropoint for an HST image in counts/s (STScI PHOTFLAM/PHOTPLAM convention)."""
    return -2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408


def njy_per_count(zeropoint_ab: float) -> float:
    """Flux density in nJy corresponding to 1 count/s at the given AB zeropoint."""
    return 1e9 * 10 ** (-0.4 * (zeropoint_ab - 8.90))


class CandelsMosaic:
    bunit = "count/s"

    def __init__(self, path: str | Path):
        self._hdul = fits.open(path, memmap=True)
        self.hdu = self._hdul[0]
        self.wcs = WCS(self.hdu.header)
        self.shape = self.hdu.shape
        self.zeropoint = ab_zeropoint(self.hdu.header["PHOTFLAM"], self.hdu.header["PHOTPLAM"])
        self.njy_per_count = njy_per_count(self.zeropoint)

    def close(self) -> None:
        self._hdul.close()

    def region_slices(self, ra_min, ra_max, dec_min, dec_max) -> tuple[slice, slice] | None:
        """Array slices covering a sky box, or None if the box misses the mosaic entirely."""
        ras = [ra_min, ra_max, ra_max, ra_min]
        decs = [dec_min, dec_min, dec_max, dec_max]
        x, y = self.wcs.all_world2pix(ras, decs, 0)
        ny, nx = self.shape
        x0 = int(np.clip(np.floor(x.min()), 0, nx))
        x1 = int(np.clip(np.ceil(x.max()) + 1, 0, nx))
        y0 = int(np.clip(np.floor(y.min()), 0, ny))
        y1 = int(np.clip(np.ceil(y.max()) + 1, 0, ny))
        if x1 <= x0 or y1 <= y0:
            return None
        return slice(y0, y1), slice(x0, x1)

    def region(self, ra_min, ra_max, dec_min, dec_max) -> tuple[np.ndarray, WCS] | None:
        """Sub-image and matching WCS for a sky box; only that part of the file is read."""
        slices = self.region_slices(ra_min, ra_max, dec_min, dec_max)
        if slices is None:
            return None
        ys, xs = slices
        return np.asarray(self.hdu.data[ys, xs], dtype=np.float32), self.wcs[ys, xs]

    @staticmethod
    def valid(data: np.ndarray) -> np.ndarray:
        return np.isfinite(data) & (data != 0)
