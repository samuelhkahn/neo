"""HST drizzled mosaics: TAN-projected FITS images in counts/s with 0 marking no data."""

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


# ACS/WFC F814W AB zeropoint; needed for mosaics whose headers omit PHOTFLAM/PHOTPLAM
# (e.g. COSMOS-Web DR1, units electron/s).
F814W_AB_ZEROPOINT = 25.94


class HstMosaic:
    bunit = "count/s"

    def __init__(self, path: str | Path, zeropoint: float | None = None):
        self.path = Path(path)
        self._hdul = fits.open(path, memmap=True)
        self.hdu = self._hdul[0]
        self.wcs = WCS(self.hdu.header)
        self.shape = self.hdu.shape
        if zeropoint is not None:
            self.zeropoint = zeropoint
        elif "PHOTFLAM" in self.hdu.header and "PHOTPLAM" in self.hdu.header:
            self.zeropoint = ab_zeropoint(self.hdu.header["PHOTFLAM"], self.hdu.header["PHOTPLAM"])
        else:
            raise ValueError(
                f"{self.path.name} has no PHOTFLAM/PHOTPLAM; pass an explicit zeropoint "
                f"(F814W AB = {F814W_AB_ZEROPOINT})"
            )
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


class MosaicSet:
    """One or more mosaics (e.g. IRSA tiles) treated as a single sky coverage."""

    bunit = HstMosaic.bunit

    def __init__(self, paths, zeropoint: float | None = None):
        self.mosaics = [HstMosaic(p, zeropoint=zeropoint) for p in paths]
        if not self.mosaics:
            raise ValueError("MosaicSet needs at least one mosaic")
        zeropoints = [m.zeropoint for m in self.mosaics]
        if max(zeropoints) - min(zeropoints) > 1e-3:
            raise ValueError(f"mosaics have different zeropoints: {zeropoints}")
        self.zeropoint = zeropoints[0]
        self.njy_per_count = self.mosaics[0].njy_per_count

    def close(self) -> None:
        for m in self.mosaics:
            m.close()

    def overlapping(self, ra_min, ra_max, dec_min, dec_max) -> list[HstMosaic]:
        return [m for m in self.mosaics if m.region_slices(ra_min, ra_max, dec_min, dec_max)]

    def regions(self, ra_min, ra_max, dec_min, dec_max) -> list[tuple[np.ndarray, WCS]]:
        out = []
        for m in self.mosaics:
            region = m.region(ra_min, ra_max, dec_min, dec_max)
            if region is not None:
                out.append(region)
        return out
