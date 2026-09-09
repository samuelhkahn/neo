"""Load a Rubin deep_coadd FITS as plain arrays with its WCS and mask-plane definitions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


@dataclass
class Coadd:
    image: np.ndarray
    mask: np.ndarray
    wcs: WCS
    mask_bits: dict[str, int]
    bunit: str | None


def mask_bits_from_header(header: fits.Header) -> dict[str, int]:
    """Rubin writes each mask plane as a MSKNxxxx (name) / MSKMxxxx (bit value) card pair."""
    return {header[key]: int(header["MSKM" + key[4:]]) for key in header if key.startswith("MSKN")}


def load_coadd(path: str | Path) -> Coadd:
    with fits.open(path) as hdul:
        image_hdu, mask_hdu = hdul["IMAGE"], hdul["MASK"]
        return Coadd(
            image=np.asarray(image_hdu.data, dtype=np.float32),
            mask=np.asarray(mask_hdu.data, dtype=np.int32),
            wcs=WCS(image_hdu.header),
            mask_bits=mask_bits_from_header(mask_hdu.header),
            bunit=image_hdu.header.get("BUNIT"),
        )


def flagged_pixels(coadd: Coadd, plane_names) -> np.ndarray:
    bits = 0
    for name in plane_names:
        if name not in coadd.mask_bits:
            raise KeyError(f"unknown mask plane {name!r}; available: {sorted(coadd.mask_bits)}")
        bits |= coadd.mask_bits[name]
    return (coadd.mask & bits) != 0
