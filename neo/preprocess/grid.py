"""Pixel grids for paired low/high resolution images."""

import numpy as np
from astropy.wcs import WCS


def upsampled_wcs(lr_wcs: WCS, factor: int) -> WCS:
    """WCS of a grid `factor` times finer than `lr_wcs` whose pixels nest exactly in the LR pixels.

    LR pixel (i, j) (0-based) covers HR pixels [factor*i, factor*i + factor) in each axis.
    """
    hr = lr_wcs.deepcopy()
    hr.wcs.crpix = (lr_wcs.wcs.crpix - 0.5) * factor + 0.5
    if lr_wcs.wcs.has_cd():
        hr.wcs.cd = lr_wcs.wcs.cd / factor
    else:
        hr.wcs.cdelt = lr_wcs.wcs.cdelt / factor
    if lr_wcs.pixel_shape is not None:
        hr.pixel_shape = tuple(n * factor for n in lr_wcs.pixel_shape)
    return hr


def lr_window_mask(hr_valid: np.ndarray, factor: int) -> np.ndarray:
    """True for LR pixels whose entire factor x factor block of HR pixels is valid."""
    ny, nx = hr_valid.shape
    return hr_valid.reshape(ny // factor, factor, nx // factor, factor).all(axis=(1, 3))
