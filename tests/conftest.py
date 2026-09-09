from astropy.wcs import WCS


def make_tan_wcs(scale_arcsec, shape, crval=(150.0, 2.0), cd=False) -> WCS:
    """North-up TAN WCS for an image of `shape` (ny, nx) centered on `crval`."""
    ny, nx = shape
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(crval)
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    s = scale_arcsec / 3600
    if cd:
        w.wcs.cd = [[-s, 0], [0, s]]
    else:
        w.wcs.pc = [[-s, 0], [0, s]]
        w.wcs.cdelt = [1, 1]
    w.pixel_shape = (nx, ny)
    return w
