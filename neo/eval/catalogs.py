"""Per-source morphology catalogs for an (HST, SR..., LR) cutout set, as in the NEO paper.

Port of the paper's final catalog pipeline (`Create Catalogs .ipynb`, July 2025, with
`jades_photutils_interface.py`): sources are detected and deblended on the HST image only, and that
one segmentation map is used to measure every super-resolved image and, reprojected to the coarse
grid, the low-resolution image. Every model is therefore measured on exactly the same sources.
The published Table 4 rows reproduce only with LR shapes taken from the LR image's smoothed copy,
as in the 2023 copy of that notebook (the July 2025 copy comments it out): catalog_set(lr_fwhm=...).
"""

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table
from photutils.datasets import make_wcs
from photutils.segmentation import (
    SegmentationImage,
    SourceCatalog,
    deblend_sources,
    detect_sources,
)
from reproject import reproject_interp

# Paper detection threshold (CreateSourceCatalog default) on HST F814W in e-/s per 0.03" pixel:
# the paper resampled CANDELS (0.03") onto its 0.028" grid preserving surface brightness, so this
# is a surface brightness per native pixel (the paper's Create Catalogs notebook, CANDELS e-/s).
PAPER_THRESHOLD_CPS = 0.00691259209997952
PAPER_THRESHOLD_PIXEL_ARCSEC = 0.03
# The paper's catalog grid: HSC 0.168" pixels and the HST grid nested 6x finer (0.028").
PAPER_LR_PIXEL_ARCSEC = 0.168
PAPER_HR_PIXEL_ARCSEC = PAPER_LR_PIXEL_ARCSEC / 6
NPIXELS = 100  # minimum source area in pixels (detection and deblending)
KERNEL_FWHM = 3.0  # detection smoothing FWHM in pixels (CreateConvolvedData)

# Same property set as the paper's COLUMNS (minus the sky_bbox_* entries, which need a WCS and
# feed no metric); `label` is kept to join sources across images.
COLUMNS = [
    "label",
    "xcentroid",
    "ycentroid",
    "area",
    "semimajor_sigma",
    "semiminor_sigma",
    "elongation",
    "orientation",
    "eccentricity",
    "ellipticity",
    "min_value",
    "max_value",
    "segment_flux",
    "kron_flux",
    "kron_radius",
    "gini",
    "fwhm",
    "cxx",
    "cxy",
    "cyy",
]


def default_threshold(njy_per_count: float, hr_pixel_arcsec: float) -> float:
    """Paper threshold in nJy per HR pixel: e-/s -> nJy, per 0.03" pixel -> per HR pixel.

    For --units paper pairs this equals PAPER_THRESHOLD_CPS * NJYPERPX of the HR cutout.
    """
    return (
        PAPER_THRESHOLD_CPS * njy_per_count * (hr_pixel_arcsec / PAPER_THRESHOLD_PIXEL_ARCSEC) ** 2
    )


def paper_npixels(hr_pixel_arcsec: float) -> int:
    """The paper's minimum area (100 px at 0.028") in HR pixels of the same sky area."""
    return int(round(NPIXELS * (PAPER_HR_PIXEL_ARCSEC / hr_pixel_arcsec) ** 2))


def kernel_width(fwhm: float) -> float:
    """RMS width (px, per axis) of the 3x3 Gaussian kernel create_convolved_data uses."""
    kernel = Gaussian2DKernel(fwhm * gaussian_fwhm_to_sigma, x_size=3, y_size=3).array
    kernel = kernel / kernel.sum()
    return float(np.sqrt((kernel.sum(axis=0) * np.array([1.0, 0.0, 1.0])).sum()))


def paper_fwhm(pixel_arcsec: float, paper_pixel_arcsec: float = PAPER_HR_PIXEL_ARCSEC) -> float:
    """FWHM (px of `pixel_arcsec`) whose 3x3 kernel smooths the sky as much as the paper's did.

    The paper smoothed with FWHM 3 px on a 3x3 kernel, so truncation sets the width: matching the
    kernels' RMS width on the sky (not the nominal FWHM) is what reproduces its detection.
    """
    target = kernel_width(KERNEL_FWHM) * paper_pixel_arcsec / pixel_arcsec
    lo, hi = 0.05, 50.0
    if not kernel_width(lo) < target < kernel_width(hi):
        raise ValueError(f"no 3x3 Gaussian kernel is {target:.3f} px wide")
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if kernel_width(mid) < target else (lo, mid)
    return (lo + hi) / 2


def create_convolved_data(data: np.ndarray, fwhm: float = KERNEL_FWHM) -> np.ndarray:
    """Smooth with a Gaussian on a 3x3 kernel (NaN -> 0), as in CreateConvolvedData (FWHM 3 px)."""
    kernel = Gaussian2DKernel(fwhm * gaussian_fwhm_to_sigma, x_size=3, y_size=3)
    data_zeros = np.where(np.isnan(data), 0, data)
    return convolve(data_zeros, kernel, normalize_kernel=True)


def detect_hst(
    data: np.ndarray, threshold: float, npixels: int = NPIXELS, fwhm: float = KERNEL_FWHM
):
    """Detect + deblend on HST (CreateSourceCatalog); returns (segm, convolved) or None."""
    convolved = create_convolved_data(data, fwhm)
    segm = detect_sources(convolved, threshold, npixels=npixels)
    if segm is None:
        return None
    segm_deblend = deblend_sources(
        convolved, segm, npixels=npixels, nlevels=32, contrast=0.001, progress_bar=False
    )
    return segm_deblend, convolved


def lr_segmap(hr_segm: SegmentationImage, lr_shape, factor: int) -> SegmentationImage | None:
    """Nearest-neighbour reprojection of the HST segmap onto the LR grid, using the paper's WCSs."""
    hr_header = make_wcs(hr_segm.data.shape).to_header()
    lr_header = make_wcs(lr_shape).to_header()
    for key in ("PC1_1", "PC1_2", "PC2_1", "PC2_2"):
        lr_header[key] *= factor
    seg_lr = reproject_interp(
        fits.PrimaryHDU(hr_segm.data, header=hr_header),
        lr_header,
        shape_out=lr_shape,
        order="nearest-neighbor",
        return_footprint=False,
    )
    seg_lr = np.nan_to_num(seg_lr).astype(int)
    if not seg_lr.any():
        return None
    return SegmentationImage(seg_lr)


def measure(cat: SourceCatalog) -> Table:
    """Paper columns plus half-light radius and C75/25, C90/50 concentrations, as plain floats."""
    tbl = cat.to_table(columns=COLUMNS)
    out = Table()
    for name in COLUMNS:
        col = tbl[name]
        out[name] = np.asarray(getattr(col, "value", col), dtype=float)
    radius = getattr(cat, "flux_radius", None) or cat.fluxfrac_radius  # renamed in photutils 3.0
    r = {f: np.asarray(radius(f).value, dtype=float) for f in (0.25, 0.5, 0.75, 0.9)}
    out["half_light_radius"] = r[0.5]
    out["flux_concentration_75_25"] = r[0.75] / r[0.25]
    out["flux_concentration_90_50"] = r[0.9] / r[0.5]
    return out


def catalog_set(
    hst,
    srs: dict,
    lr,
    threshold: float,
    factor: int = 6,
    npixels: int = NPIXELS,
    fwhm: float = KERNEL_FWHM,
    lr_fwhm: float | None = None,
):
    """Catalog one cutout set; returns (hst_table, {model: table}, lr_table) or None.

    Images are measured as given (background subtraction or clipping is the caller's). HST and
    SR shapes come from images smoothed with `fwhm`; LR shapes from the LR image itself, or from
    its copy smoothed with `lr_fwhm` (as the code behind the paper's Table 4 did). As in the
    paper, a set is kept only when the reprojected LR catalog has as many sources as HST.
    """
    detected = detect_hst(hst, threshold, npixels, fwhm)
    if detected is None:
        return None
    segm, convolved = detected
    hst_tbl = measure(SourceCatalog(hst, segm, convolved_data=convolved))
    segm_lr = lr_segmap(segm, lr.shape, factor)
    if segm_lr is None:
        return None
    lr_conv = None if lr_fwhm is None else create_convolved_data(lr, lr_fwhm)
    lr_tbl = measure(SourceCatalog(lr, segm_lr, convolved_data=lr_conv))
    if len(lr_tbl) != len(hst_tbl):
        return None
    sr_tbls = {
        name: measure(SourceCatalog(sr, segm, convolved_data=create_convolved_data(sr, fwhm)))
        for name, sr in srs.items()
    }
    return hst_tbl, sr_tbls, lr_tbl
