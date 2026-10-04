"""Per-source morphology catalogs for an (HST, SR..., LR) cutout set, as in the NEO paper.

Port of the paper's final catalog pipeline (`Create Catalogs .ipynb`, July 2025, with
`jades_photutils_interface.py`): sources are detected and deblended on the HST image only, and that
one segmentation map is used to measure every super-resolved image and, reprojected to the coarse
grid, the low-resolution image. Every model is therefore measured on exactly the same sources.
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

# Paper detection threshold on HST F814W in counts/s per 0.028" pixel (CreateSourceCatalog default).
PAPER_THRESHOLD_CPS = 0.00691259209997952
PAPER_HR_PIXEL_ARCSEC = 0.168 / 6
NPIXELS = 100

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
    """Paper threshold in the pairs' units: counts/s -> nJy, 0.028" -> our HR pixel area."""
    return PAPER_THRESHOLD_CPS * njy_per_count * (hr_pixel_arcsec / PAPER_HR_PIXEL_ARCSEC) ** 2


def create_convolved_data(data: np.ndarray) -> np.ndarray:
    """Smooth with a FWHM = 3 px Gaussian on a 3x3 kernel (NaN -> 0), as in CreateConvolvedData."""
    kernel = Gaussian2DKernel(3.0 * gaussian_fwhm_to_sigma, x_size=3, y_size=3)
    data_zeros = np.where(np.isnan(data), 0, data)
    return convolve(data_zeros, kernel, normalize_kernel=True)


def detect_hst(data: np.ndarray, threshold: float, npixels: int = NPIXELS):
    """Detect + deblend on HST (CreateSourceCatalog); returns (segm, convolved) or None."""
    convolved = create_convolved_data(data)
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


def catalog_set(hst, srs: dict, lr, threshold: float, factor: int = 6, npixels: int = NPIXELS):
    """Catalog one cutout set; returns (hst_table, {model: table}, lr_table) or None.

    `hst` and each SR image must already be background subtracted; `lr` is used as-is. As in the
    paper, a set is kept only when the reprojected LR catalog has as many sources as HST.
    """
    detected = detect_hst(hst, threshold, npixels)
    if detected is None:
        return None
    segm, convolved = detected
    hst_tbl = measure(SourceCatalog(hst, segm, convolved_data=convolved))
    segm_lr = lr_segmap(segm, lr.shape, factor)
    if segm_lr is None:
        return None
    lr_tbl = measure(SourceCatalog(lr, segm_lr))
    if len(lr_tbl) != len(hst_tbl):
        return None
    sr_tbls = {
        name: measure(SourceCatalog(sr, segm, convolved_data=create_convolved_data(sr)))
        for name, sr in srs.items()
    }
    return hst_tbl, sr_tbls, lr_tbl
