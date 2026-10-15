"""Source realism of super-resolved images: sources HST does not show (ghosts), and ones it misses.

The Table-4 comparison (neo.eval.catalogs.catalog_set) detects sources on HST only and measures
every SR image inside that one segmentation map, so a source a model invents in empty sky changes
none of its numbers. Here each SR image is detected on its own, with exactly its pair's HST
detection (same image preparation, threshold in nJy per HR pixel, minimum area, smoothing and
deblending: catalogs.detect_hst), and its sources are matched to HST's by position, with
hysteresis: a counterpart in the other image is looked for down to half the threshold and a quarter
of the minimum area (the relaxed cut, relaxed()).
  footprint   HST's segmentation maps at the full and the relaxed cut, dilated by a margin
              (default 0.2", a disk)
  ghost       an SR detection (full cut) whose centroid lies outside the footprint: HST shows no
              source there even at the relaxed cut; otherwise it is matched
  recovered   an HST source (full cut) with an SR detection centroid, at the full or the relaxed
              cut, inside its own segment dilated by the same margin
  sky excess  sum of SR - HST over the pixels outside the footprint, in nJy, on the images before
              default mode's per-image SEP background subtraction (which would take out flux spread
              over its 64 px mesh, and all of a uniform offset): flux invented (or lost) in empty
              sky, including flux too faint or too spread out to be detected
Over cutouts, per model: purity (matched / all SR detections), ghosts per cutout and per arcmin^2 of
sky (outside the footprint, where ghosts can lie), completeness (all HST sources, and per HST kron
magnitude bin) and the median sky excess per arcmin^2 of sky, each with a bootstrap 95% CI that
resamples whole cutouts, or whole groups of overlapping cutouts when the split has groups.csv
(neo.eval.subsets.load_groups): overlapping cutouts show the same sources, so they are not
independent.

Why hysteresis: the full cut is only ~4.4 sky rms per pixel, and many real HST sources sit just
under it. Matched at that cut alone, on the 50 local val cutouts, an exact HST copy made 10%
brighter had 0.24 ghosts per cutout (0.56 in paper mode), in default mode all of them real sources
HST shows at white-noise S/N 55-190, and one made 10% fainter lost 8% of HST's sources: flux and
size biases, which Table 4 already measures, read as invented or missing sources. With hysteresis,
copies scaled by 0.9-1.1, blurred by 1 px, re-noised or clipped at 0 give no ghost and lose no
source in either mode. The price: the relaxed footprint covers ~4.3% of a cutout (5.5% in paper
mode) against ~1.7% for the full cut, so an invented source that lands on a faint real HST source
counts as matched: of 23.5 mag ghosts placed at random in the sky outside the full-cut footprint,
95-96% stay ghosts. Biases on sources further below the cut, and noise, can still make ghosts (none
in the controls above, against ~1 per cutout for one injected 23.5 mag ghost): read the ghost rate
of a noisy model, such as a single diffusion draw, against a noise control (HST plus matched noise).
realism_sr.csv gives, under each SR detection, the HST flux above its mean sky and its white-noise
S/N (sky sigma times sqrt(area); drizzled HST noise is correlated, so the true S/N is lower), and
flags detections matched only at the relaxed cut, so borderline cases can be filtered.

Sky excess also moves with the sky noise, not only with invented flux: summed over ~3e5 sky pixels,
any change of the noise floor becomes flux. On local val, a copy of HST clipped at 0 (as the
training targets are) reads about +36,000 nJy per cutout against default mode's unclipped HST, 25
times the flux of a 23.5 mag ghost; in --paper-mode, where HST is clipped too, that offset is far
smaller. A median stack shifts it too. Flux biases on sources below the relaxed cut reach it as
well, as does, in paper mode, the clipped sky's positive mean: a copy 10% too bright reads +60 nJy
per cutout in default mode, +4,100 in paper mode. Compare it only between images whose skies share
their noise. HST's own sky level is a term common to every model of a pair, and a mean stack's sky
excess is the mean of its draws' (a sum over a fixed mask is linear).

Gaia stars (neo.eval.stars): stars moved between the HST epoch and the LSST epoch, so an SR image
that puts one where the LR image shows it would read as a ghost plus a miss. With a star mask
(measure_pair star_mask=...; compare.py builds it from the cutout's WCS), SR detections and HST
sources whose centroid falls in the mask are flagged near_star in realism_sr.csv / realism_hst.csv
and counted in n_star_sr / n_star_hst. By default (set_aside=True, realism.csv stars_set_aside = 1)
they are set aside: left out of every ghost, match, purity and completeness count, and the mask is
taken out of the sky (sky area, sky excess). With compare.py --keep-stars (set_aside=False,
stars_set_aside = 0) they keep their flags but count like any other source. n_gaia_stars is the
number of catalog stars whose masks reach into the cutout, and empty when no catalog lookup was
made for it (then realism.md says the stars were not checked rather than that there are none).

Noise control, HST depth and bright ghosts: most ghosts at the HST threshold are bumps of the SR
sky texture rather than invented galaxies (the 2026-10 survey found ghost counts equal to a
phase-randomised noise control in every GAN run, concentrated in the deepest HST fields). So each
cutout also records, per model, the detections on a noise control of the SR sky (noise_control:
the sky pixels phase-randomised and rank-matched back to their own values, i.e. the SR sky's power
spectrum and histogram with no sources; counted over the whole cutout), the HST sky sigma
(hst_sky_sigma_njy, so realism.md bins ghosts by HST depth, terciles of that sigma), and the
bright ghosts (peak more than BRIGHT_PEAK_SIGMA x the HST sky sigma above the SR image's own sky
mean, with no HST segment, full or relaxed cut, within BRIGHT_RADIUS_ARCSEC), which sky texture at
these thresholds does not make. The peak is taken above the SR sky mean because in --paper-mode
the SR image is not background subtracted, so its sky level would otherwise count towards the
height, and the HST sky sigma is the width of HST's sky around its own median.
The sky the control is built from leaves out the bright ghosts' segments (dilated by the margin):
they are invented sources, not texture, and their pixels would otherwise feed the control's
bright tail; fainter ghosts stay in, since the control is meant to reproduce exactly those bumps.
realism.md reports ghosts as the excess over the control (per arcmin^2, bootstrap CI), bright
ghosts per arcmin^2, and both per depth bin.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
from photutils.segmentation import SourceCatalog
from scipy import ndimage

from neo.eval.catalogs import detect_hst

MARGIN_ARCSEC = 0.2  # one LR pixel: how far an SR centroid may sit from HST's segments
# Hysteresis: a counterpart in the other image is looked for down to half the detection threshold
# and a quarter of its minimum area, so a real source just under the cut in one image and just
# over it in the other is neither a ghost nor a miss (module docstring).
RELAX_THRESHOLD = 0.5
RELAX_NPIXELS = 0.25
AB_NJY = 31.4  # AB magnitude of 1 nJy
# HST kron magnitude bin edges for completeness. At the paper threshold (1.3 nJy per 0.0333" px,
# 23.7 mag/arcsec^2) and 100 px minimum area, HST detections on our COSMOS-Web cutouts reach only
# kron mag ~25 (local val: 17.2-25.2, median 23.4), so the bins resolve 21-25 rather than 23-27.
MAG_EDGES = (21.0, 22.0, 23.0, 24.0, 25.0)
N_BOOT = 1000
BRIGHT_PEAK_SIGMA = 10.0  # a bright ghost peaks above this many HST sky sigmas...
BRIGHT_RADIUS_ARCSEC = 0.5  # ...with no HST segment (either cut) within this radius
N_DEPTH_BINS = 3  # cutouts binned at the terciles of the HST sky sigma

PAIR_FIELDS = [
    "name",
    "model",
    "n_hst",
    "n_sr",
    "n_matched",
    "n_relaxed_only",
    "n_ghost",
    "ghost_flux_njy",
    "n_recovered",
    "area_arcmin2",
    "sky_area_arcmin2",
    "sky_excess_njy",
    "sky_excess_njy_per_arcmin2",
    "margin_px",
    "threshold_njy",
    "npixels",
    "fwhm",
    "relaxed_threshold_njy",
    "relaxed_npixels",
    "n_star_sr",
    "n_star_hst",
    "n_gaia_stars",
    "stars_set_aside",
    "hst_sky_sigma_njy",
    "n_noise_control",
    "n_bright_ghost",
]
SR_FIELDS = [
    "name",
    "model",
    "label",
    "xcentroid",
    "ycentroid",
    "area",
    "segment_flux_njy",
    "mag",
    "ghost",
    "relaxed_only",
    "hst_labels",
    "hst_flux_njy",
    "hst_snr",
    "near_star",
    "peak_njy",
    "bright_ghost",
]
HST_FIELDS = [
    "name",
    "label",
    "xcentroid",
    "ycentroid",
    "area",
    "segment_flux_njy",
    "kron_flux_njy",
    "kron_mag",
    "mag_bin",
    "near_star",
]
FILES = ("realism.csv", "realism_sr.csv", "realism_hst.csv", "realism_summary.csv")


def ab_mag(flux_njy) -> np.ndarray:
    """AB magnitude of a flux in nJy; NaN where the flux is not positive."""
    flux = np.asarray(flux_njy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, AB_NJY - 2.5 * np.log10(flux), np.nan)


def bin_labels(edges=MAG_EDGES) -> list[str]:
    inner = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    return [f"<{edges[0]:g}", *inner, f">={edges[-1]:g}"]


def mag_bin(mag, edges=MAG_EDGES) -> np.ndarray:
    """Index into bin_labels(edges) per magnitude; -1 for NaN (no positive kron flux)."""
    mag = np.asarray(mag, dtype=float)
    return np.where(np.isfinite(mag), np.digitize(mag, edges), -1)


def disk(radius: float) -> np.ndarray:
    """Boolean disk of `radius` px (pixel centres within it), at least the single centre pixel."""
    r = int(np.floor(radius + 1e-6))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return np.hypot(xx, yy) <= radius + 1e-6  # 0.2" / (0.2" / 6) is 6 only up to rounding


def positions(cat: SourceCatalog, labels: np.ndarray):
    """Centroids (x = column, y = row) of a catalog's sources; for the rare source whose centroid
    is undefined (its smoothed flux sums to <= 0), the geometric centre of its segment."""
    x = np.array(cat.xcentroid, dtype=float, ndmin=1)
    y = np.array(cat.ycentroid, dtype=float, ndmin=1)
    bad = ~(np.isfinite(x) & np.isfinite(y))
    if bad.any():
        com = ndimage.center_of_mass(np.ones(labels.shape), labels, np.asarray(cat.labels)[bad])
        y[bad], x[bad] = np.asarray(com, dtype=float).T
    return x, y


def labels_near(labels: np.ndarray, x, y, structure: np.ndarray) -> list[np.ndarray]:
    """HST labels within `structure` of each position: the segments whose dilation contains it.

    Same pixel test as ndimage.binary_dilation(segment, structure) at the rounded position (the
    disk is symmetric), done only where the SR sources are.
    """
    r = structure.shape[0] // 2
    padded = np.pad(labels, r)
    ny, nx = labels.shape
    near = []
    for xi, yi in zip(x, y, strict=True):
        col = int(np.clip(np.rint(xi), 0, nx - 1))
        row = int(np.clip(np.rint(yi), 0, ny - 1))
        window = padded[row : row + 2 * r + 1, col : col + 2 * r + 1][structure]
        near.append(np.unique(window[window > 0]))
    return near


def relaxed(threshold: float, npixels: int):
    """(threshold, npixels) of the relaxed cut at which hysteresis looks for counterparts."""
    return threshold * RELAX_THRESHOLD, max(int(npixels * RELAX_NPIXELS), 1)


def segment_labels(detected, shape) -> np.ndarray:
    """Label image of a detect_hst result; all zeros when nothing was detected."""
    return np.zeros(shape, dtype=int) if detected is None else detected[0].data


def catalog(image: np.ndarray, detected):
    """SourceCatalog of a detect_hst result (centroids from its smoothed image, as in Table 4)."""
    segm, convolved = detected
    return SourceCatalog(image, segm, convolved_data=convolved)


def as_array(values) -> np.ndarray:
    return np.array(getattr(values, "value", values), dtype=float, ndmin=1)


def in_mask(mask, x, y) -> np.ndarray:
    """Whether each (x, y) position, rounded to a pixel, falls inside the boolean `mask` (all
    False for mask None)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if mask is None or not ok.any():
        return np.zeros(x.shape, dtype=bool)
    ny, nx = mask.shape
    col = np.clip(np.rint(np.where(ok, x, 0)), 0, nx - 1).astype(int)
    row = np.clip(np.rint(np.where(ok, y, 0)), 0, ny - 1).astype(int)
    return mask[row, col] & ok


def hst_sources(name: str, hst: np.ndarray, detected, edges=MAG_EDGES, star_mask=None) -> list:
    """HST_FIELDS rows of the sources HST shows at the full cut (near_star: inside star_mask)."""
    cat = catalog(hst, detected)
    x, y = positions(cat, detected[0].data)
    near_star = in_mask(star_mask, x, y)
    kron = as_array(cat.kron_flux)
    mags = ab_mag(kron)
    bins = mag_bin(mags, edges)
    flux, area = as_array(cat.segment_flux), as_array(cat.area)
    return [
        {
            "name": name,
            "label": int(label),
            "xcentroid": float(x[i]),
            "ycentroid": float(y[i]),
            "area": float(area[i]),
            "segment_flux_njy": float(flux[i]),
            "kron_flux_njy": float(kron[i]),
            "kron_mag": float(mags[i]),
            "mag_bin": int(bins[i]),
            "near_star": int(near_star[i]),
        }
        for i, label in enumerate(cat.labels)
    ]


def sky_level(image: np.ndarray, sky: np.ndarray):
    """(mean, sigma) per pixel of an image's sky pixels; sigma is the distribution's upper
    half-width (84th - 50th percentile), which clipping at 0 (paper mode) leaves intact."""
    values = image[sky]
    if not len(values):
        return 0.0, float("nan")
    p50, p84 = np.percentile(values, [50, 84.134])
    return float(values.mean()), float(p84 - p50)


def noise_control(image: np.ndarray, sky: np.ndarray, seed: int = 0) -> np.ndarray:
    """A source-free image with the sky's power spectrum and pixel distribution.

    The image, its sky mean taken out and its non-sky pixels set to that mean (zero), is Fourier
    phase-randomised (the power spectrum, i.e. the texture's correlation structure, is kept;
    everything else is scrambled), then every pixel is rank-matched to the sky pixels' own sorted
    values, so the control has exactly the sky's histogram: the clipped pile at zero of a
    paper-mode image, the stretched positive tail of a GAN sky. Detected with the cutout's
    settings, it gives the detections the sky texture alone produces (the method of the 2026-10
    survey tool, neo_outputs/scratch/all_experiments_quality/tool/quality.py noise_control, which
    ran on background-subtracted panels; here a paper-mode SR image keeps its sky level, so the
    holes must be filled at that level or their edges would be scrambled into texture). All zeros
    when there is no sky.
    """
    image = np.asarray(image, dtype=np.float64)
    values = np.sort(image[sky])
    if not len(values):
        return np.zeros_like(image)
    rng = np.random.default_rng(seed)
    spectrum = np.abs(np.fft.fft2(np.where(sky, image - values.mean(), 0.0)))
    phases = np.exp(2j * np.pi * rng.random(image.shape))
    scrambled = np.fft.ifft2(spectrum * phases).real
    ranks = np.argsort(np.argsort(scrambled, axis=None, kind="stable"), kind="stable")
    quantile = (ranks + 0.5) / ranks.size
    grid = (np.arange(values.size) + 0.5) / values.size
    return np.interp(quantile, grid, values).reshape(image.shape)


def control_seed(name: str) -> int:
    """One seed per cutout, so every model's control shares the same random phases."""
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 2**32


def measure_pair(
    name: str,
    hst: np.ndarray,
    srs: dict,
    detected,
    threshold: float,
    npixels: int,
    fwhm: float,
    pixel_arcsec: float,
    margin_arcsec: float = MARGIN_ARCSEC,
    edges=MAG_EDGES,
    hst_raw: np.ndarray | None = None,
    srs_raw: dict | None = None,
    star_mask: np.ndarray | None = None,
    n_stars: int | None = None,
    set_aside: bool = True,
    bright_sigma: float = BRIGHT_PEAK_SIGMA,
    bright_radius_arcsec: float = BRIGHT_RADIUS_ARCSEC,
) -> dict:
    """Realism record of one cutout: {"pairs": per-model rows, "sr": SR detections, "hst": HST
    sources}, plain picklable rows (PAIR_FIELDS, SR_FIELDS, HST_FIELDS + recovered:<model>).

    `hst` and `srs` are the images as cataloged (nJy per HR pixel, same preparation); `detected`
    is detect_hst(hst, threshold, npixels, fwhm), None when HST shows no source at that cut. Each
    SR image is detected with the same threshold, npixels and fwhm; HST, and SR images that miss
    an HST source, again at the relaxed cut (relaxed()). `hst_raw` and `srs_raw` are the same
    images before any background subtraction, for the sky excess (default: as cataloged).
    `star_mask` (neo.eval.stars.star_mask) flags the sources inside it; with `set_aside` they are
    set aside and the mask is taken out of the sky, otherwise they count like any other (module
    docstring). `n_stars` is the number of catalog stars behind the mask, None when the cutout
    was not checked against the catalog. The noise control and the bright ghosts are always
    recorded; `bright_sigma` and `bright_radius_arcsec` define a bright ghost.
    """
    hst_raw = hst if hst_raw is None else hst_raw
    srs_raw = srs if srs_raw is None else srs_raw
    radius = margin_arcsec / pixel_arcsec
    structure = disk(radius)
    wide = disk(bright_radius_arcsec / pixel_arcsec)
    loose = relaxed(threshold, npixels)
    hst_labels = segment_labels(detected, hst.shape)
    hst_loose = segment_labels(detect_hst(hst, *loose, fwhm), hst.shape)
    hst_any = np.where(hst_labels > 0, hst_labels, hst_loose)  # either cut, for bright ghosts
    hst_rows = [] if detected is None else hst_sources(name, hst, detected, edges, star_mask)
    star_labels = {row["label"] for row in hst_rows if row["near_star"]}  # flagged near a star
    aside = star_labels if set_aside else set()  # ... and left out of the counts
    n_hst = len(hst_rows) - len(aside)
    # every pixel above the full cut is also above the relaxed one; the union only guards against
    # deblending assigning a few of them differently
    footprint = ndimage.binary_dilation((hst_labels > 0) | (hst_loose > 0), structure=structure)
    sky = ~footprint
    if star_mask is not None and set_aside:
        sky &= ~star_mask  # a star's disk is neither source nor sky: nothing in it counts
    arcmin2_per_px = pixel_arcsec**2 / 3600
    area_arcmin2 = hst.size * arcmin2_per_px
    sky_area = float(sky.sum() * arcmin2_per_px)
    mean, sigma = sky_level(hst, sky)
    noise = sigma if sigma > 0 else float("nan")  # a sky clipped to mostly zeros has no width
    seed = control_seed(name)

    pair_rows, sr_rows = [], []
    for model, sr in srs.items():
        recovered = set()
        n_ghost, n_loose, n_star, n_bright, ghost_flux = 0, 0, 0, 0, 0.0
        found = detect_hst(sr, threshold, npixels, fwhm)  # the HST detection, run on this SR image
        sr_mean = sky_level(sr, sky)[0]  # a bright ghost's peak is measured above the SR sky
        if found is not None:
            cat = catalog(sr, found)
            x, y = positions(cat, found[0].data)
            flux, area = as_array(cat.segment_flux), as_array(cat.area)
            mags = ab_mag(flux)
            labels = np.asarray(cat.labels)
            peaks = np.asarray(ndimage.maximum(sr, found[0].data, labels), dtype=float)
            # what HST holds under each SR segment, above its mean sky (clipped noise in paper mode)
            under = ndimage.sum_labels(hst, found[0].data, labels) - mean * area
            near = labels_near(hst_labels, x, y, structure)
            near_loose = labels_near(hst_loose, x, y, structure)
            near_wide = labels_near(hst_any, x, y, wide)
            starred = in_mask(star_mask, x, y)
            bright_labels = []
            for i, label in enumerate(cat.labels):
                recovered.update(int(v) for v in near[i])
                loose_only = not len(near[i]) and len(near_loose[i]) > 0
                ghost = not len(near[i]) and not len(near_loose[i])
                if starred[i]:
                    n_star += 1
                    if set_aside:  # neither a ghost nor a match
                        loose_only = ghost = False
                bright = (
                    ghost and peaks[i] - sr_mean > bright_sigma * noise and not len(near_wide[i])
                )
                n_ghost += int(ghost)
                n_loose += int(loose_only)
                n_bright += int(bright)
                if bright:
                    bright_labels.append(int(label))
                ghost_flux += float(flux[i]) if ghost else 0.0
                sr_rows.append(
                    {
                        "name": name,
                        "model": model,
                        "label": int(label),
                        "xcentroid": float(x[i]),
                        "ycentroid": float(y[i]),
                        "area": float(area[i]),
                        "segment_flux_njy": float(flux[i]),
                        "mag": float(mags[i]),
                        "ghost": int(ghost),
                        "relaxed_only": int(loose_only),
                        "hst_labels": ";".join(str(int(v)) for v in near[i]),
                        "hst_flux_njy": float(under[i]),
                        "hst_snr": float(under[i] / (noise * np.sqrt(area[i]))),
                        "near_star": int(starred[i]),
                        "peak_njy": float(peaks[i]),
                        "bright_ghost": int(bright),
                    }
                )
        if len(recovered - aside) < n_hst:  # a source missed at the full cut may show below it
            found_loose = detect_hst(sr, *loose, fwhm)
            if found_loose is not None:
                x, y = positions(catalog(sr, found_loose), found_loose[0].data)
                for labels in labels_near(hst_labels, x, y, structure):
                    recovered.update(int(v) for v in labels)
        n_sr = (0 if found is None else found[0].nlabels) - (n_star if set_aside else 0)
        for row in hst_rows:
            row[f"recovered:{model}"] = int(row["label"] in recovered)
        recovered -= aside
        excess = float(np.sum((np.asarray(srs_raw[model], dtype=float) - hst_raw)[sky]))
        control_sky = sky
        if found is not None and bright_labels:  # invented sources are not sky texture
            segments = np.isin(found[0].data, bright_labels)
            control_sky = sky & ~ndimage.binary_dilation(segments, structure=structure)
        control = detect_hst(noise_control(sr, control_sky, seed), threshold, npixels, fwhm)
        pair_rows.append(
            {
                "name": name,
                "model": model,
                "n_hst": n_hst,
                "n_sr": n_sr,
                "n_matched": n_sr - n_ghost,
                "n_relaxed_only": n_loose,
                "n_ghost": n_ghost,
                "ghost_flux_njy": ghost_flux,
                "n_recovered": len(recovered),
                "area_arcmin2": area_arcmin2,
                "sky_area_arcmin2": sky_area,
                "sky_excess_njy": excess,
                "sky_excess_njy_per_arcmin2": excess / sky_area if sky_area > 0 else float("nan"),
                "margin_px": radius,
                "threshold_njy": threshold,
                "npixels": npixels,
                "fwhm": fwhm,
                "relaxed_threshold_njy": loose[0],
                "relaxed_npixels": loose[1],
                "n_star_sr": n_star,
                "n_star_hst": len(star_labels),
                "n_gaia_stars": None if n_stars is None else int(n_stars),
                "stars_set_aside": int(set_aside),
                "hst_sky_sigma_njy": sigma,
                "n_noise_control": 0 if control is None else int(control[0].nlabels),
                "n_bright_ghost": n_bright,
            }
        )
    return {"pairs": pair_rows, "sr": sr_rows, "hst": hst_rows}


def resample_weights(clusters, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """(n_boot, n_cutouts) times each cutout is drawn when whole clusters are resampled."""
    ids, index = np.unique(np.asarray(clusters, dtype=str), return_inverse=True)
    if not len(ids):
        return np.zeros((n_boot, 0))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    counts = np.zeros((n_boot, len(ids)))
    np.add.at(counts, (np.arange(n_boot)[:, None], draws), 1)
    return counts[:, np.ravel(index)]


def interval(point: float, boot: np.ndarray, n: int) -> dict:
    boot = boot[np.isfinite(boot)]
    if not len(boot):
        return {"value": point, "ci_lo": float("nan"), "ci_hi": float("nan"), "n": n}
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"value": point, "ci_lo": float(lo), "ci_hi": float(hi), "n": n}


def ratio(num, den, weights, n=None) -> dict:
    """sum(num) / sum(den) over cutouts, with its bootstrap CI (NaN where den sums to 0); n is
    sum(den) (detections, sources, cutouts) unless given."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        point = num.sum() / den.sum() if den.sum() > 0 else float("nan")
        boot = (weights @ num) / (weights @ den)
    return interval(float(point), boot, int(round(den.sum())) if n is None else n)


def median(values, weights) -> dict:
    """Median over cutouts (finite values only), with its bootstrap CI."""
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    if not ok.any():
        return interval(float("nan"), np.array([]), 0)
    order = np.argsort(values[ok])
    v = values[ok][order]
    cum = np.cumsum(weights[:, ok][:, order], axis=1)
    total = cum[:, -1]
    # middle element(s) of each resample, as if every cutout were repeated as often as drawn
    lo = np.argmax(cum > ((total - 1) // 2)[:, None], axis=1)
    hi = np.argmax(cum > (total // 2)[:, None], axis=1)
    boot = np.where(total > 0, (v[lo] + v[hi]) / 2, np.nan)
    return interval(float(np.median(v)), boot, int(ok.sum()))


def excess(ghosts, sky_area, control, area, weights, n) -> dict:
    """sum(ghosts) / sum(sky_area) - sum(control) / sum(area) over cutouts, with its bootstrap
    CI: ghosts per arcmin^2 of sky beyond what the noise control (over the whole cutout) gives."""
    ghosts, sky_area = np.asarray(ghosts, dtype=float), np.asarray(sky_area, dtype=float)
    control, area = np.asarray(control, dtype=float), np.asarray(area, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        point = ghosts.sum() / sky_area.sum() - control.sum() / area.sum()
        boot = (weights @ ghosts) / (weights @ sky_area) - (weights @ control) / (weights @ area)
    return interval(
        float(point) if sky_area.sum() > 0 and area.sum() > 0 else float("nan"), boot, n
    )


def depth_bins(sigma, n_bins: int = N_DEPTH_BINS):
    """Bin index per cutout (0 = deepest, lowest HST sky sigma; -1 without a sigma) at the
    quantiles of the finite sigmas, and the bin edges."""
    sigma = np.asarray(sigma, dtype=float)
    finite = np.isfinite(sigma)
    if not finite.any():
        return np.full(sigma.shape, -1), []
    edges = np.percentile(sigma[finite], np.linspace(0, 100, n_bins + 1)[1:-1])
    index = np.digitize(np.where(finite, sigma, 0), edges)
    return np.where(finite, index, -1), [float(e) for e in edges]


def summarize(found: dict, models, groups=None, edges=MAG_EDGES, n_boot=N_BOOT, seed=0) -> list:
    """Per model: counts and {statistic: {value, ci_lo, ci_hi, n}} over the cutouts in `found`.

    Every model is resampled with the same draws, so their intervals are directly comparable.
    """
    names = sorted({r["name"] for r in found["pairs"]})
    index = {n: i for i, n in enumerate(names)}
    weights = resample_weights([(groups or {}).get(n, n) for n in names], n_boot, seed)
    labels = bin_labels(edges)
    hst = found["hst"]
    where = np.array([index[r["name"]] for r in hst], dtype=int)
    bins = np.array([r["mag_bin"] for r in hst], dtype=int)
    # HST sources near Gaia stars are set aside from the completeness bins too, unless the cutout
    # was measured with the stars counted (compare.py --keep-stars: stars_set_aside = 0)
    aside = {}
    for r in found["pairs"]:
        aside.setdefault(r["name"], r.get("stars_set_aside", 1))
    near = np.array([r.get("near_star", 0) for r in hst], dtype=float) > 0
    set_aside = np.array([aside.get(r["name"], 1) for r in hst], dtype=float) > 0
    bins = np.where(near & set_aside, -1, bins)
    per_bin = np.zeros((len(names), len(labels)))
    np.add.at(per_bin, (where[bins >= 0], bins[bins >= 0]), 1)
    # the HST sky sigma is a property of the cutout: the same in every model's row
    by_name = {}
    for r in found["pairs"]:
        by_name.setdefault(r["name"], r)
    depth, depth_edges = depth_bins([by_name[n].get("hst_sky_sigma_njy", np.nan) for n in names])
    summary = []
    for model in models:
        rows = {r["name"]: r for r in found["pairs"] if r["model"] == model}
        col = {
            k: np.array([rows[n].get(k, np.nan) for n in names], dtype=float)
            for k in PAIR_FIELDS[2:]
        }
        ghosts = [r["mag"] for r in found["sr"] if r["model"] == model and r["ghost"]]
        ghost_mags = np.asarray(ghosts, dtype=float)
        ghost_mags = ghost_mags[np.isfinite(ghost_mags)]
        stats = {
            "purity": ratio(col["n_matched"], col["n_sr"], weights),
            "ghosts_per_cutout": ratio(col["n_ghost"], np.ones(len(names)), weights),
            # per arcmin^2 of sky: ghosts can only lie outside the dilated HST maps
            "ghosts_per_arcmin2": ratio(
                col["n_ghost"], col["sky_area_arcmin2"], weights, len(names)
            ),
            "completeness": ratio(col["n_recovered"], col["n_hst"], weights),
        }
        rec = np.array([r.get(f"recovered:{model}", 0) for r in hst], dtype=float)
        rec_bin = np.zeros_like(per_bin)
        np.add.at(rec_bin, (where[bins >= 0], bins[bins >= 0]), rec[bins >= 0])
        for b, label in enumerate(labels):
            stats[f"completeness_{label}"] = ratio(rec_bin[:, b], per_bin[:, b], weights)
        stats["sky_excess_njy_per_arcmin2"] = median(col["sky_excess_njy_per_arcmin2"], weights)
        # ghosts against the noise control and by HST depth; the control is counted over the
        # whole cutout, ghosts over its sky
        n = len(names)
        ghost, sky, ctrl, area = (
            np.nan_to_num(col[k])
            for k in ("n_ghost", "sky_area_arcmin2", "n_noise_control", "area_arcmin2")
        )
        bright = np.nan_to_num(col["n_bright_ghost"])
        stats["noise_ghosts_per_arcmin2"] = ratio(ctrl, area, weights, n)
        stats["ghost_excess_per_arcmin2"] = excess(ghost, sky, ctrl, area, weights, n)
        stats["bright_ghosts_per_arcmin2"] = ratio(bright, sky, weights, n)
        depth_n, depth_sigma = [], []
        for b in range(N_DEPTH_BINS):
            inb = (depth == b).astype(float)
            n_b = int(inb.sum())
            depth_n.append(n_b)
            sig_b = col["hst_sky_sigma_njy"][depth == b]
            depth_sigma.append([float(sig_b.min()), float(sig_b.max())] if n_b else [np.nan] * 2)
            stats[f"ghosts_per_arcmin2_depth{b + 1}"] = ratio(ghost * inb, sky * inb, weights, n_b)
            stats[f"noise_ghosts_per_arcmin2_depth{b + 1}"] = ratio(
                ctrl * inb, area * inb, weights, n_b
            )
            stats[f"ghost_excess_per_arcmin2_depth{b + 1}"] = excess(
                ghost * inb, sky * inb, ctrl * inb, area * inb, weights, n_b
            )
            stats[f"bright_ghosts_per_arcmin2_depth{b + 1}"] = ratio(
                bright * inb, sky * inb, weights, n_b
            )
        stars = np.nan_to_num(col["n_gaia_stars"])  # NaN: the cutout was not checked
        summary.append(
            {
                "model": model,
                "n_cutouts": len(names),
                "n_groups": len(set((groups or {}).get(n, n) for n in names)),
                "n_hst": int(col["n_hst"].sum()),
                "area_arcmin2": float(col["area_arcmin2"].sum()),
                "sky_area_arcmin2": float(col["sky_area_arcmin2"].sum()),
                "n_sr": int(col["n_sr"].sum()),
                "n_relaxed_only": int(col["n_relaxed_only"].sum()),
                "n_ghost": int(col["n_ghost"].sum()),
                "ghost_mag_median": float(np.median(ghost_mags)) if len(ghost_mags) else None,
                "n_bright_ghost": int(bright.sum()),
                "n_noise_control": int(ctrl.sum()),
                "n_star_sr": int(np.nan_to_num(col["n_star_sr"]).sum()),
                "n_star_hst": int(np.nan_to_num(col["n_star_hst"]).sum()),
                "n_gaia_stars": int(stars.sum()),
                "n_cutouts_with_stars": int((stars > 0).sum()),
                "stars_checked": bool(np.isfinite(col["n_gaia_stars"]).any()),
                "stars_set_aside": bool(np.all(np.nan_to_num(col["stars_set_aside"], nan=1) > 0)),
                "depth_edges": depth_edges,
                "depth_n": depth_n,
                "depth_sigma": depth_sigma,
                "stats": stats,
            }
        )
    return summary


def write_csv(path: Path, rows: list, fields: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cell(s: dict, digits: int = 3) -> str:
    if not np.isfinite(s["value"]):
        return "n/a"
    if not np.isfinite(s["ci_lo"]):
        return f"{s['value']:.{digits}g}"
    return f"{s['value']:.{digits}g} [{s['ci_lo']:.{digits}g}, {s['ci_hi']:.{digits}g}]"


def report_lines(
    summary: list, margin_arcsec: float, margin_px, edges=MAG_EDGES, n_failed: int = 0
) -> list:
    """realism.md body: definitions, the per-model table and completeness by magnitude.

    `margin_px` is the margin in HR pixels of each cutout (realism.csv margin_px); `n_failed`
    counts the pairs whose realism failed and are left out.
    """
    if not summary:
        return ["No models."]
    first = summary[0]
    n, area, sky = first["n_cutouts"], first["area_arcmin2"], first["sky_area_arcmin2"]
    density = f"{first['n_hst'] / area:.3g}" if area > 0 else "n/a"
    resampled = (
        f"whole groups of overlapping cutouts ({first['n_groups']} groups, groups.csv)"
        if first["n_groups"] < n
        else "cutouts"
    )
    failed = f"; {n_failed} more failed (see the log) and are left out" if n_failed else ""
    lines = [
        "Each SR image is detected on its own with its pair's HST detection settings above "
        "(deblended) and matched to HST with hysteresis: a counterpart is looked for down to "
        f"{RELAX_THRESHOLD:g} x the threshold and {RELAX_NPIXELS:g} x the minimum area (the "
        "relaxed cut). An SR detection is a ghost when its centroid lies outside HST's "
        f'segmentation maps at the full and the relaxed cut dilated by {margin_arcsec:g}" (a '
        f"disk of radius {span(margin_px)} HR px), i.e. HST shows nothing there even at the "
        "relaxed cut; otherwise it is matched (relaxed only: matched by the relaxed map alone). "
        "An HST source is recovered when an SR centroid, at the full or the relaxed cut, lies "
        "inside its own segment dilated by the same margin. So a real source just under the "
        "cut in one image and just over it in the other is neither a ghost nor a miss, but an "
        "invented source landing on a faint real HST source counts as matched, and flux or size "
        "biases on sources further below the cut, and noise, can still reach these counts "
        "(neo/eval/realism.py: compare a noisy model with a noise control). Purity = matched / "
        "all SR detections. Ghosts per arcmin^2 and sky excess are per arcmin^2 of sky (outside "
        "the dilated maps); sky excess = sum of SR - HST over that sky, on the images before any "
        "background subtraction (nJy / arcmin^2). It moves with the sky noise too (clipping at "
        "0, a median stack), not only with invented flux: compare it only between images whose "
        "skies share their noise.",
        f"{n} cutouts ({area:.3g} arcmin^2, {sky:.3g} of it sky), "
        f"{first['n_hst']} HST sources ({density} per arcmin^2), every pair counted (also those "
        f"Table 4 drops){failed}. Ratios are totals over cutouts; [bootstrap 95% CI] resampling "
        f"{resampled}.",
        "",
        "| model | SR detections | relaxed only | purity | ghosts / cutout | ghosts / arcmin^2 "
        "sky | median ghost mag | completeness | median sky excess (nJy / arcmin^2) |",
        "|" + "---|" * 9,
    ]
    for s in summary:
        st = s["stats"]
        ghost_mag = "n/a" if s["ghost_mag_median"] is None else f"{s['ghost_mag_median']:.2f}"
        lines.append(
            f"| {s['model']} | {s['n_sr']} | {s['n_relaxed_only']} | {cell(st['purity'])} | "
            f"{cell(st['ghosts_per_cutout'])} | {cell(st['ghosts_per_arcmin2'])} | {ghost_mag} | "
            f"{cell(st['completeness'])} | {cell(st['sky_excess_njy_per_arcmin2'])} |"
        )
    labels = bin_labels(edges)
    counts = [first["stats"][f"completeness_{b}"]["n"] for b in labels]
    heads = [f"{b} (n={c})" for b, c in zip(labels, counts, strict=True)]
    lines += [
        "",
        "Completeness by HST kron magnitude (AB; n = HST sources in the bin, those without a "
        "positive kron flux in none):",
        "",
        "| model | " + " | ".join(heads) + " |",
        "|" + "---|" * (len(labels) + 1),
    ]
    for s in summary:
        cells = " | ".join(cell(s["stats"][f"completeness_{b}"]) for b in labels)
        lines.append(f"| {s['model']} | {cells} |")
    lines += [
        "",
        "Ghosts against a noise control. Each SR image's sky (outside the dilated HST maps, the "
        "star masks that are set aside, and the segments of its bright ghosts dilated by the "
        "margin) is Fourier phase-randomised and rank-matched back to its own pixel values: a "
        "source-free image with the SR sky's power spectrum and histogram. Detected with the "
        "same settings, it gives the detections sky texture alone produces (counted over the "
        "whole cutout, per arcmin^2). Excess = ghosts per arcmin^2 of sky - control detections "
        "per arcmin^2. The pixels of fainter ghosts stay in the sky the control is made from, so "
        "invented faint compact sources feed the control's histogram too: the excess is a lower "
        "bound on invented sources, which is why bright ghosts are counted directly. Bright "
        f"ghosts: peak more than {BRIGHT_PEAK_SIGMA:g} x the HST sky sigma above the SR image's "
        "own sky mean, with no HST segment (full or relaxed cut) within "
        f'{BRIGHT_RADIUS_ARCSEC:g}", which sky texture at these thresholds does not make.',
        "",
        "| model | ghosts / arcmin^2 sky | noise control / arcmin^2 | excess / arcmin^2 | "
        "bright ghosts | bright ghosts / arcmin^2 |",
        "|" + "---|" * 6,
    ]
    for s in summary:
        st = s["stats"]
        lines.append(
            f"| {s['model']} | {cell(st['ghosts_per_arcmin2'])} | "
            f"{cell(st['noise_ghosts_per_arcmin2'])} | {cell(st['ghost_excess_per_arcmin2'])} | "
            f"{s['n_bright_ghost']} | {cell(st['bright_ghosts_per_arcmin2'])} |"
        )
    lines += [
        "",
        f"Ghosts by HST depth: cutouts split into {N_DEPTH_BINS} bins at the quantiles of the HST "
        "sky sigma (nJy per HR px; depth 1 is the deepest, lowest sigma; n = cutouts):",
        "",
        "| model | depth (sigma range, n) | ghosts / arcmin^2 sky | noise control / arcmin^2 | "
        "excess / arcmin^2 | bright ghosts / arcmin^2 |",
        "|" + "---|" * 6,
    ]
    for s in summary:
        st = s["stats"]
        for b in range(N_DEPTH_BINS):
            lo, hi = s["depth_sigma"][b]
            span_b = f"{lo:.3g}-{hi:.3g}" if np.isfinite(lo) else "n/a"
            lines.append(
                f"| {s['model']} | {b + 1} ({span_b}, n={s['depth_n'][b]}) | "
                f"{cell(st[f'ghosts_per_arcmin2_depth{b + 1}'])} | "
                f"{cell(st[f'noise_ghosts_per_arcmin2_depth{b + 1}'])} | "
                f"{cell(st[f'ghost_excess_per_arcmin2_depth{b + 1}'])} | "
                f"{cell(st[f'bright_ghosts_per_arcmin2_depth{b + 1}'])} |"
            )
    per_model = ", ".join(f"{s['model']} {s['n_star_sr']}" for s in summary)
    if first["n_gaia_stars"]:
        inside = (
            f"{first['n_star_hst']} HST sources, and SR detections per model: {per_model} "
            "(near_star in realism_hst.csv / realism_sr.csv)."
        )
        lines += [
            "",
            f"Gaia DR3 stars (neo/eval/stars.py): {first['n_gaia_stars']} in "
            f"{first['n_cutouts_with_stars']} of the {n} cutouts, each masked around both its "
            "HST-epoch and LSST-epoch positions (the mask radius is stated above). "
            + (
                f"Set aside from every count above: {inside}"
                if first["stars_set_aside"]
                else f"Counted like any other source (--keep-stars): {inside}"
            ),
        ]
    elif first["stars_checked"]:
        lines += ["", "Gaia DR3 stars (neo/eval/stars.py): none in these cutouts."]
    else:
        lines += [
            "",
            "Gaia DR3 stars (neo/eval/stars.py): not checked (no star masks were built, so any "
            "source inside one is counted like any other).",
        ]
    return lines


def span(values) -> str:
    """One value, or the range of per-cutout values."""
    if not len(values):
        return "n/a"
    lo, hi = min(values), max(values)
    return f"{lo:.3g}" if np.isclose(lo, hi, rtol=1e-6) else f"{lo:.3g}-{hi:.3g}"


def write_report(
    out: Path, found: dict, models, groups, intro, margin_arcsec: float, n_failed: int = 0
):
    """Write realism.csv, realism_sr.csv, realism_hst.csv, realism_summary.csv and realism.md
    (`intro` lines first; `n_failed` pairs whose realism failed); returns (summary, the
    realism.md lines)."""
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(found, models, groups)
    write_csv(out / "realism.csv", found["pairs"], PAIR_FIELDS)
    write_csv(out / "realism_sr.csv", found["sr"], SR_FIELDS)
    recovered = [f"recovered:{m}" for m in models]
    write_csv(out / "realism_hst.csv", found["hst"], HST_FIELDS + recovered)
    rows = [
        {"model": s["model"], "statistic": stat, **v}
        for s in summary
        for stat, v in s["stats"].items()
    ]
    write_csv(
        out / "realism_summary.csv", rows, ["model", "statistic", "value", "ci_lo", "ci_hi", "n"]
    )
    margin_px = [r["margin_px"] for r in found["pairs"]]
    lines = [*intro, *report_lines(summary, margin_arcsec, margin_px, n_failed=n_failed)]
    (out / "realism.md").write_text("\n".join(lines) + "\n")
    return summary, lines
