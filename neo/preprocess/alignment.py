"""Measure how well each pair's HR (HST) and LR (LSST) images line up on the sky.

The pairs are nested by construction (LR pixel i covers HR pixels factor*i .. factor*i+factor-1), so
any residual offset comes from the two surveys' astrometry. To measure it, the HR cutout is
smoothed to the LR resolution (Gaussian of FWHM sqrt(lr_fwhm^2 - hr_fwhm^2)) and summed in
factor x factor blocks onto the LR grid; then the offset of the LR image relative to that is
measured two independent ways:
  xcorr     sub-pixel phase cross-correlation of the cutout (Hann-windowed)
  centroid  median offset of windowed centroids of bright sources detected in the LR image
Offsets are of LR content relative to HR content, in LR pixels (+y = higher row index).

Usage:  python -m neo.preprocess.alignment --pairs $NEO_DATA/pairs/cosmos_web_i --n 500
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import sep
from astropy.io import fits
from astropy.stats import mad_std
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import gaussian_filter
from skimage.filters import window
from skimage.registration import phase_cross_correlation


def degrade(hr, factor, sigma_hr_px):
    """HR smoothed to LR resolution and summed onto the LR grid."""
    smooth = gaussian_filter(np.asarray(hr, dtype=np.float64), sigma_hr_px, mode="reflect")
    ny, nx = smooth.shape[0] // factor, smooth.shape[1] // factor
    return smooth[: ny * factor, : nx * factor].reshape(ny, factor, nx, factor).sum(axis=(1, 3))


def xcorr_offset(lr, hr_low, margin=8):
    """Offset (dy, dx) of lr content relative to hr_low content, in LR pixels."""
    a = np.asarray(lr, dtype=np.float64)[margin:-margin, margin:-margin]
    b = hr_low[margin:-margin, margin:-margin]
    w = window("hann", a.shape)
    a = (a - np.median(a)) * w
    b = (b - np.median(b)) * w
    shift, _, _ = phase_cross_correlation(b, a, upsample_factor=100, normalization=None)
    return -float(shift[0]), -float(shift[1])  # shift registers lr onto hr; content offset is minus


def centroid_offsets(lr, hr_low, nsigma=15.0, minarea=5, margin=10):
    """Per-source (dy, dx) of windowed centroids, LR minus degraded HR, for bright LR sources."""
    data = np.ascontiguousarray(lr, dtype=np.float64)
    bkg = sep.Background(data)
    sub = data - bkg.back()
    objects = sep.extract(sub, nsigma, err=bkg.globalrms, minarea=minarea)
    if len(objects) == 0:
        return np.empty((0, 2))
    keep = (
        (objects["x"] > margin)
        & (objects["x"] < data.shape[1] - margin)
        & (objects["y"] > margin)
        & (objects["y"] < data.shape[0] - margin)
        & (objects["flag"] == 0)
    )
    objects = objects[keep]
    if len(objects) == 0:
        return np.empty((0, 2))
    sig = 2.0 / 2.35 * np.full(len(objects), 4.0)  # windowed centroid on ~ the LR PSF (4 px)
    hr = np.ascontiguousarray(hr_low - sep.Background(np.ascontiguousarray(hr_low)).back())
    xl, yl, fl = sep.winpos(sub, objects["x"], objects["y"], sig)
    xh, yh, fh = sep.winpos(hr, objects["x"], objects["y"], sig)
    ok = (fl == 0) & (fh == 0)
    return np.column_stack([yl - yh, xl - xh])[ok]


def measure(lr_path, hr_path, lr_fwhm, hr_fwhm):
    lr_hdu, hr_hdu = fits.open(lr_path)[0], fits.open(hr_path)[0]
    lr_pix = float(np.mean(proj_plane_pixel_scales(WCS(lr_hdu.header).celestial))) * 3600
    factor = int(hr_hdu.header.get("SRFACTOR", round(hr_hdu.data.shape[0] / lr_hdu.data.shape[0])))
    hr_pix = lr_pix / factor
    sigma = np.sqrt(max(lr_fwhm**2 - hr_fwhm**2, 0.0)) / 2.3548 / hr_pix
    hr_low = degrade(hr_hdu.data, factor, sigma)
    dy, dx = xcorr_offset(lr_hdu.data, hr_low)
    cents = centroid_offsets(lr_hdu.data, hr_low)
    return lr_pix, dy, dx, cents, lr_hdu.header.get("LRFILE", "")


def summarize(values, lr_pix):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "n/a"
    med, nmad = np.median(v), mad_std(v)
    return f'{med:+.3f} px ({med * lr_pix:+.4f}"), NMAD {nmad:.3f} px, n={len(v)}'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", required=True, help="pairs directory with train/ and val/")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--n", type=int, default=500, help="random pairs to measure (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lr-fwhm", type=float, default=0.81, help='LR PSF FWHM, arcsec (LSST DP2 i ~0.81")'
    )
    parser.add_argument(
        "--hr-fwhm", type=float, default=0.10, help='HR PSF FWHM, arcsec (ACS F814W ~0.10")'
    )
    parser.add_argument("--out", help="per-pair CSV (default: <pairs>/alignment.csv)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pairs = Path(args.pairs)
    names = [(s, p.name) for s in args.splits for p in sorted((pairs / s / "hr").glob("*.fits"))]
    if args.n and len(names) > args.n:
        pick = np.random.default_rng(args.seed).choice(len(names), args.n, replace=False)
        names = [names[i] for i in sorted(pick)]
    rows, all_cents, lr_pix = [], [], 0.2
    for split, name in names:
        lr_pix, dy, dx, cents, lrfile = measure(
            pairs / split / "lr" / name, pairs / split / "hr" / name, args.lr_fwhm, args.hr_fwhm
        )
        cy, cx = np.median(cents, axis=0) if len(cents) else (np.nan, np.nan)
        tract = re.match(r"deep_coadd_(\d+)_", lrfile or name)
        rows.append(
            {
                "split": split,
                "name": name,
                "tract": tract.group(1) if tract else "",
                "xcorr_dy": dy,
                "xcorr_dx": dx,
                "centroid_dy": cy,
                "centroid_dx": cx,
                "n_sources": len(cents),
            }
        )
        all_cents.append(cents)
    out = Path(args.out) if args.out else pairs / "alignment.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cents = np.concatenate(all_cents) if all_cents else np.empty((0, 2))
    col = lambda k: [r[k] for r in rows]  # noqa: E731
    shift = np.hypot(np.array(col("xcorr_dy")), np.array(col("xcorr_dx")))
    print(f'{len(rows)} pairs from {pairs} (LR pixel {lr_pix:.3f}")')
    print("cross-correlation   dy:", summarize(col("xcorr_dy"), lr_pix))
    print("                    dx:", summarize(col("xcorr_dx"), lr_pix))
    print("source centroids    dy:", summarize(cents[:, 0], lr_pix), "(all sources)")
    print("                    dx:", summarize(cents[:, 1], lr_pix))
    print(
        f"pairs with |xcorr offset| > 0.25 LR px: {np.mean(shift > 0.25):.1%}; "
        f"> 0.5 px: {np.mean(shift > 0.5):.1%}"
    )
    for t in sorted({r["tract"] for r in rows}):
        sel = [r for r in rows if r["tract"] == t]
        dy = np.median([r["xcorr_dy"] for r in sel])
        dx = np.median([r["xcorr_dx"] for r in sel])
        print(
            f"  tract {t}: {len(sel):4d} pairs, median xcorr offset (dy, dx) = "
            f"({dy:+.3f}, {dx:+.3f}) px"
        )
    print(f"per-pair results: {out}")


if __name__ == "__main__":
    main()
