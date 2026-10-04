"""Compare super-resolution models with the NEO paper's morphology evaluation.

For every pair in a split that all models have predicted: the HST cutout (central 600 px, SEP
background subtracted) is cataloged with the paper's detection and deblending, and that same
segmentation is used to measure each model's output (identically background subtracted) and the
LR cutout (central 100 px, segmap reprojected to the coarse grid). Writes:
  sources.csv  per-source metric values for every image set
  table4.csv / table4.md   paper Table 4 statistics (median, bootstrap 95% CI, NMAD; mean, std)
  gains.csv    per model: share of sources it improves over the LR image (paper's gain metric)
  pairwise.csv per model pair: share of sources where the first model is closer to HST

--subset select|report|all splits pairs deterministically by name (20% select / 80% report), so
checkpoint selection and the reported numbers never use the same pairs.
"""

import argparse
import csv
import hashlib
import itertools
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from neo.eval import metrics
from neo.eval.catalogs import NPIXELS, catalog_set, default_threshold
from neo.eval.postprocess import HR_SIZE, LR_SIZE, balance_noise, center_crop, subtract_background

LR_KEY = "lr"


def in_subset(name: str, subset: str) -> bool:
    if subset == "all":
        return True
    selected = int(hashlib.md5(name.encode()).hexdigest(), 16) % 5 == 0
    return selected if subset == "select" else not selected


def threshold_for(hr_path: Path, override, nsigma, hst) -> float:
    if nsigma is not None:
        import sep

        return nsigma * sep.Background(np.ascontiguousarray(hst)).globalrms
    if override is not None:
        return override
    header = fits.getheader(hr_path)
    scale = float(np.mean(proj_plane_pixel_scales(WCS(header)))) * 3600
    return default_threshold(float(header["HRSCALE"]), scale)


def process(job):
    name, split, preds, opts = job
    warnings.simplefilter("ignore")
    hst = subtract_background(center_crop(fits.getdata(split / "hr" / name), HR_SIZE))
    lr = np.asarray(center_crop(fits.getdata(split / "lr" / name), LR_SIZE), dtype=np.float64)
    rng = np.random.default_rng(int(hashlib.md5(name.encode()).hexdigest(), 16) % 2**32)
    srs = {}
    for model, directory in preds.items():
        sr = np.asarray(fits.getdata(directory / name), dtype=np.float64)
        if opts["balance_noise"]:
            sr = balance_noise(sr, rng=rng)
        srs[model] = subtract_background(sr)
    threshold = threshold_for(split / "hr" / name, opts["threshold"], opts["nsigma"], hst)
    try:
        result = catalog_set(hst, srs, lr, threshold, opts["factor"], opts["npixels"])
    except Exception as exc:  # noqa: BLE001 - one bad cutout must not stop the run
        return name, None, f"{type(exc).__name__}: {exc}"
    return name, result, None


def per_source_rows(name, hst_tbl, sr_tbls, lr_tbl, factor):
    sets = {**sr_tbls, LR_KEY: lr_tbl}
    vals = {
        k: metrics.per_source(hst_tbl, t, factor if k == LR_KEY else 1.0) for k, t in sets.items()
    }
    errs = {k: metrics.errors(hst_tbl, t, factor if k == LR_KEY else 1.0) for k, t in sets.items()}
    rows = []
    for i in range(len(hst_tbl)):
        row = {
            "name": name,
            "label": int(hst_tbl["label"][i]),
            "hst_flux": float(hst_tbl["segment_flux"][i]),
            "hst_half_light_radius": float(hst_tbl["half_light_radius"][i]),
            "hst_ellipticity": float(hst_tbl["ellipticity"][i]),
        }
        for k in sets:
            for p in metrics.PARAMETERS:
                row[f"{k}:{p}"] = float(vals[k][p][i])
                row[f"{k}:err:{p}"] = float(errs[k][p][i])
        rows.append(row)
    return rows


def fmt(s: dict) -> str:
    if not s.get("n"):
        return "n/a"
    return (
        f"{s['median']:+.3g} [{s['median_ci_lo']:+.3g}, {s['median_ci_hi']:+.3g}] "
        f"(NMAD {s['nmad']:.3g})"
    )


def column(rows, key, mask=None):
    x = np.array([r[key] for r in rows], dtype=float)
    return x if mask is None else x[mask]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split-dir", required=True, help="pairs split with lr/ and hr/")
    parser.add_argument(
        "--pred", action="append", required=True, metavar="MODEL=DIR", help="repeat per model"
    )
    parser.add_argument("--out", required=True, help="output directory for the report")
    parser.add_argument("--subset", choices=["report", "select", "all"], default="report")
    parser.add_argument("--limit", type=int, help="first N common pairs only")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--factor", type=int, default=6, help="HR/LR pixel scale ratio")
    parser.add_argument("--npixels", type=int, default=NPIXELS)
    parser.add_argument(
        "--threshold", type=float, help="HST detection threshold in pair units (default: paper's)"
    )
    parser.add_argument("--nsigma", type=float, help="threshold as N x HST background rms instead")
    parser.add_argument(
        "--min-ellipticity",
        type=float,
        default=0.1,
        help="orientation is only scored where HST ellipticity >= this",
    )
    parser.add_argument("--balance-noise", action="store_true", help="refill zeros in SR outputs")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    split = Path(args.split_dir)
    preds = {}
    for item in args.pred:
        model, _, directory = item.partition("=")
        preds[model] = Path(directory)
    if LR_KEY in preds:
        raise SystemExit(f"'{LR_KEY}' is reserved for the low-resolution baseline")
    names = sorted(p.name for p in (split / "hr").glob("*.fits"))
    names = [n for n in names if in_subset(n, args.subset)]
    names = [n for n in names if all((d / n).exists() for d in preds.values())][: args.limit]
    if not names:
        raise SystemExit("no pairs predicted by every model in this subset")
    opts = {
        "factor": args.factor,
        "npixels": args.npixels,
        "threshold": args.threshold,
        "nsigma": args.nsigma,
        "balance_noise": args.balance_noise,
    }
    print(f"{len(names)} pairs ({args.subset}) x models {list(preds)}")

    rows, kept, t0 = [], 0, time.time()
    jobs = [(n, split, preds, opts) for n in names]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, (name, result, error) in enumerate(pool.map(process, jobs, chunksize=4), 1):
            if error:
                print(f"  {name}: {error}")
            elif result is not None:
                kept += 1
                rows.extend(per_source_rows(name, *result, args.factor))
            if k % 100 == 0 or k == len(jobs):
                elapsed = time.time() - t0
                print(
                    f"  {k}/{len(jobs)} pairs, {kept} kept, {len(rows)} sources ({elapsed:.0f} s)"
                )
    if not rows:
        raise SystemExit("no sources survived cataloging")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sources.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sets = list(preds) + [LR_KEY]
    elongated = column(rows, "hst_ellipticity") >= args.min_ellipticity
    table4, gains, pairwise = [], [], []
    for s in sets:
        entry = {"image": s}
        for p in metrics.PARAMETERS:
            mask = elongated if p == "orientation" else None
            entry[p] = metrics.summarize(column(rows, f"{s}:{p}", mask))
        table4.append(entry)
    for m in preds:
        for p in metrics.PARAMETERS:
            mask = elongated if p == "orientation" else None
            g = metrics.gain(
                column(rows, f"{LR_KEY}:err:{p}", mask), column(rows, f"{m}:err:{p}", mask)
            )
            gains.append({"model": m, "parameter": p, **g})
    for a, b in itertools.combinations(preds, 2):
        for p in metrics.PARAMETERS:
            mask = elongated if p == "orientation" else None
            ea, eb = column(rows, f"{a}:err:{p}", mask), column(rows, f"{b}:err:{p}", mask)
            ok = np.isfinite(ea) & np.isfinite(eb)
            pairwise.append(
                {
                    "a": a,
                    "b": b,
                    "parameter": p,
                    "n": int(ok.sum()),
                    "frac_a_closer": float(np.mean(ea[ok] < eb[ok])) if ok.any() else float("nan"),
                }
            )

    with open(out / "table4.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "parameter",
                "n",
                "median",
                "median_ci_lo",
                "median_ci_hi",
                "nmad",
                "mean",
                "std",
            ]
        )
        for entry in table4:
            for p in metrics.PARAMETERS:
                s = entry[p]
                writer.writerow(
                    [entry["image"], p, s.get("n", 0)]
                    + [
                        s.get(k, "")
                        for k in ("median", "median_ci_lo", "median_ci_hi", "nmad", "mean", "std")
                    ]
                )
    for fname, data in (("gains.csv", gains), ("pairwise.csv", pairwise)):
        if data:
            with open(out / fname, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)

    header = "| image | " + " | ".join(metrics.PARAMETERS) + " |"
    lines = [
        f"# Morphology comparison ({args.subset} subset)",
        "",
        f"{kept} cutout sets, {len(rows)} HST-detected sources; orientation scored on "
        f"{int(elongated.sum())} sources with HST ellipticity >= {args.min_ellipticity}.",
        "Median [bootstrap 95% CI] (NMAD). Relative bias for R_e, FWHM, C75/25 and flux; absolute "
        "bias for q; S = 1 - |cos dtheta| for orientation (0 = perfect).",
        "",
        header,
        "|" + "---|" * (len(metrics.PARAMETERS) + 1),
    ]
    for entry in table4:
        lines.append(
            f"| {entry['image']} | " + " | ".join(fmt(entry[p]) for p in metrics.PARAMETERS) + " |"
        )
    lines += ["", "Share of sources each model measures closer to HST than the LR image does:", ""]
    lines.append("| model | " + " | ".join(metrics.PARAMETERS) + " |")
    lines.append("|" + "---|" * (len(metrics.PARAMETERS) + 1))
    for m in preds:
        cells = [g for g in gains if g["model"] == m]
        lines.append(
            f"| {m} | "
            + " | ".join(f"{100 * g['frac_improved']:.1f}%" if g.get("n") else "n/a" for g in cells)
            + " |"
        )
    (out / "table4.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
