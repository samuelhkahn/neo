"""Compare super-resolution models with the NEO paper's morphology evaluation.

For every pair in a split that all models have predicted: the HST cutout (central 600 px, SEP
background subtracted) is cataloged with the paper's detection and deblending, and that same
segmentation is used to measure each model's output (identically background subtracted) and the
LR cutout (central 100 px, segmap reprojected to the coarse grid). Writes:
  sources.csv  per-source metric values for every image set
  table4.csv / table4.md   paper Table 4 statistics (median, bootstrap 95% CI, NMAD; mean, std)
  gains.csv    per model: share of sources it improves over the LR image (paper's gain metric)
  pairwise.csv per model pair: share of sources where the first model is closer to HST
  table4.png   median bias with 95% CI per parameter and image set
All of it, with the run's parameters and each model's checkpoint, is tracked on Comet.

--subset select|report|all splits pairs deterministically (20% select / 80% report), so checkpoint
selection and the reported numbers never use the same pairs. With <split>/groups.csv (written by
neo.preprocess.leakage) whole groups of overlapping cutouts are assigned together, so the two
subsets share no sky; without it, pairs are assigned by name.
"""

import argparse
import csv
import hashlib
import itertools
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from astropy.wcs.utils import proj_plane_pixel_scales  # noqa: E402

from neo.eval import metrics  # noqa: E402
from neo.eval.catalogs import NPIXELS, catalog_set, default_threshold  # noqa: E402
from neo.eval.postprocess import (  # noqa: E402
    HR_SIZE,
    LR_SIZE,
    balance_noise,
    center_crop,
    subtract_background,
)
from neo.eval.subsets import in_subset, load_groups, pair_id  # noqa: E402,F401
from neo.eval.tracking import start_experiment  # noqa: E402

LR_KEY = "lr"


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


def bias_figure(table4, title):
    """Median (95% CI) per parameter, one marker per image set."""
    params = list(metrics.PARAMETERS)
    fig, axes = plt.subplots(1, len(params), figsize=(3 * len(params), 3.6), sharey=False)
    for ax, p in zip(axes, params, strict=True):
        for i, entry in enumerate(table4):
            st = entry[p]
            if not st.get("n"):
                continue
            err = [[st["median"] - st["median_ci_lo"]], [st["median_ci_hi"] - st["median"]]]
            ax.errorbar(i, st["median"], yerr=err, fmt="o", capsize=3)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(len(table4)), [e["image"] for e in table4], rotation=45)
        ax.set_title(p)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def verify_predictions(split: Path, preds: dict, names: list) -> dict:
    """Each prediction must show its HR cutout's sky, and each directory hold one checkpoint.

    Returns {model: "run/checkpoint/step"}. Stops on predictions from another pairs build or a
    directory mixing checkpoints, which would otherwise be scored against the wrong cutouts.
    """
    checkpoints = {}
    for model, directory in preds.items():
        seen = set()
        for name in names:
            header = fits.getheader(directory / name)
            expected = pair_id(fits.getheader(split / "hr" / name))
            if header.get("PAIRID") != expected:
                raise SystemExit(
                    f"{directory / name} shows {header.get('PAIRID')!r} but {split / 'hr' / name} "
                    f"is {expected!r}: predictions from another pairs build? Re-run predict.py."
                )
            seen.add(tuple(str(header.get(k)) for k in ("NEORUN", "NEOCKPT", "NEOSTEP")))
        if len(seen) != 1:
            raise SystemExit(
                f"{directory} mixes predictions from several checkpoints: {sorted(seen)}"
            )
        checkpoints[model] = "/".join(seen.pop())
    return checkpoints


def track(
    args,
    split,
    preds,
    names,
    kept,
    rows,
    elongated,
    table4,
    gains,
    pairwise,
    out,
    fig,
    groups,
    checkpoints,
):
    experiment = start_experiment(
        f"compare {split.name}-{args.subset}: {' vs '.join(preds)}",
        ["comparison", args.subset, *preds, *args.tag],
    )
    experiment.log_parameters(
        {
            "split_dir": str(split.resolve()),
            "subset": args.subset,
            "subset_by": "sky group" if groups else "name",
            "models": ",".join(preds),
            "factor": args.factor,
            "npixels": args.npixels,
            "threshold": args.threshold if args.threshold is not None else "paper (per pair)",
            "nsigma": args.nsigma,
            "min_ellipticity": args.min_ellipticity,
            "balance_noise": args.balance_noise,
            "n_pairs": len(names),
            "n_sets_kept": kept,
            "n_sources": len(rows),
            "n_sources_orientation": int(elongated.sum()),
            "out": str(out.resolve()),
            **{f"pred_dir/{m}": str(d.resolve()) for m, d in preds.items()},
            **{f"checkpoint/{m}": c for m, c in checkpoints.items()},
        }
    )
    for entry in table4:
        for p, st in entry.items():
            if p != "image" and st.get("n"):
                for k in ("median", "median_ci_lo", "median_ci_hi", "nmad", "mean", "std", "n"):
                    experiment.log_metric(f"{entry['image']}/{p}/{k}", st[k])
    for g in gains:
        if g.get("n"):
            for k in ("frac_improved", "mean_log_gain", "n"):
                experiment.log_metric(f"{g['model']}/{g['parameter']}/gain_{k}", g[k])
    for pw in pairwise:
        experiment.log_metric(
            f"{pw['a']}_vs_{pw['b']}/{pw['parameter']}/frac_a_closer", pw["frac_a_closer"]
        )
    for fname in ("table4.csv", "gains.csv", "pairwise.csv", "sources.csv"):
        if (out / fname).exists():
            experiment.log_table(str(out / fname))
    experiment.log_asset(str(out / "table4.md"))
    experiment.log_text((out / "table4.md").read_text())
    experiment.log_figure(figure_name="table4 median bias", figure=fig)
    experiment.end()


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
    parser.add_argument("--tag", action="append", default=[], help="extra Comet tag (repeatable)")
    parser.add_argument("--no-comet", action="store_true", help="do not track the run on Comet")
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
    groups = load_groups(split)
    if args.subset != "all" and not groups:
        print(f"no {split / 'groups.csv'}: assigning {args.subset} by name, not by sky group")
    names = sorted(p.name for p in (split / "hr").glob("*.fits"))
    names = [n for n in names if in_subset(n, args.subset, groups)]
    names = [n for n in names if all((d / n).exists() for d in preds.values())][: args.limit]
    if not names:
        raise SystemExit("no pairs predicted by every model in this subset")
    checkpoints = verify_predictions(split, preds, names)
    for model, checkpoint in checkpoints.items():
        print(f"  {model}: {checkpoint}")
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
    fig = bias_figure(table4, f"{split.name} {args.subset}: {kept} sets, {len(rows)} sources")
    fig.savefig(out / "table4.png", dpi=120)
    if not args.no_comet:
        track(
            args,
            split,
            preds,
            names,
            kept,
            rows,
            elongated,
            table4,
            gains,
            pairwise,
            out,
            fig,
            groups,
            checkpoints,
        )
    plt.close(fig)
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
