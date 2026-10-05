"""Compare super-resolution models with the NEO paper's morphology evaluation.

For every pair in a split that all models have predicted: the HST cutout (central 600 px, SEP
background subtracted) is cataloged with the paper's detection and deblending, and that same
segmentation is used to measure each model's output (identically background subtracted) and the
LR cutout (central 100 px, segmap reprojected to the coarse grid). Every image is measured in nJy
per pixel: HR and LR cutouts are converted with their own NJYPERPX card (neo.preprocess.pairs;
1.0 for nJy pairs) and predictions are already nJy (neo.eval.predict). Writes:
  sources.csv  per-source metric values for every image set
  table4.csv / table4.md   paper Table 4 statistics (median, bootstrap 95% CI, NMAD; mean, std)
  paper_q.csv  q as the paper's code computed it (--paper-mode only)
  gains.csv    per model: share of sources it improves over the LR image (paper's gain metric)
  pairwise.csv per model pair: share of sources where the first model is closer to HST
  table4.png   median bias with 95% CI per parameter and image set
All of it, with the run's parameters and each model's checkpoint, is tracked on Comet.

--subset select|report|all splits pairs deterministically (20% select / 80% report), so checkpoint
selection and the reported numbers never use the same pairs. With <split>/groups.csv (written by
neo.preprocess.leakage) whole groups of overlapping cutouts are assigned together, so the two
subsets share no sky; without it, pairs are assigned by name.

--paper-mode reproduces the procedure behind the paper's Table 4 (its Create Catalogs and Table-4
notebooks): HST and LR measured as clip(x, 0, p99.999) per cutout (the training dataset's
clip), no background subtraction (SR as predicted); smoothing FWHM and minimum area matched on
the sky to the paper's 3 px and 100 px at 0.028" (71 px at 0.0333"); LR shapes from its smoothed
copy; every source scored (no ellipticity cut); table4.md shows median +/- std (the paper's +/-)
and q also as the paper's code computed it (metrics.paper_q_statistic, also in paper_q.csv).
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
from neo.eval.catalogs import (  # noqa: E402
    KERNEL_FWHM,
    NPIXELS,
    PAPER_LR_PIXEL_ARCSEC,
    catalog_set,
    default_threshold,
    paper_fwhm,
    paper_npixels,
)
from neo.eval.postprocess import (  # noqa: E402
    HR_SIZE,
    LR_SIZE,
    balance_noise,
    center_crop,
    njy_per_px,
    paper_clip,
    subtract_background,
)
from neo.eval.subsets import in_subset, load_groups, pair_id  # noqa: E402,F401
from neo.eval.tracking import start_experiment  # noqa: E402

LR_KEY = "lr"
PAPER_Q_KEY = "ellipticity_bias"  # per-source B of the paper code's q (paper mode only)
DEFAULT_MIN_ELLIPTICITY = 0.1
# Options --paper-mode sets itself (module docstring).
PAPER_MODE_FIXED = ("npixels", "threshold", "nsigma", "min_ellipticity", "balance_noise")


def pixel_arcsec(header) -> float:
    return float(np.mean(proj_plane_pixel_scales(WCS(header)))) * 3600


def threshold_for(header, override, nsigma, hst) -> float:
    """HST detection threshold in nJy per HR pixel."""
    if nsigma is not None:
        import sep

        return nsigma * sep.Background(np.ascontiguousarray(hst)).globalrms
    if override is not None:
        return override
    return default_threshold(float(header["HRSCALE"]), pixel_arcsec(header))


def load_cutout(path: Path, size: int, paper_mode: bool):
    """Central `size` px of a pair cutout in nJy per pixel (paper mode: clipped as in training)."""
    data, header = fits.getdata(path, header=True)
    crop = center_crop(data, size)
    if paper_mode:
        crop = paper_clip(crop)  # in stored units, exactly as the dataset clips
    return np.asarray(crop, dtype=np.float64) * njy_per_px(header), header


def process(job):
    name, split, preds, opts = job
    warnings.simplefilter("ignore")
    paper = opts["paper_mode"]
    hst, hr_header = load_cutout(split / "hr" / name, HR_SIZE, paper)
    lr, lr_header = load_cutout(split / "lr" / name, LR_SIZE, paper)
    if not paper:
        hst = subtract_background(hst)
    rng = np.random.default_rng(int(hashlib.md5(name.encode()).hexdigest(), 16) % 2**32)
    srs = {}
    for model, directory in preds.items():
        sr = np.asarray(fits.getdata(directory / name), dtype=np.float64)  # nJy (predict.py)
        if opts["balance_noise"]:
            sr = balance_noise(sr, rng=rng)
        srs[model] = sr if paper else subtract_background(sr)
    settings = {
        "threshold": threshold_for(hr_header, opts["threshold"], opts["nsigma"], hst),
        "npixels": opts["npixels"],
        "fwhm": KERNEL_FWHM,
        "lr_fwhm": None,
    }
    if paper:
        settings["npixels"] = paper_npixels(pixel_arcsec(hr_header))
        settings["fwhm"] = paper_fwhm(pixel_arcsec(hr_header))
        settings["lr_fwhm"] = paper_fwhm(pixel_arcsec(lr_header), PAPER_LR_PIXEL_ARCSEC)
    try:
        result = catalog_set(hst, srs, lr, factor=opts["factor"], **settings)
    except Exception as exc:  # noqa: BLE001 - one bad cutout must not stop the run
        return name, None, f"{type(exc).__name__}: {exc}", settings
    return name, result, None, settings


def per_source_rows(name, hst_tbl, sr_tbls, lr_tbl, factor, paper_mode=False):
    sets = {**sr_tbls, LR_KEY: lr_tbl}
    vals = {
        k: metrics.per_source(hst_tbl, t, factor if k == LR_KEY else 1.0) for k, t in sets.items()
    }
    errs = {k: metrics.errors(hst_tbl, t, factor if k == LR_KEY else 1.0) for k, t in sets.items()}
    ebias = {k: metrics.ellipticity_bias(hst_tbl, t) for k, t in sets.items()} if paper_mode else {}
    rows = []
    for i in range(len(hst_tbl)):
        row = {
            "name": name,
            "label": int(hst_tbl["label"][i]),
            "hst_flux": float(hst_tbl["segment_flux"][i]),
            # the paper's magnitude axis is HST kron_flux; in nJy, AB mag = 31.4 - 2.5 log10
            "hst_kron_flux": float(hst_tbl["kron_flux"][i]),
            "hst_half_light_radius": float(hst_tbl["half_light_radius"][i]),
            "hst_ellipticity": float(hst_tbl["ellipticity"][i]),
        }
        for k in sets:
            for p in metrics.PARAMETERS:
                row[f"{k}:{p}"] = float(vals[k][p][i])
                row[f"{k}:err:{p}"] = float(errs[k][p][i])
            if paper_mode:
                row[f"{k}:{PAPER_Q_KEY}"] = float(ebias[k][i])
        rows.append(row)
    return rows


def fmt(s: dict, paper_mode: bool = False) -> str:
    if not s.get("n"):
        return "n/a"
    if paper_mode:
        return f"{s['median']:+.3g} ± {s['std']:.3g}"
    return (
        f"{s['median']:+.3g} [{s['median_ci_lo']:+.3g}, {s['median_ci_hi']:+.3g}] "
        f"(NMAD {s['nmad']:.3g})"
    )


def span(values) -> str:
    """One value, or the range of per-pair values, for the report."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "none"
    lo, hi = min(vals), max(vals)
    return f"{lo:.4g}" if np.isclose(lo, hi, rtol=1e-6) else f"{lo:.4g}-{hi:.4g}"


def mode_lines(args, used) -> list[str]:
    """table4.md lines stating how the images were measured."""
    if not args.paper_mode:
        return [
            "Default mode: HST and SR SEP background subtracted, LR raw; all images in nJy per "
            f"pixel. Detection threshold {span(used['threshold'])} nJy per HR px, npixels "
            f"{span(used['npixels'])}, smoothing FWHM {KERNEL_FWHM:g} px.",
        ]
    return [
        "Paper mode: the procedure behind the NEO paper's Table 4. HST and LR measured as "
        "clip(x, 0, p99.999) per cutout (the training dataset's clip) with no background "
        "subtraction, SR as predicted; all images in nJy per pixel. Detection on HST smoothed by "
        f"a 3x3 Gaussian of FWHM {span(used['fwhm'])} px (as wide on the sky as the paper kernel), "
        f'threshold {span(used["threshold"])} nJy per HR px (0.0069126 e-/s per 0.03" px), '
        f'npixels {span(used["npixels"])} (100 at 0.028"); LR shapes from its copy smoothed with '
        f'FWHM {span(used["lr_fwhm"])} LR px (as wide on the sky as 3 px at 0.168").',
    ]


def prediction_lines(args, checkpoints) -> list[str]:
    """table4.md lines naming each model's predictions; paper mode flags non-paper generator modes.

    The paper generated its SR images with the generator in train mode at batch size 1
    (neo.eval.predict --gen-mode train); NEOGMODE is the last field of each checkpoint entry.
    """
    lines = ["Predictions: " + "; ".join(f"{m} = {c}" for m, c in checkpoints.items()) + "."]
    modes = {m: c.rsplit("/", 1)[-1] for m, c in checkpoints.items()}
    other = [f"{m} ({g})" for m, g in modes.items() if g != "train"]
    if args.paper_mode and other:
        lines.append(
            f"Not the paper's generator mode for {', '.join(other)}: the paper's SR images came "
            "from train mode at batch 1 (predict.py --gen-mode train)."
        )
    return lines


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

    Returns {model: "run/checkpoint/step/generator mode"}. Stops on predictions from another
    pairs build, not in nJy, or a directory mixing checkpoints or generator modes, which would
    otherwise be scored against the wrong cutouts or in the wrong units.
    """
    checkpoints = {}
    for model, directory in preds.items():
        seen = set()
        for name in names:
            header = fits.getheader(directory / name)
            hr_header = fits.getheader(split / "hr" / name)
            expected = pair_id(hr_header)
            if header.get("PAIRID") != expected:
                raise SystemExit(
                    f"{directory / name} shows {header.get('PAIRID')!r} but {split / 'hr' / name} "
                    f"is {expected!r}: predictions from another pairs build? Re-run predict.py."
                )
            if header.get("BUNIT") != "nJy" and njy_per_px(hr_header) != 1.0:
                raise SystemExit(
                    f"{directory / name} is in {header.get('BUNIT')!r}, not nJy: predicted before "
                    "predict.py converted paper-unit pairs to nJy? Re-run predict.py."
                )
            keys = ("NEORUN", "NEOCKPT", "NEOSTEP", "NEOGMODE")
            seen.add(tuple(str(header.get(k)) for k in keys))
        if len(seen) != 1:
            raise SystemExit(
                f"{directory} mixes predictions from several checkpoints or modes: {sorted(seen)}"
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
    used,
    paper_q,
):
    mode = "paper" if args.paper_mode else "default"
    experiment = start_experiment(
        f"compare {split.name}-{args.subset} ({mode} mode): {' vs '.join(preds)}",
        ["comparison", args.subset, f"{mode}-mode", *preds, *args.tag],
    )
    experiment.log_parameters(
        {
            "split_dir": str(split.resolve()),
            "subset": args.subset,
            "subset_by": "sky group" if groups else "name",
            "models": ",".join(preds),
            "mode": mode,
            "paper_mode": args.paper_mode,
            "units": "nJy per pixel",
            "background_subtraction": "none" if args.paper_mode else "SEP (HST, SR)",
            "clip": "0..p99.999 (HST, LR)" if args.paper_mode else "none",
            "factor": args.factor,
            "npixels": span(used["npixels"]),
            "kernel_fwhm_px": span(used["fwhm"]),
            "lr_kernel_fwhm_px": span(used["lr_fwhm"]),
            "threshold": args.threshold if args.threshold is not None else "paper (per pair)",
            "threshold_njy": span(used["threshold"]),
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
    for q in paper_q:
        if q.get("n"):
            for k in ("q68", "median", "std", "n"):
                experiment.log_metric(f"{q['image']}/q_paper_code/{k}", q[k])
    for g in gains:
        if g.get("n"):
            for k in ("frac_improved", "mean_log_gain", "n"):
                experiment.log_metric(f"{g['model']}/{g['parameter']}/gain_{k}", g[k])
    for pw in pairwise:
        experiment.log_metric(
            f"{pw['a']}_vs_{pw['b']}/{pw['parameter']}/frac_a_closer", pw["frac_a_closer"]
        )
    for fname in ("table4.csv", "paper_q.csv", "gains.csv", "pairwise.csv", "sources.csv"):
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
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="the procedure behind the paper's Table 4 (module docstring)",
    )
    parser.add_argument("--npixels", type=int, help=f"minimum source area (default {NPIXELS})")
    parser.add_argument(
        "--threshold",
        type=float,
        help="HST detection threshold in nJy per HR pixel (default: the paper's, converted)",
    )
    parser.add_argument("--nsigma", type=float, help="threshold as N x HST background rms instead")
    parser.add_argument(
        "--min-ellipticity",
        type=float,
        help="orientation is only scored where HST ellipticity >= this "
        f"(default {DEFAULT_MIN_ELLIPTICITY})",
    )
    parser.add_argument("--balance-noise", action="store_true", help="refill zeros in SR outputs")
    parser.add_argument("--tag", action="append", default=[], help="extra Comet tag (repeatable)")
    parser.add_argument("--no-comet", action="store_true", help="do not track the run on Comet")
    return parser.parse_args(argv)


def resolve_mode(args) -> None:
    """Fill the mode's defaults; --paper-mode refuses options that would change its procedure."""
    if args.paper_mode:
        values = {k: getattr(args, k) for k in PAPER_MODE_FIXED}
        # identity, not ==: an explicit 0 (e.g. --threshold 0) counts as given
        given = [k for k, v in values.items() if v is not None and v is not False]
        if given:
            flags = ", ".join("--" + k.replace("_", "-") for k in given)
            raise SystemExit(f"--paper-mode sets these itself: drop {flags}")
        args.min_ellipticity = 0.0
    else:
        args.npixels = NPIXELS if args.npixels is None else args.npixels
        if args.min_ellipticity is None:
            args.min_ellipticity = DEFAULT_MIN_ELLIPTICITY


def main(argv=None) -> None:
    args = parse_args(argv)
    resolve_mode(args)
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
    for line in prediction_lines(args, checkpoints)[1:]:
        print(f"  {line}")
    opts = {
        "factor": args.factor,
        "npixels": args.npixels,
        "threshold": args.threshold,
        "nsigma": args.nsigma,
        "balance_noise": args.balance_noise,
        "paper_mode": args.paper_mode,
    }
    mode = "paper" if args.paper_mode else "default"
    print(f"{len(names)} pairs ({args.subset}, {mode} mode) x models {list(preds)}")

    rows, kept, t0 = [], 0, time.time()
    used = {"threshold": [], "npixels": [], "fwhm": [], "lr_fwhm": []}
    jobs = [(n, split, preds, opts) for n in names]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(process, jobs, chunksize=4)
        for k, (name, result, error, settings) in enumerate(results, 1):
            for key, value in settings.items():
                used[key].append(value)
            if error:
                print(f"  {name}: {error}")
            elif result is not None:
                kept += 1
                rows.extend(per_source_rows(name, *result, args.factor, args.paper_mode))
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
    if args.paper_mode:
        elongated = np.ones(len(rows), dtype=bool)  # Table 4 scored every source
    else:
        elongated = column(rows, "hst_ellipticity") >= args.min_ellipticity
    table4, gains, pairwise, paper_q = [], [], [], []
    for s in sets:
        entry = {"image": s}
        for p in metrics.PARAMETERS:
            mask = elongated if p == "orientation" else None
            entry[p] = metrics.summarize(column(rows, f"{s}:{p}", mask))
        table4.append(entry)
        if args.paper_mode:
            q = metrics.paper_q_statistic(column(rows, f"{s}:{PAPER_Q_KEY}"))
            paper_q.append({"image": s, **q})
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
    for fname, data in (("paper_q.csv", paper_q), ("gains.csv", gains), ("pairwise.csv", pairwise)):
        if data:
            with open(out / fname, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)

    labels = [f"{p} (b/a)" if p == "q" and args.paper_mode else p for p in metrics.PARAMETERS]
    header = "| image | " + " | ".join(labels) + " |"
    if args.paper_mode:
        scored = f"every source scored ({int(elongated.sum())} for orientation, no ellipticity cut)"
        stat = (
            "Median ± std over all sources (std is the paper's ±). Relative bias for R_e, FWHM, "
            "C75/25 and flux (LR R_e and FWHM x6, as in the paper's text and figures; its printed "
            "HSC R_e and FWHM entries omit the x6); q = b/a with absolute bias; "
            "S = 1 - |cos dtheta| for orientation (0 = perfect)."
        )
    else:
        scored = (
            f"orientation scored on {int(elongated.sum())} sources with HST ellipticity >= "
            f"{args.min_ellipticity}"
        )
        stat = (
            "Median [bootstrap 95% CI] (NMAD). Relative bias for R_e, FWHM, C75/25 and flux; "
            "absolute bias for q; S = 1 - |cos dtheta| for orientation (0 = perfect)."
        )
    lines = [
        f"# Morphology comparison ({args.subset} subset, {mode} mode)",
        "",
        *mode_lines(args, used),
        *prediction_lines(args, checkpoints),
        f"{kept} cutout sets, {len(rows)} HST-detected sources; {scored}.",
        stat,
        "",
        header,
        "|" + "---|" * (len(metrics.PARAMETERS) + 1),
    ]
    for entry in table4:
        cells = " | ".join(fmt(entry[p], args.paper_mode) for p in metrics.PARAMETERS)
        lines.append(f"| {entry['image']} | {cells} |")
    if args.paper_mode:
        lines += [
            "",
            "q as the paper's code computed it (the q column of its Table 4): photutils "
            "ellipticity e = 1 - b/a, B = e_X - e_HST, entry = 68th percentile of "
            "|B - median(B)| ± std(B). median(B) equals minus the q = b/a bias above.",
            "",
            "| image | q (paper code) | median(B) |",
            "|---|---|---|",
        ]
        for q in paper_q:
            cell = f"{q['q68']:.3g} ± {q['std']:.3g}" if q.get("n") else "n/a"
            med = f"{q['median']:+.3g}" if q.get("n") else "n/a"
            lines.append(f"| {q['image']} | {cell} | {med} |")
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
    fig = bias_figure(
        table4, f"{split.name} {args.subset} ({mode} mode): {kept} sets, {len(rows)} sources"
    )
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
            used,
            paper_q,
        )
    plt.close(fig)
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
