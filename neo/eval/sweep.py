"""Score every saved checkpoint of one NEO training run on the same fixed set of val images.

The Comet validation curves are one random batch of 8 every 25 steps, so a single point scatters by
~4% on L1 and ~20% on segmap L1, and points inside discriminator-dominance bursts look best (smooth,
noise-free outputs). This scores each step_*.pt in a run's checkpoint directory on every image of
the run's validation set (the config's hst_path_val / hsc_path_val, i.e. val_select), with the
generator in train mode at batch size 1 as the paper generated its SR images (dropout on, batch norm
on each image's own statistics) and a fixed per-image seed, so every checkpoint sees the same images
with the same dropout masks and differences between checkpoints are paired.

Per checkpoint it records the losses as training defines them (neo/pix2pix.py: L1, segmap L1 and
VGG on the central 600 px in log space; adversarial and discriminator BCE with the checkpoint's own
PatchGAN), averaged over images with standard errors, and the fine-structure metric. For a run
trained with [NOISE] (neo/noise.py) the clean output is scored, and the adversarial and
discriminator columns judge it with the target's sky noise added, as training did (VGG stays on
the clean output: the sweep compares checkpoints by what they produce, not by a noise draw); the
same per-image seed also fixes that noise draw
(neo.eval.fine_structure: transfer T, coherence rho and power ratio R inside HST sources, k50, each
with a 68% bootstrap interval over sources as <key>_lo / <key>_hi, and the fractions of sources the
generator saturated or blew up, which are counted rather than summed) on the images in nJy. HST
sources inside the Gaia star masks (neo.eval.stars: stars moved between the HST and LSST epochs)
are left out of the fine-structure metric and counted as fine_n_star; --keep-stars keeps them (the
masks are still built and the stars per image counted, so the two runs differ only in which
sources the metric uses). The catalog ships with the code (neo/data/gaia_dr3_cosmos_stars.csv):
without it the sweep stops before any work, unless --keep-stars is given. --average A:B also
scores the uniform average of the step files from A to B, a free stand-in for an EMA of the
weights or a decaying learning rate.

--ensemble K (default 1; needs --mode train, in eval mode every pass is the same image) runs
every image K times, with seeds seed + i + 1000 k (k = 0 is the single pass, so its row is exactly
the K = 1 row), and also scores the stacks of the K draws (neo.eval.stack: the per-pixel median in
log space and the mean in linear flux) as extra rows labelled <checkpoint>:medianK and
<checkpoint>:meanK, with the same losses (the adversarial and discriminator losses judge the
stack) and fine-structure metrics; sweep.png draws them as dashed (median) and dotted (mean)
lines. Cost: K generator passes per image, plus the discriminator, VGG and fine-structure metric
three times per image instead of once (the stacks are built in float64 on the CPU), so at small K
the fixed part roughly triples.

Writes, in --out: sweep.csv (one row per checkpoint, plus the stack rows), per_image.csv (for
paired comparisons), fine_structure.json (T, rho, R per k), sweep.png, and settings.json.
Checkpoints already in sweep.csv with the same settings are skipped, so rerunning picks up only new
checkpoints (a checkpoint counts as done only when its single-pass and, with K > 1, both stack
rows are there; a rerun fills in the missing ones); a sweep.csv with other columns (an older
version of the metric) is refused before anything is loaded. Tracked on Comet. Training is not
touched: it reads checkpoints only.

Usage:  python -m neo.eval.sweep --config neo/configs/lux.ini --out $NEO_DATA/sweeps/lux \
            [--average 650000:850000] [--every 50000] [--limit 50] [--ensemble 8]
"""

import argparse
import configparser
import csv
import json
import os
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from astropy.io import fits  # noqa: E402
from torchvision.transforms import CenterCrop  # noqa: E402

from neo.data.augment import dataset_classes  # noqa: E402
from neo.eval import fine_structure, stack, stars  # noqa: E402
from neo.eval.postprocess import HR_SIZE, center_crop, njy_per_px, to_physical  # noqa: E402
from neo.eval.predict import pick_device  # noqa: E402
from neo.eval.tracking import start_experiment  # noqa: E402
from neo.models.patchgan import PatchGAN  # noqa: E402
from neo.models.registry import check_generator, make_generator  # noqa: E402
from neo.models.vgg19_loss import VGGLoss  # noqa: E402
from neo.noise import NoiseModel  # noqa: E402

STEP_FILE = re.compile(r"^step_(\d+)\.pt$")
PLAIN = re.compile(r"^step_\d+$")  # a numbered checkpoint's single-pass row
LOSSES = ("l1", "segmap_l1", "vgg", "adversarial", "discriminator")
FINE = ("k50", "T_band", "rho_band", "R_band", "saturated_fraction", "blown_fraction")
FINE_CI = ("k50", "T_band", "rho_band", "R_band")  # get <key>_lo / <key>_hi columns
# sweep.csv's columns, in summarize's order; a sweep.csv with other columns is from another
# version of the metric and is refused up front
COLUMNS = (
    "checkpoint",
    "step",
    "n_images",
    *[c for k in LOSSES for c in (k, f"{k}_se")],
    *FINE,
    *[c for k in FINE_CI for c in (f"{k}_lo", f"{k}_hi")],
    "fine_n_sources",
    "fine_n_saturated",
    "fine_n_blown",
    "fine_n_star",
)
STACKS = ("median", "mean")  # ensemble stacks scored besides the single pass (stack.IMAGES)
SATURATION_LOG = 0.99  # a pixel this close to the generator's tanh ceiling of 1 is saturated


def step_files(ckpt_dir: Path) -> dict:
    """{step: path} of the numbered checkpoints in ckpt_dir."""
    found = {}
    for path in Path(ckpt_dir).glob("step_*.pt"):
        match = STEP_FILE.match(path.name)
        if match:
            found[int(match.group(1))] = path
    return dict(sorted(found.items()))


def load_state(path: Path, config=None) -> dict:
    """One checkpoint; with config, refused unless written by the generator the config builds
    (neo and neo_relu share their keys, so load_state_dict alone would take either)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if config is not None:
        check_generator(state, config, path)
    return state


def average_states(paths, config=None) -> dict:
    """Uniform average of the generator and discriminator weights of several checkpoints.

    Floating-point tensors are averaged; integer buffers (BatchNorm's num_batches_tracked) are
    taken from the last file. The batch-norm running statistics are averaged too, though in train
    mode (as here) batch norm uses each image's own statistics.
    """
    total = {}
    for i, path in enumerate(paths):
        state = load_state(path, config)  # each file is checked: the average keeps no record
        for part in ("gen", "patch_gan"):
            dest = total.setdefault(part, {})
            for key, value in state[part].items():
                if value.is_floating_point():
                    dest[key] = value.double() if i == 0 else dest[key] + value.double()
                else:
                    dest[key] = value
    for part in total.values():
        for key, value in part.items():
            if value.is_floating_point():
                part[key] = (value / len(paths)).float()
    return total


def load_images(config, limit=None, star_options=None):
    """The run's validation images through its own val dataset class (hsc_hr_8bit included), in
    filename order, plus each HST cutout's central 600 px in nJy for the fine-structure metric
    and, with `star_options` (neo.eval.stars.star_tracks keywords), its Gaia star mask."""
    _, _, val_dataset = dataset_classes(config)
    hst_dir = Path(os.path.expandvars(config["DEFAULT"]["hst_path_val"]))
    hsc_dir = Path(os.path.expandvars(config["DEFAULT"]["hsc_path_val"]))
    dataset = val_dataset(
        hst_path=str(hst_dir),
        hsc_path=str(hsc_dir),
        hr_size=[HR_SIZE, HR_SIZE],
        lr_size=[100, 100],
        transform_type="ds9_scale",
        data_aug=False,
        experiment=None,
    )
    order = sorted(range(len(dataset)), key=lambda i: dataset.filenames[i])[:limit]
    images = []
    for i in order:
        hst, lr, hsc_hr, seg = dataset[i]
        if hst is None:
            continue
        name = dataset.filenames[i]
        raw, header = fits.getdata(hst_dir / name, header=True)
        njy = njy_per_px(header)
        hst_njy = center_crop(np.asarray(raw, dtype=np.float64), HR_SIZE) * njy
        star_mask, n_stars = None, 0
        if star_options is not None:
            tracks = stars.star_tracks(header, hst_njy.shape, **star_options)
            if tracks.notice:
                print(f"  {name}: {tracks.notice}")
            if len(tracks):
                star_mask, n_stars = tracks.mask(), len(tracks)
        images.append(
            {
                "name": name,
                "hst": hst,
                "lr": lr,
                "hsc_hr": hsc_hr,
                "seg": seg,
                "njy": njy,
                "hst_njy": hst_njy,
                "star_mask": star_mask,
                "n_stars": n_stars,
            }
        )
    return images


@torch.no_grad()
def evaluate(
    gen,
    disc,
    vgg,
    images,
    hst_sources,
    spectra,
    device,
    seed=0,
    mode="train",
    noise=None,
    ensemble=1,
):
    """Per-image losses and the pooled fine-structure result for one set of weights, as
    {variant: (rows, fine)}: variant "" is the single pass; with ensemble K > 1 also ":medianK"
    and ":meanK", the stacks of K passes (module docstring).

    noise: the run's NoiseModel ([NOISE]) or None; with it the discriminator sees the clean
    output plus the target's sky noise, as in training."""
    gen.train(mode == "train")
    disc.train(mode == "train")
    bce = nn.BCEWithLogitsLoss()
    crop = CenterCrop(HR_SIZE)
    variants = variant_labels(ensemble)
    accs = {v: fine_structure.Accumulator(spectra) for v in variants}
    rows = {v: [] for v in variants}
    saturation_log = float(to_physical(np.full((1, 768, 768), SATURATION_LOG))[0, 0, 0])
    for i, im in enumerate(images):
        lr = im["lr"][None, None].to(device)
        r = crop(im["hst"][None, None].to(device))
        s = crop(im["seg"][None, None].to(device))
        c = crop(im["hsc_hr"][None, None].to(device))

        def judged(f, r=r, s=s):
            return f if noise is None else noise.add_noise(f, noise.target_sigma(r, s))

        torch.manual_seed(seed + i)  # the same dropout masks for this image in every checkpoint
        fake = gen(lr)
        outputs = {"": (fake, judged(crop(fake)))}  # the noise draw follows the pass, as before
        if ensemble > 1:
            draws = [fake.float().cpu()]
            for k in range(1, ensemble):
                torch.manual_seed(seed + i + 1000 * k)
                draws.append(gen(lr).float().cpu())
            stacked = stack.stack(
                torch.stack(draws)
            )  # on the CPU: the log/linear maps need float64
            torch.manual_seed(seed + i + 1000 * ensemble)  # one fixed noise draw for the stacks
            for kind in STACKS:
                out = stacked[kind].to(fake.dtype).to(device)
                outputs[f":{kind}{ensemble}"] = (out, judged(crop(out)))
        for variant, (out, judge) in outputs.items():
            f = crop(out)
            err = (f - r).abs()
            fake_logits, real_logits = disc(judge, c).flatten(), disc(r, c).flatten()
            rows[variant].append(
                {
                    "image": im["name"],
                    "l1": err.mean().item(),
                    "segmap_l1": (err * s).sum().item() / s.sum().item() if s.sum() > 0 else np.nan,
                    "vgg": float(vgg(f, r)),
                    "adversarial": bce(fake_logits, torch.ones_like(fake_logits)).item(),
                    "discriminator": (
                        bce(real_logits, torch.ones_like(real_logits))
                        + bce(fake_logits, torch.zeros_like(fake_logits))
                    ).item(),
                }
            )
            sr_njy = to_physical(out[0, 0].float().cpu().numpy()) * im["njy"]
            accs[variant].add(hst_sources[i], sr_njy, saturation_log * im["njy"])
    return {v: (rows[v], accs[v].result()) for v in variants}


def variant_labels(ensemble: int) -> list:
    """The row suffixes one checkpoint gets: "" for the single pass and, with ensemble K > 1,
    ":medianK" and ":meanK" for its stacks."""
    return [""] + ([f":{kind}{ensemble}" for kind in STACKS] if ensemble > 1 else [])


def summarize(label, step, rows, fine):
    out = {"checkpoint": label, "step": step, "n_images": len(rows)}
    for key in LOSSES:
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        out[key] = float(v.mean()) if v.size else np.nan
        out[f"{key}_se"] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan
    out.update({key: fine[key] for key in FINE})
    for key in FINE_CI:
        out[f"{key}_lo"], out[f"{key}_hi"] = fine[f"{key}_lo"], fine[f"{key}_hi"]
    out["fine_n_sources"] = fine["n_sources"]
    out["fine_n_saturated"] = fine["n_saturated"]
    out["fine_n_blown"] = fine["n_blown"]
    out["fine_n_star"] = fine["n_excluded"]
    return out


def read_rows(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_columns(path: Path, columns) -> None:
    """Refuse a CSV whose header is not `columns` (written by another version of the sweep)."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        header = next(csv.reader(f), None)
    if header is not None and header != list(columns):
        raise SystemExit(
            f"{path} has other columns (written by another version of the sweep): use another --out"
        )


def append_rows(path: Path, rows: list) -> None:
    if not rows:
        return
    new = not path.exists()
    check_columns(path, list(rows[0]))  # a backstop: main() checks sweep.csv before any work
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def variant_of(checkpoint: str) -> str:
    """ "" for a single-pass row, else the stack suffix ("median8") of an ensemble row."""
    return checkpoint.partition(":")[2]


def sweep_figure(rows, title):
    """Each metric against step: numbered checkpoints as a line with error bars (standard errors
    for the losses, the 68% bootstrap interval for the fine-structure summaries), ensemble stacks
    as dashed (median) and dotted (mean) lines, averages as stars at the end of their step range.
    The saturated_fraction panel also shows blown_fraction."""
    panels = [*LOSSES, "k50", "T_band", "saturated_fraction"]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), layout="constrained")
    variants = sorted({variant_of(r["checkpoint"]) for r in rows}, key=lambda v: (v != "", v))
    averaged = [r for r in rows if r["checkpoint"].startswith("avg_")]

    def errors(series, key):
        if key in LOSSES:
            return [float(r.get(f"{key}_se") or 0) for r in series]
        if key in FINE_CI:
            lo = [float(r[key]) - float(r.get(f"{key}_lo") or r[key]) for r in series]
            hi = [float(r.get(f"{key}_hi") or r[key]) - float(r[key]) for r in series]
            return [np.nan_to_num(lo), np.nan_to_num(hi)]
        return None

    def draw(ax, series, key, fmt, label):
        x = [int(r["step"]) / 1000 for r in series]
        y = [float(r[key]) for r in series]
        ax.errorbar(x, y, yerr=errors(series, key), fmt=fmt, ms=3, lw=1, capsize=2, label=label)

    for ax, key in zip(axes.flat, panels, strict=True):
        for variant in variants:
            series = [
                r
                for r in rows
                if PLAIN.match(r["checkpoint"].partition(":")[0])
                and variant_of(r["checkpoint"]) == variant
            ]
            if not series:
                continue
            fmt = "o-" if not variant else ("s--" if variant.startswith("median") else "^:")
            draw(ax, series, key, fmt, variant or "checkpoint")
            # an older sweep.csv (replotted on its own) has no blown_fraction column
            has_blown = all("blown_fraction" in r for r in series)
            if key == "saturated_fraction" and not variant and has_blown:
                draw(ax, series, "blown_fraction", "x-", "blown_fraction")
        for r in averaged:
            ax.plot(int(r["step"]) / 1000, float(r[key]), "*", ms=12, label=r["checkpoint"])
        ax.set_title(key)
        ax.set_xlabel("step (thousands)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    axes.flat[-1].legend(fontsize=8)
    fig.suptitle(title)
    return fig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="the run's training .ini")
    parser.add_argument("--ckpt-dir", help="default: the config's [CHECKPOINT] ckpt_dir")
    parser.add_argument("--out", required=True, help="output directory (reruns resume here)")
    parser.add_argument("--every", type=int, default=0, help="only steps divisible by this")
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument(
        "--average",
        action="append",
        default=[],
        metavar="A:B",
        help="also score the average of the step files with A <= step <= B (repeatable)",
    )
    parser.add_argument("--limit", type=int, help="first N val images only (by name)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="generator mode: train = the paper's deployment mode (default)",
    )
    parser.add_argument(
        "--ensemble",
        type=int,
        default=1,
        metavar="K",
        help="passes per image; K > 1 also scores the median and mean stacks (module docstring)",
    )
    parser.add_argument(
        "--keep-stars",
        action="store_true",
        help="keep HST sources inside the Gaia star masks in the fine-structure metric (they are "
        "still counted per image)",
    )
    parser.add_argument(
        "--star-radius",
        type=float,
        default=stars.MIN_RADIUS_ARCSEC,
        metavar="ARCSEC",
        help=f"minimum star mask radius (default {stars.MIN_RADIUS_ARCSEC:g})",
    )
    parser.add_argument("--hst-epoch", type=float, default=stars.HST_EPOCH)
    parser.add_argument("--lsst-epoch", type=float, default=stars.LSST_EPOCH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-comet", action="store_true")
    return parser.parse_args(argv)


def star_options(args):
    """neo.eval.stars.star_tracks keywords from the CLI (the masks are built and the stars
    counted with or without --keep-stars, which only decides whether the metric uses the
    sources inside them)."""
    return {
        "hst_epoch": args.hst_epoch,
        "lsst_epoch": args.lsst_epoch,
        "radius_arcsec": args.star_radius,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.ensemble < 1:
        raise SystemExit("--ensemble must be at least 1")
    if args.ensemble > 1 and args.mode == "eval":
        raise SystemExit(
            f"--ensemble {args.ensemble} needs --mode train: in eval mode every pass is the same "
            "image, so the stacks would repeat the single row"
        )
    config = configparser.ConfigParser()
    config.read(args.config)
    # before anything is loaded or written: the Gaia catalog must be there (the Lux compute
    # nodes have no network to fetch it), and an existing sweep.csv must be this version's
    options = star_options(args)
    try:
        stars.load_catalog()
    except FileNotFoundError as exc:
        if not args.keep_stars:
            raise SystemExit(f"{exc} (or pass --keep-stars)") from None
        print(f"{exc}: Gaia stars not checked (--keep-stars)")
        options = None
    out = Path(args.out)
    check_columns(out / "sweep.csv", COLUMNS)
    device = pick_device(args.device)
    ckpt_dir = Path(os.path.expandvars(args.ckpt_dir or config["CHECKPOINT"]["ckpt_dir"]))
    out.mkdir(parents=True, exist_ok=True)

    files = {
        s: p
        for s, p in step_files(ckpt_dir).items()
        if s >= args.min_step
        and (args.max_step is None or s <= args.max_step)
        and (not args.every or s % args.every == 0)
    }
    jobs = [(path.stem, step, [path]) for step, path in files.items()]
    for spec in args.average:
        lo, hi = (int(v) for v in spec.split(":"))
        members = [p for s, p in step_files(ckpt_dir).items() if lo <= s <= hi]
        if len(members) < 2:
            raise SystemExit(f"--average {spec}: only {len(members)} step files in {ckpt_dir}")
        jobs.append((f"avg_{lo}-{hi}", hi, members))

    images = load_images(config, args.limit, options)
    settings = {
        "config": str(Path(args.config).resolve()),
        "ckpt_dir": str(ckpt_dir),
        "seed": args.seed,
        "mode": args.mode,
        "images": [im["name"] for im in images],
    }
    # only non-default choices are recorded, so settings.json of an older sweep still matches
    if args.ensemble > 1:
        settings["ensemble"] = args.ensemble
    if args.keep_stars:
        settings["keep_stars"] = True
    if options is not None and options != star_options(parse_args(["--config", "x", "--out", "y"])):
        settings["stars"] = options
    settings_path = out / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise SystemExit(f"{out} holds a sweep with other settings: use another --out")
    settings_path.write_text(json.dumps(settings, indent=1))
    # a checkpoint is done when every row it gets (variant_labels) is there: a run killed between
    # a checkpoint's single-pass row and its stack rows is completed, not skipped
    variants = variant_labels(args.ensemble)
    present = {r["checkpoint"] for r in read_rows(out / "sweep.csv")}
    done = {
        c for c in {p.partition(":")[0] for p in present} if all(c + v in present for v in variants)
    }
    jobs = [j for j in jobs if j[0] not in done]
    print(f"{len(images)} val images, {len(jobs)} checkpoints to score ({len(done)} done) -> {out}")

    spectra = fine_structure.Spectra()
    hst_sources = [
        fine_structure.HstSources(
            im["hst_njy"], spectra, exclude=None if args.keep_stars else im["star_mask"]
        )
        for im in images
    ]
    n_star = sum(h.n_excluded for h in hst_sources)
    with_stars = sum(1 for im in images if im["n_stars"])
    if options is not None:
        left_out = (
            "none left out of the fine-structure metric (--keep-stars)"
            if args.keep_stars
            else f"{n_star} HST sources inside their masks left out of the fine-structure metric"
        )
        n_stars = sum(im["n_stars"] for im in images)
        print(f"Gaia stars: {n_stars} in {with_stars} images; {left_out}")
    gen = make_generator(config).to(device)
    disc = PatchGAN(2).to(device)
    weights = eval(config["VGG_LOSS_WEIGHTS"]["vgg_loss_weights"])
    vgg = VGGLoss(device, weights=weights)
    # the run's cached template (or [NOISE] template) only: never measure one from local pairs
    noise = None
    if NoiseModel.enabled(config):
        noise = NoiseModel.from_config(config, device, measure=False)
    identifier = config.get("IDENTIFIER", "identifier", fallback="").strip('"')

    experiment = None
    if not args.no_comet and jobs:
        experiment = start_experiment(f"sweep {identifier}", ["sweep", identifier])
        experiment.log_parameters({k: v for k, v in settings.items() if k != "images"})
        experiment.log_parameter("n_images", len(images))
        experiment.log_parameter("ensemble", args.ensemble)

    fine_out = out / "fine_structure.json"
    fine_all = json.loads(fine_out.read_text()) if fine_out.exists() else {}
    for label, step, paths in jobs:
        t0 = time.time()
        state = average_states(paths, config) if len(paths) > 1 else load_state(paths[0], config)
        gen.load_state_dict(state["gen"])
        disc.load_state_dict(state["patch_gan"])
        results = evaluate(
            gen,
            disc,
            vgg,
            images,
            hst_sources,
            spectra,
            device,
            args.seed,
            args.mode,
            noise,
            args.ensemble,
        )
        summaries, per_image = [], []
        for variant, (rows, fine) in results.items():
            summary = summarize(label + variant, step, rows, fine)
            # a row already there (this checkpoint's rerun after a kill) has the same numbers:
            # the seeds fix every draw, so it is not written twice
            if label + variant not in present:
                summaries.append(summary)
                per_image += [{"checkpoint": label + variant, **r} for r in rows]
                fine_all[label + variant] = fine
            interval = f"[{summary['T_band_lo']:.3f}, {summary['T_band_hi']:.3f}]"
            print(
                f"  {label + variant}: L1 {summary['l1']:.5f} +- {summary['l1_se']:.5f}, segmap "
                f"{summary['segmap_l1']:.4f}, VGG {summary['vgg']:.3f}, k50 {summary['k50']:.2f}, "
                f"T {summary['T_band']:.3f} {interval}, R {summary['R_band']:.3f}, blown "
                f"{summary['fine_n_blown']}"
                + (f" ({time.time() - t0:.0f} s)" if not variant else "")
            )
            if experiment is not None:
                prefix = ("avg " if label.startswith("avg_") else "") + (
                    f"{variant[1:]} " if variant else ""
                )
                experiment.log_metrics(
                    {f"{prefix}{k}": summary[k] for k in (*LOSSES, *FINE)}, step=step
                )
        # one write per file per checkpoint, so its rows land together
        append_rows(out / "sweep.csv", summaries)
        append_rows(out / "per_image.csv", per_image)
        fine_out.write_text(json.dumps(fine_all))

    all_rows = read_rows(out / "sweep.csv")
    if all_rows:
        fig = sweep_figure(all_rows, f"{identifier}: fixed-set sweep ({len(images)} val images)")
        fig.savefig(out / "sweep.png", dpi=110)
        if experiment is not None:
            experiment.log_figure(figure_name="sweep", figure=fig)
            experiment.log_table(str(out / "sweep.csv"))
        plt.close(fig)
    if experiment is not None:
        experiment.end()


if __name__ == "__main__":
    main()
