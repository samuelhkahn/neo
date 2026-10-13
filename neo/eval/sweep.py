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
PatchGAN), averaged over images with standard errors, and the fine-structure metric
(neo.eval.fine_structure: transfer T, coherence rho and power ratio R inside HST sources, k50) on
the images in nJy. --average A:B also scores the uniform average of the step files from A to B, a
free stand-in for an EMA of the weights or a decaying learning rate.

Writes, in --out: sweep.csv (one row per checkpoint), per_image.csv (for paired comparisons),
fine_structure.json (T, rho, R per k), sweep.png, and settings.json. Checkpoints already in
sweep.csv with the same settings are skipped, so rerunning picks up only new checkpoints. Tracked
on Comet. Training is not touched: it reads checkpoints only.

Usage:  python -m neo.eval.sweep --config neo/configs/lux.ini --out $NEO_DATA/sweeps/lux \
            [--average 650000:850000] [--every 50000] [--limit 50]
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
from neo.eval import fine_structure  # noqa: E402
from neo.eval.postprocess import HR_SIZE, center_crop, njy_per_px, to_physical  # noqa: E402
from neo.eval.predict import pick_device  # noqa: E402
from neo.eval.tracking import start_experiment  # noqa: E402
from neo.models.patchgan import PatchGAN  # noqa: E402
from neo.models.registry import check_generator, make_generator  # noqa: E402
from neo.models.vgg19_loss import VGGLoss  # noqa: E402

STEP_FILE = re.compile(r"^step_(\d+)\.pt$")
LOSSES = ("l1", "segmap_l1", "vgg", "adversarial", "discriminator")
FINE = ("k50", "T_band", "rho_band", "R_band")


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


def load_images(config, limit=None):
    """The run's validation images through its own val dataset class (hsc_hr_8bit included), in
    filename order, plus each HST cutout's central 600 px in nJy for the fine-structure metric."""
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
        images.append(
            {
                "name": name,
                "hst": hst,
                "lr": lr,
                "hsc_hr": hsc_hr,
                "seg": seg,
                "njy": njy,
                "hst_njy": hst_njy,
            }
        )
    return images


@torch.no_grad()
def evaluate(gen, disc, vgg, images, hst_sources, spectra, device, seed=0, mode="train"):
    """Per-image losses and the pooled fine-structure result for one set of weights."""
    gen.train(mode == "train")
    disc.train(mode == "train")
    bce = nn.BCEWithLogitsLoss()
    crop = CenterCrop(HR_SIZE)
    acc = fine_structure.Accumulator(spectra)
    rows = []
    for i, im in enumerate(images):
        torch.manual_seed(seed + i)  # the same dropout masks for this image in every checkpoint
        fake = gen(im["lr"][None, None].to(device))
        f = crop(fake)
        r = crop(im["hst"][None, None].to(device))
        s = crop(im["seg"][None, None].to(device))
        c = crop(im["hsc_hr"][None, None].to(device))
        err = (f - r).abs()
        fake_logits, real_logits = disc(f, c).flatten(), disc(r, c).flatten()
        rows.append(
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
        sr_njy = to_physical(fake[0, 0].float().cpu().numpy()) * im["njy"]
        acc.add(hst_sources[i], sr_njy)
    return rows, acc.result()


def summarize(label, step, rows, fine):
    out = {"checkpoint": label, "step": step, "n_images": len(rows)}
    for key in LOSSES:
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        out[key] = float(v.mean()) if v.size else np.nan
        out[f"{key}_se"] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan
    out.update({key: fine[key] for key in FINE})
    out["fine_n_sources"] = fine["n_sources"]
    return out


def read_rows(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_rows(path: Path, rows: list) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def sweep_figure(rows, title):
    """Each metric against step: numbered checkpoints as a line with standard errors, averages as
    stars at the end of their step range."""
    panels = [*LOSSES, "k50", "T_band", "R_band"]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), layout="constrained")
    plain = [r for r in rows if r["checkpoint"].startswith("step_")]
    averaged = [r for r in rows if r["checkpoint"].startswith("avg_")]
    for ax, key in zip(axes.flat, panels, strict=True):
        if plain:
            x = [int(r["step"]) / 1000 for r in plain]
            y = [float(r[key]) for r in plain]
            se = [float(r.get(f"{key}_se") or 0) for r in plain] if key in LOSSES else None
            ax.errorbar(x, y, yerr=se, fmt="o-", ms=3, lw=1, capsize=2, label="checkpoint")
        for r in averaged:
            ax.plot(int(r["step"]) / 1000, float(r[key]), "*", ms=12, label=r["checkpoint"])
        ax.set_title(key)
        ax.set_xlabel("step (thousands)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-comet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = configparser.ConfigParser()
    config.read(args.config)
    device = pick_device(args.device)
    ckpt_dir = Path(os.path.expandvars(args.ckpt_dir or config["CHECKPOINT"]["ckpt_dir"]))
    out = Path(args.out)
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

    images = load_images(config, args.limit)
    settings = {
        "config": str(Path(args.config).resolve()),
        "ckpt_dir": str(ckpt_dir),
        "seed": args.seed,
        "mode": args.mode,
        "images": [im["name"] for im in images],
    }
    settings_path = out / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise SystemExit(f"{out} holds a sweep with other settings: use another --out")
    settings_path.write_text(json.dumps(settings, indent=1))
    done = {r["checkpoint"] for r in read_rows(out / "sweep.csv")}
    jobs = [j for j in jobs if j[0] not in done]
    print(f"{len(images)} val images, {len(jobs)} checkpoints to score ({len(done)} done) -> {out}")

    spectra = fine_structure.Spectra()
    hst_sources = [fine_structure.HstSources(im["hst_njy"], spectra) for im in images]
    gen = make_generator(config).to(device)
    disc = PatchGAN(2).to(device)
    weights = eval(config["VGG_LOSS_WEIGHTS"]["vgg_loss_weights"])
    vgg = VGGLoss(device, weights=weights)
    identifier = config.get("IDENTIFIER", "identifier", fallback="").strip('"')

    experiment = None
    if not args.no_comet and jobs:
        experiment = start_experiment(f"sweep {identifier}", ["sweep", identifier])
        experiment.log_parameters({k: v for k, v in settings.items() if k != "images"})
        experiment.log_parameter("n_images", len(images))

    fine_out = out / "fine_structure.json"
    fine_all = json.loads(fine_out.read_text()) if fine_out.exists() else {}
    for label, step, paths in jobs:
        t0 = time.time()
        state = average_states(paths, config) if len(paths) > 1 else load_state(paths[0], config)
        gen.load_state_dict(state["gen"])
        disc.load_state_dict(state["patch_gan"])
        rows, fine = evaluate(
            gen, disc, vgg, images, hst_sources, spectra, device, args.seed, args.mode
        )
        summary = summarize(label, step, rows, fine)
        append_rows(out / "sweep.csv", [summary])
        append_rows(out / "per_image.csv", [{"checkpoint": label, **r} for r in rows])
        fine_all[label] = fine
        fine_out.write_text(json.dumps(fine_all))
        print(
            f"  {label}: L1 {summary['l1']:.5f} +- {summary['l1_se']:.5f}, segmap "
            f"{summary['segmap_l1']:.4f}, VGG {summary['vgg']:.3f}, k50 {summary['k50']:.2f}, "
            f"T {summary['T_band']:.3f}, R {summary['R_band']:.3f} ({time.time() - t0:.0f} s)"
        )
        if experiment is not None:
            prefix = "avg " if label.startswith("avg_") else ""
            experiment.log_metrics(
                {f"{prefix}{k}": summary[k] for k in (*LOSSES, *FINE)}, step=step
            )

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
