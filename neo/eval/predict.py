"""Run a trained model over a pairs split and write super-resolved images in physical units.

Inputs go through the training dataset class unchanged (center crop -> clip -> log scale -> pad,
no augmentation), so every model sees exactly what it was trained on. Outputs are cropped to the
central 600 px and inverse log-scaled (neo.eval.postprocess.to_physical) into the HR pairs'
stored units, converted to nJy per HR pixel with the HR cutout's NJYPERPX (1.0 for nJy pairs),
then written as <out>/<pair name>.fits (BUNIT nJy) with the HR cutout's WCS. predictions.csv
records the log-space L1 against the HR target over the same 600 px region.

--gen-mode train runs the GAN generator as the paper generated its SR images: the whole pickled
generator was loaded and never put in eval mode, so dropout stayed on and batch norm used each
image's own statistics, at batch size 1 (the paper's neo/analysis/generate_sr_images.py loads the
pickled generator, never calls .eval() and uses batch_size=1). The default (eval) is deterministic.

--subset select|report|all (neo.eval.subsets) picks which val pairs to predict. Score candidate
checkpoints on select only (their L1 here, or compare.py --subset select); predict report only
for the chosen checkpoint, so the reported numbers never influence the choice.

Each output records its checkpoint (NEORUN, NEOCKPT, NEOSTEP), generator mode (NEOGMODE), units
(BUNIT) and sky (PAIRID). An existing file is reused only when all of those match; anything else
stops the run unless --overwrite is given, so predictions from another checkpoint, mode or pairs
build cannot be mixed in. The run is tracked on Comet (neo.eval.tracking): parameters, the
config, per-pair L1, example images, predictions.csv.

--samples K (K > 1, stochastic models only: diffusion, or a GAN with --gen-mode train) draws K
samples per pair and writes <out>/single, <out>/mean, <out>/median (each a predictions directory
for compare.py, with its own predictions.csv) and <out>/std (per-pixel spread of the draws, nJy);
see neo.eval.stack. Their files also record NEOSTACK (which stack) and NEOSAMPL (K), and the
example figures compare one draw with the stacks.
"""

import argparse
import configparser
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from neo.data.dataset import SR_HST_HSC_Dataset  # noqa: E402
from neo.eval import stack  # noqa: E402
from neo.eval.postprocess import (  # noqa: E402
    HR_SIZE,
    LR_SIZE,
    center_crop,
    njy_per_px,
    to_physical,
)
from neo.eval.predictors import build_predictor  # noqa: E402
from neo.eval.subsets import SUBSETS, in_subset, load_groups, pair_id  # noqa: E402
from neo.eval.tracking import start_experiment  # noqa: E402
from neo.models.registry import generator_name  # noqa: E402

N_EXAMPLES = 3


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cropped_wcs_header(hr_path: Path, size: int) -> fits.Header:
    header = fits.getheader(hr_path)
    ny, nx = header["NAXIS2"], header["NAXIS1"]
    y0, x0 = (ny - size) // 2, (nx - size) // 2
    return WCS(header)[y0 : y0 + size, x0 : x0 + size].to_header()


def example_figure(name, lr, sr, hr):
    """LR | SR | HST in the training log space, on the HST image's color scale."""
    panels = [center_crop(lr, LR_SIZE), center_crop(sr, HR_SIZE), center_crop(hr, HR_SIZE)]
    vmin, vmax = np.percentile(panels[2], [1, 99.8])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, img, title in zip(axes, panels, ["LR", "SR", "HST"], strict=True):
        im = ax.imshow(img, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="log-scaled flux")
    fig.suptitle(name)
    return fig


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="the .ini the model was trained with")
    parser.add_argument("--checkpoint", required=True, help="latest.pt or step_*.pt")
    parser.add_argument("--split-dir", required=True, help="pairs split with lr/ and hr/")
    parser.add_argument("--out", required=True, help="directory for <name>.fits predictions")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, help="first N pairs only (sorted by name)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0, help="for stochastic models")
    parser.add_argument(
        "--gen-mode",
        choices=["eval", "train"],
        default="eval",
        help="GAN generator mode at inference (train: dropout and per-image BatchNorm, as the "
        "paper generated its SR images; needs --batch-size 1)",
    )
    parser.add_argument(
        "--subset", choices=SUBSETS, default="all", help="val pairs to predict (module docstring)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="draws per pair; K > 1 writes single/mean/median/std stacks (module docstring)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing predictions")
    parser.add_argument("--tag", action="append", default=[], help="extra Comet tag (repeatable)")
    parser.add_argument("--no-comet", action="store_true", help="do not track the run on Comet")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.gen_mode == "train" and args.batch_size != 1:
        raise SystemExit("--gen-mode train uses batch statistics: use --batch-size 1")
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    config = configparser.ConfigParser()
    config.read(args.config)

    split = Path(args.split_dir)
    dataset = SR_HST_HSC_Dataset(
        hst_path=str(split / "hr"),
        hsc_path=str(split / "lr"),
        hr_size=[HR_SIZE, HR_SIZE],
        lr_size=[100, 100],
        transform_type="ds9_scale",
        data_aug=False,
        experiment=None,
    )
    model = generator_name(config)
    if args.samples > 1 and model != "diffusion" and args.gen_mode == "eval":
        raise SystemExit(
            f"--samples {args.samples}: the {model} generator in eval mode is deterministic, so "
            "every draw would be the same (use --gen-mode train for dropout draws)"
        )
    groups = load_groups(split)
    order = sorted(range(len(dataset)), key=lambda i: dataset.filenames[i])
    order = [i for i in order if in_subset(dataset.filenames[i], args.subset, groups)][: args.limit]
    out = Path(args.out)
    kinds = stack.KINDS if args.samples > 1 else (None,)  # None: one draw, written to out itself
    dirs = {kind: out / kind if kind else out for kind in kinds}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    def stack_cards(kind):
        if kind is None:
            return {}
        return {"NEOSTACK": kind, "NEOSAMPL": args.samples}

    predict = build_predictor(config, args.checkpoint, device, mode=args.gen_mode)
    print(f"device {device} | {len(order)} pairs from {split} -> {out}")

    identifier = config.get("IDENTIFIER", "identifier", fallback="").strip('"')
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    step = int(state.get("step", -1))
    del state
    source = {
        "NEORUN": checkpoint.parent.name,
        "NEOCKPT": checkpoint.name,
        "NEOSTEP": step,
        "NEOGMODE": args.gen_mode,
    }
    experiment = None
    if not args.no_comet:
        experiment = start_experiment(
            f"predict {identifier or model} {checkpoint.stem} {split.name}-{args.subset}",
            ["predict", model, args.subset, *args.tag],
        )
        experiment.log_parameters(
            {
                "generator": model,
                "identifier": identifier,
                "config": args.config,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_step": step,
                "split_dir": str(split.resolve()),
                "subset": args.subset,
                "out": str(out.resolve()),
                "n_pairs": len(order),
                "batch_size": args.batch_size,
                "seed": args.seed,
                "gen_mode": args.gen_mode,
                "samples": args.samples,
                "device": device,
            }
        )
        experiment.log_asset(args.config)

    scored = [k for k in kinds if k != "std"]
    rows, n_done, t0 = {k: [] for k in scored}, 0, time.time()
    for start in range(0, len(order), args.batch_size):
        batch = []
        for i in order[start : start + args.batch_size]:
            name = dataset.filenames[i]
            hr_header = fits.getheader(split / "hr" / name)
            existing = [k for k in kinds if (dirs[k] / name).exists()]
            if existing and not args.overwrite:
                for kind in existing:
                    expected = {**source, "BUNIT": "nJy", "PAIRID": pair_id(hr_header)}
                    expected.update(stack_cards(kind))
                    header = fits.getheader(dirs[kind] / name)
                    differs = {k: header.get(k) for k, v in expected.items() if header.get(k) != v}
                    if differs:
                        raise SystemExit(
                            f"{dirs[kind] / name} exists but came from {differs}, not {expected}: "
                            "predict into a new --out (or pass --overwrite)"
                        )
                if len(existing) == len(kinds):
                    continue
            hst, hsc, hsc_hr, _ = dataset[i]
            if hst is None:
                print(f"  skipped {name}: dataset returned no sample")
                continue
            batch.append((name, hst, hsc, hsc_hr, hr_header))
        if not batch:
            continue
        lr = torch.stack([b[2] for b in batch]).unsqueeze(1)
        cond = torch.stack([b[3] for b in batch]).unsqueeze(1).to(device)
        draws = torch.stack([predict(lr, cond).detach().float().cpu() for _ in range(args.samples)])
        images = stack.stack(draws) if args.samples > 1 else {None: draws[0]}
        for j, (name, hst, hsc, _, hr_header) in enumerate(batch):
            target = hst.numpy()
            base = cropped_wcs_header(split / "hr" / name, HR_SIZE)
            base["BUNIT"] = "nJy"
            njy = njy_per_px(hr_header)
            base["HRNJYPX"] = (njy, "NJYPERPX of the HR cutout, applied to get nJy")
            base.update(source)
            base["PAIRID"] = pair_id(hr_header)
            for kind in kinds:
                p = images[kind][j, 0].numpy()
                header = base.copy()
                header.update(stack_cards(kind))
                if kind == "std":  # already linear, in the HR pairs' stored units
                    sr = center_crop(p.astype(np.float64), HR_SIZE) * njy
                else:
                    l1 = stack.l1(p, target)
                    header["L1LOG"] = l1
                    rows[kind].append({"name": name, "l1_log": l1})
                    sr = to_physical(p) * njy
                fits.PrimaryHDU(sr.astype(np.float32), header=header).writeto(
                    dirs[kind] / name, overwrite=True
                )
            n_done += 1
            if experiment is not None:
                experiment.log_metrics(
                    {"l1_log" + (f"_{k}" if k else ""): rows[k][-1]["l1_log"] for k in scored},
                    step=n_done,
                )
                if n_done <= N_EXAMPLES:
                    if args.samples > 1:
                        fig = stack.stack_figure(
                            name,
                            hsc.numpy(),
                            target,
                            {k: images[k][j, 0].numpy() for k in stack.KINDS},
                            args.samples,
                        )
                    else:
                        fig = example_figure(name, hsc.numpy(), images[None][j, 0].numpy(), target)
                    experiment.log_figure(figure_name=f"example {n_done}: {name}", figure=fig)
                    plt.close(fig)
        done = min(start + args.batch_size, len(order))
        print(f"  {done}/{len(order)} ({(time.time() - t0) / done:.2f} s/pair)")

    if n_done:
        summary = {"n_written": n_done, "s_per_pair": (time.time() - t0) / n_done}
        for kind in scored:
            path = dirs[kind] / "predictions.csv"
            new = not path.exists()
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["name", "l1_log"])
                if new:
                    writer.writeheader()
                writer.writerows(rows[kind])
            mean_l1 = np.mean([r["l1_log"] for r in rows[kind]])
            label = f" ({kind} of {args.samples} draws)" if kind else ""
            print(f"wrote {n_done} predictions{label}; mean log-space L1 {mean_l1:.4f}")
            summary["mean_l1_log" + (f"_{kind}" if kind else "")] = mean_l1
            if experiment is not None:
                experiment.log_table(str(path))
        if experiment is not None:
            experiment.log_metrics(summary)
    if experiment is not None:
        experiment.end()


if __name__ == "__main__":
    main()
