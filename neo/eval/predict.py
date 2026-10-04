"""Run a trained model over a pairs split and write super-resolved images in physical units.

Inputs go through the training dataset class unchanged (center crop -> clip -> log scale -> pad,
no augmentation), so every model sees exactly what it was trained on. Outputs are cropped to the
central 600 px and inverse log-scaled (neo.eval.postprocess.to_physical), then written as
<out>/<pair name>.fits with the HR cutout's WCS. predictions.csv records the log-space L1 against
the HR target over the same 600 px region.

--subset select|report|all (neo.eval.subsets) picks which val pairs to predict. Score candidate
checkpoints on select only (their L1 here, or compare.py --subset select); predict report only
for the chosen checkpoint, so the reported numbers never influence the choice.

Each output records its checkpoint (NEORUN, NEOCKPT, NEOSTEP) and sky (PAIRID). An existing file
is reused only when all of those match; anything else stops the run unless --overwrite is given,
so predictions from another checkpoint or pairs build cannot be mixed in. The run is tracked on
Comet (neo.eval.tracking): parameters, the config, per-pair L1, example images, predictions.csv.
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
from neo.eval.postprocess import HR_SIZE, LR_SIZE, center_crop, to_physical  # noqa: E402
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
        help="GAN generator mode at inference (train keeps dropout/batch-stat BatchNorm)",
    )
    parser.add_argument(
        "--subset", choices=SUBSETS, default="all", help="val pairs to predict (module docstring)"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing predictions")
    parser.add_argument("--tag", action="append", default=[], help="extra Comet tag (repeatable)")
    parser.add_argument("--no-comet", action="store_true", help="do not track the run on Comet")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.gen_mode == "train" and args.batch_size != 1:
        raise SystemExit("--gen-mode train uses batch statistics: use --batch-size 1")
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
    groups = load_groups(split)
    order = sorted(range(len(dataset)), key=lambda i: dataset.filenames[i])
    order = [i for i in order if in_subset(dataset.filenames[i], args.subset, groups)][: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predict = build_predictor(config, args.checkpoint, device, mode=args.gen_mode)
    print(f"device {device} | {len(order)} pairs from {split} -> {out}")

    model = generator_name(config)
    identifier = config.get("IDENTIFIER", "identifier", fallback="").strip('"')
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    step = int(state.get("step", -1))
    del state
    source = {"NEORUN": checkpoint.parent.name, "NEOCKPT": checkpoint.name, "NEOSTEP": step}
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
                "device": device,
            }
        )
        experiment.log_asset(args.config)

    rows, t0 = [], time.time()
    for start in range(0, len(order), args.batch_size):
        batch = []
        for i in order[start : start + args.batch_size]:
            name = dataset.filenames[i]
            expected = {**source, "PAIRID": pair_id(fits.getheader(split / "hr" / name))}
            if (out / name).exists() and not args.overwrite:
                header = fits.getheader(out / name)
                differs = {k: header.get(k) for k, v in expected.items() if header.get(k) != v}
                if differs:
                    raise SystemExit(
                        f"{out / name} exists but came from {differs}, not {expected}: predict "
                        "into a new --out (or pass --overwrite)"
                    )
                continue
            hst, hsc, hsc_hr, _ = dataset[i]
            if hst is None:
                print(f"  skipped {name}: dataset returned no sample")
                continue
            batch.append((name, hst, hsc, hsc_hr))
        if not batch:
            continue
        lr = torch.stack([b[2] for b in batch]).unsqueeze(1)
        cond = torch.stack([b[3] for b in batch]).unsqueeze(1).to(device)
        pred = predict(lr, cond).detach().float().cpu().numpy()[:, 0]
        for (name, hst, hsc, _), p in zip(batch, pred, strict=True):
            l1 = float(np.mean(np.abs(center_crop(p, HR_SIZE) - center_crop(hst.numpy(), HR_SIZE))))
            header = cropped_wcs_header(split / "hr" / name, HR_SIZE)
            header["BUNIT"] = fits.getheader(split / "hr" / name).get("BUNIT", "")
            header.update(source)
            header["PAIRID"] = pair_id(fits.getheader(split / "hr" / name))
            header["L1LOG"] = l1
            fits.PrimaryHDU(to_physical(p).astype(np.float32), header=header).writeto(
                out / name, overwrite=True
            )
            rows.append({"name": name, "l1_log": l1})
            if experiment is not None:
                experiment.log_metric("l1_log", l1, step=len(rows))
                if len(rows) <= N_EXAMPLES:
                    fig = example_figure(name, hsc.numpy(), p, hst.numpy())
                    experiment.log_figure(figure_name=f"example {len(rows)}: {name}", figure=fig)
                    plt.close(fig)
        done = min(start + args.batch_size, len(order))
        print(f"  {done}/{len(order)} ({(time.time() - t0) / done:.2f} s/pair)")

    if rows:
        path = out / "predictions.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "l1_log"])
            if new:
                writer.writeheader()
            writer.writerows(rows)
        mean_l1 = np.mean([r["l1_log"] for r in rows])
        print(f"wrote {len(rows)} predictions; mean log-space L1 {mean_l1:.4f}")
        if experiment is not None:
            experiment.log_metrics(
                {
                    "mean_l1_log": mean_l1,
                    "n_written": len(rows),
                    "s_per_pair": (time.time() - t0) / len(rows),
                }
            )
            experiment.log_table(str(path))
    if experiment is not None:
        experiment.end()


if __name__ == "__main__":
    main()
