"""Run a trained model over a pairs split and write super-resolved images in physical units.

Inputs go through the training dataset class unchanged (center crop -> clip -> log scale -> pad,
no augmentation), so every model sees exactly what it was trained on. Outputs are cropped to the
central 600 px and inverse log-scaled (neo.eval.postprocess.to_physical), then written as
<out>/<pair name>.fits with the HR cutout's WCS. predictions.csv records the log-space L1 against
the HR target over the same 600 px region (a cheap checkpoint-selection proxy).
"""

import argparse
import configparser
import csv
import time
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS

from neo.data.dataset import SR_HST_HSC_Dataset
from neo.eval.postprocess import HR_SIZE, center_crop, to_physical
from neo.eval.predictors import build_predictor


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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
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
    order = sorted(range(len(dataset)), key=lambda i: dataset.filenames[i])[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predict = build_predictor(config, args.checkpoint, device, mode=args.gen_mode)
    print(f"device {device} | {len(order)} pairs from {split} -> {out}")

    rows, t0 = [], time.time()
    for start in range(0, len(order), args.batch_size):
        batch = []
        for i in order[start : start + args.batch_size]:
            name = dataset.filenames[i]
            if (out / name).exists() and not args.overwrite:
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
        for (name, hst, _, _), p in zip(batch, pred, strict=True):
            l1 = float(np.mean(np.abs(center_crop(p, HR_SIZE) - center_crop(hst.numpy(), HR_SIZE))))
            header = cropped_wcs_header(split / "hr" / name, HR_SIZE)
            header["BUNIT"] = fits.getheader(split / "hr" / name).get("BUNIT", "")
            header["NEOCKPT"] = Path(args.checkpoint).name
            header["L1LOG"] = l1
            fits.PrimaryHDU(to_physical(p).astype(np.float32), header=header).writeto(
                out / name, overwrite=True
            )
            rows.append({"name": name, "l1_log": l1})
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


if __name__ == "__main__":
    main()
