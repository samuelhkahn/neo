"""Training script for the diffusion super-resolution experiment (EDM, conditional on the LR image).

Same data, preprocessing, validation logging and checkpointing as train.py; only the model and its
objective differ. The dataset class is used unchanged (center crop -> clip -> ds9 log scale ->
pad), and the model learns p(HR | 6x-upsampled LR) in that log space (see neo/diffusion.py).

Usage:
    python train_diffusion.py <config_file> [--resume]

Example:
    python train_diffusion.py neo/configs/lux_diffusion.ini --resume
"""

import argparse
import configparser
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from comet_ml import Experiment, OfflineExperiment
from torchvision.transforms import CenterCrop
from tqdm import tqdm

from neo import diffusion
from neo.data.augment import dataset_classes
from neo.data.collate_fn import collate_fn
from neo.data.dataset import SR_HST_HSC_Dataset
from neo.eval import stack
from neo.log_figure import log_figure


def save_checkpoint(path, model, ema, opt, step, model_name):
    """Write model + EMA + optimizer state atomically; a killed job never leaves a corrupt file."""
    state = {
        "step": step,
        "model_name": model_name,
        "model": model.state_dict(),
        "ema": ema.model.state_dict(),
        "opt": opt.state_dict(),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, ema, opt, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    ema.model.load_state_dict(state["ema"])
    opt.load_state_dict(state["opt"])
    return state["step"]


@torch.no_grad()
def measure_stats(dataloader, n_batches):
    """Log-space mean/std of the HR targets and conditioning images, and the HR range, over the
    central 600 px of n_batches training batches."""
    sums = torch.zeros(2, 3, dtype=torch.float64)
    hr_min, hr_max = float("inf"), float("-inf")
    for i, (hr, _, hsc_hr, _) in enumerate(dataloader):
        if i == n_batches:
            break
        for row, images in enumerate([hr, hsc_hr]):
            v = CenterCrop(600)(images).double()
            sums[row] += torch.tensor([v.numel(), v.sum(), (v**2).sum()])
        hr_min = min(hr_min, CenterCrop(600)(hr).min().item())
        hr_max = max(hr_max, CenterCrop(600)(hr).max().item())
    n, s, s2 = sums.unbind(1)
    mean = s / n
    std = (s2 / n - mean**2).sqrt()
    return {
        "hr_mean": mean[0].item(),
        "hr_std": std[0].item(),
        "cond_mean": mean[1].item(),
        "cond_std": std[1].item(),
        "hr_min": hr_min,
        "hr_max": hr_max,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the diffusion SR model from an .ini config")
    parser.add_argument("config", help="training .ini file")
    parser.add_argument(
        "--resume", action="store_true", help="continue from <ckpt_dir>/latest.pt if it exists"
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")

    config = configparser.ConfigParser()
    config.read(args.config)
    dcfg = config["DIFFUSION"]

    # Data paths; environment variables such as ${NEO_DATA} are expanded
    hst_path_train = os.path.expandvars(config["DEFAULT"]["hst_path_train"])
    hsc_path_train = os.path.expandvars(config["DEFAULT"]["hsc_path_train"])
    hst_path_val = os.path.expandvars(config["DEFAULT"]["hst_path_val"])
    hsc_path_val = os.path.expandvars(config["DEFAULT"]["hsc_path_val"])

    hst_dim = int(config["HST_DIM"]["hst_dim"])
    hsc_dim = int(config["HSC_DIM"]["hsc_dim"])

    comet_tag = config["COMET_TAG"]["comet_tag"]
    comet_project = config.get("COMET_PROJECT", "comet_project", fallback="neo-rubin-lsst")
    batch_size = int(config["BATCH_SIZE"]["batch_size"])
    save_steps = int(config["SAVE_STEPS"]["save_steps"])
    ckpt_dir = Path(os.path.expandvars(config.get("CHECKPOINT", "ckpt_dir", fallback="models")))
    latest_every = config.getint("CHECKPOINT", "latest_every", fallback=save_steps)
    num_workers = config.getint("DATALOADER", "num_workers", fallback=0)
    data_aug = eval(config["DATA_AUG"]["data_aug"])
    identifier = eval(config["IDENTIFIER"]["identifier"])
    display_step = eval(config["DISPLAY_STEPS"]["display_steps"])

    # Diffusion settings
    total_steps = dcfg.getint("train_steps")
    lr = dcfg.getfloat("lr", 1e-4)
    warmup_steps = dcfg.getint("warmup_steps", 5000)
    grad_clip = dcfg.getfloat("grad_clip", 1.0)
    ema_decay = dcfg.getfloat("ema_decay", 0.9999)
    crop_size = dcfg.getint("crop_size", 256)
    p_mean = dcfg.getfloat("p_mean", -1.2)
    p_std = dcfg.getfloat("p_std", 1.2)
    sample_steps = dcfg.getint("sample_steps", 18)
    sample_every = dcfg.getint("sample_every", 5000)
    stack_samples = dcfg.getint("stack_samples", 8)
    stats_batches = dcfg.getint("stats_batches", 50)

    # Comet ML experiment tracking; logs locally when no API key is set
    api_key = os.environ.get("COMET_ML_ASTRO_API_KEY")
    comet_kwargs = dict(project_name=comet_project, workspace="samkahn-astro")
    if api_key:
        experiment = Experiment(api_key=api_key, **comet_kwargs)
    else:
        print("COMET_ML_ASTRO_API_KEY not set; logging offline to ./comet_offline")
        experiment = OfflineExperiment(offline_directory="comet_offline", **comet_kwargs)

    experiment.add_tag(comet_tag)
    experiment.log_asset(args.config)
    experiment.log_parameter("generator", "diffusion")
    experiment.log_parameter("batch_size", batch_size)
    experiment.log_parameter("total_steps", total_steps)
    experiment.log_parameter("save_steps", save_steps)
    experiment.log_parameter("data_aug", data_aug)
    experiment.log_parameter("display_step", display_step)
    experiment.log_parameters(
        {f"diffusion_{k}": v for k, v in dcfg.items() if k not in config.defaults()}
    )

    model_name = f"edm_{identifier}_lr={lr}_crop={crop_size}_ema={ema_decay}"
    print(model_name)

    def make_loader(hst_path, hsc_path, dataset_class=SR_HST_HSC_Dataset):
        return torch.utils.data.DataLoader(
            dataset_class(
                hst_path=hst_path,
                hsc_path=hsc_path,
                hr_size=[hst_dim, hst_dim],
                lr_size=[hsc_dim, hsc_dim],
                transform_type="ds9_scale",
                data_aug=data_aug,
                experiment=None,
            ),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=device == "cuda",
        )

    # [DATA_AUG] augment applies to training only (val keeps the centre crop), as in train.py
    augment, TrainDataset, ValDataset = dataset_classes(config)
    experiment.log_parameter("augment", augment)
    dataloader_train = make_loader(hst_path_train, hsc_path_train, TrainDataset)
    dataloader_val = make_loader(hst_path_val, hsc_path_val, ValDataset)
    val_iter = iter(dataloader_val)

    def next_val_batch():
        nonlocal val_iter
        try:
            return next(val_iter)
        except StopIteration:
            val_iter = iter(dataloader_val)
            return next(val_iter)

    # Initialize model
    model = diffusion.build(dcfg).to(device)
    ema = diffusion.EMA(model, decay=ema_decay)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"generator: diffusion ({n_params / 1e6:.1f} M parameters)")
    experiment.log_parameter("generator_params", n_params)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cur_step = 0
    latest = ckpt_dir / "latest.pt"
    if args.resume and latest.exists():
        cur_step = load_checkpoint(latest, model, ema, opt, device)
        print(f"resumed from {latest} at step {cur_step}")
    else:
        stats = measure_stats(dataloader_train, stats_batches)
        model.set_stats(**stats)
        ema.model.set_stats(**stats)
        print("log-space stats: " + ", ".join(f"{k} {v:.4f}" for k, v in stats.items()))
    experiment.log_parameter("start_step", cur_step)
    experiment.log_parameters({k: v.item() for k, v in model.named_buffers() if v.ndim == 0})

    # Training loop
    model.train()
    while cur_step < total_steps:
        for hr_real, _, hsc_hr, _ in tqdm(dataloader_train, position=0):
            hr_crop, cond_crop = diffusion.random_crops(
                hr_real.unsqueeze(1), hsc_hr.unsqueeze(1), crop_size
            )
            y = model.normalize(hr_crop.to(device))
            cond = model.normalize_cond(cond_crop.to(device))

            for group in opt.param_groups:
                group["lr"] = lr * min(1.0, (cur_step + 1) / warmup_steps)
            loss = diffusion.edm_loss(model, y, cond, p_mean, p_std)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ema.update(model, cur_step)

            experiment.log_metrics(
                {
                    "EDM Loss": loss.item(),
                    "Grad Norm": grad_norm.item(),
                },
                step=cur_step,
            )

            # Validation and logging
            if cur_step % display_step == 0 and cur_step > 0:
                hr_real_val, lr_val, hsc_hr_val, _ = next_val_batch()
                hr_real_val = hr_real_val.unsqueeze(1).to(device)
                hsc_hr_val = hsc_hr_val.unsqueeze(1).to(device)
                with torch.no_grad():
                    hr_crop, cond_crop = diffusion.random_crops(hr_real_val, hsc_hr_val, crop_size)
                    val_loss = diffusion.edm_loss(
                        ema.model,
                        ema.model.normalize(hr_crop),
                        ema.model.normalize_cond(cond_crop),
                        p_mean,
                        p_std,
                    ).item()
                print(f"Step: {cur_step}, EDM loss: {loss.item():.5f}, val loss: {val_loss:.5f}")
                val_metrics = {"EDM Val Loss": val_loss}

                # Full-cutout samples: slow (2*sample_steps-1 network calls), so less often
                if cur_step % sample_every == 0:
                    fake_val_images = diffusion.super_resolve(ema.model, hsc_hr_val, sample_steps)
                    val_metrics["L1 Val Reconstruction Loss"] = torch.mean(
                        torch.abs(CenterCrop(600)(fake_val_images) - CenterCrop(600)(hr_real_val))
                    ).item()

                    hr_val = hr_real_val[0, 0].cpu()
                    lr_val_img = lr_val[0].cpu()
                    fake_val = fake_val_images[0, 0].cpu().double()
                    img_diff = CenterCrop(600)(fake_val - hr_val).numpy()
                    vmax = np.abs(img_diff).max()

                    log_figure(
                        CenterCrop(100)(lr_val_img).numpy(),
                        "100x100 Conditioned Val Image (LR)",
                        experiment,
                        step=cur_step,
                    )
                    log_figure(
                        CenterCrop(600)(fake_val).numpy(),
                        "600x600 Generated Val Image (SR)",
                        experiment,
                        step=cur_step,
                    )
                    log_figure(
                        CenterCrop(600)(hr_val).numpy(),
                        "600x600 Real Val Image (HST)",
                        experiment,
                        step=cur_step,
                    )
                    log_figure(
                        img_diff,
                        "Paired Image Difference",
                        experiment,
                        cmap="bwr_r",
                        set_lims=True,
                        lims=[-vmax, vmax],
                        step=cur_step,
                    )

                    # One draw vs the mean and median of stack_samples draws of the same val image
                    # (neo/eval/stack.py): features in one draw but not the median are invented
                    if stack_samples > 1:
                        cond_k = hsc_hr_val[:1].repeat(stack_samples, 1, 1, 1)
                        draws = diffusion.super_resolve(ema.model, cond_k, sample_steps).cpu()
                        stacked = stack.stack(draws)
                        for kind in stack.IMAGES:
                            val_metrics[f"L1 Val {kind} of {stack_samples}"] = stack.l1(
                                stacked[kind][0].numpy(), hr_val.numpy()
                            )
                        fig = stack.stack_figure(
                            f"step {cur_step}",
                            lr_val_img.numpy(),
                            hr_val.numpy(),
                            {k: v[0].numpy() for k, v in stacked.items()},
                            stack_samples,
                        )
                        experiment.log_figure(
                            figure_name="Single vs Stacked Samples", figure=fig, step=cur_step
                        )
                        plt.close(fig)

                experiment.log_metrics(val_metrics, step=cur_step)

            cur_step += 1
            # cur_step now counts completed steps, so a resumed run continues exactly here
            if cur_step % save_steps == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{cur_step:08d}.pt", model, ema, opt, cur_step, model_name
                )
            if cur_step % latest_every == 0:
                save_checkpoint(latest, model, ema, opt, cur_step, model_name)
            if cur_step >= total_steps:
                break

    save_checkpoint(latest, model, ema, opt, cur_step, model_name)


if __name__ == "__main__":
    main()
