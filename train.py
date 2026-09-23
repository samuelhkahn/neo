"""Training script for the NEO super-resolution model.

Trains a Pix2Pix conditional GAN to translate ground-based HSC images to
space-based HST quality using paired FITS image cutouts.

Usage:
    python train.py <config_file> [--resume]

Example:
    python train.py neo/configs/example.ini
    python train.py neo/configs/lux.ini --resume   # continue from <ckpt_dir>/latest.pt
"""

import argparse
import configparser
import os
from pathlib import Path

import numpy as np
import torch
from comet_ml import Experiment, OfflineExperiment
from torchvision.transforms import CenterCrop
from tqdm import tqdm

from neo.data.collate_fn import collate_fn
from neo.data.dataset import SR_HST_HSC_Dataset
from neo.log_figure import log_figure
from neo.pix2pix import Pix2Pix


def save_checkpoint(path, pix2pix, step, model_name):
    """Write model + optimizer state atomically, so a killed job never leaves a corrupt file."""
    state = {
        "step": step,
        "model_name": model_name,
        "gen": pix2pix.gen.state_dict(),
        "patch_gan": pix2pix.patch_gan.state_dict(),
        "gen_opt": pix2pix.gen_opt.state_dict(),
        "disc_opt": pix2pix.disc_opt.state_dict(),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, pix2pix, device):
    state = torch.load(path, map_location=device, weights_only=True)
    pix2pix.gen.load_state_dict(state["gen"])
    pix2pix.patch_gan.load_state_dict(state["patch_gan"])
    pix2pix.gen_opt.load_state_dict(state["gen_opt"])
    pix2pix.disc_opt.load_state_dict(state["disc_opt"])
    return state["step"]


def main():
    parser = argparse.ArgumentParser(description="Train NEO from an .ini config")
    parser.add_argument("config", help="training .ini file")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <ckpt_dir>/latest.pt if it exists")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"device: {device}")

    # Load configuration
    config = configparser.ConfigParser()
    config.read(args.config)

    # Data paths; environment variables such as ${NEO_DATA} are expanded
    hst_path_train = os.path.expandvars(config["DEFAULT"]["hst_path_train"])
    hsc_path_train = os.path.expandvars(config["DEFAULT"]["hsc_path_train"])
    hst_path_val = os.path.expandvars(config["DEFAULT"]["hst_path_val"])
    hsc_path_val = os.path.expandvars(config["DEFAULT"]["hsc_path_val"])

    # Image dimensions
    hst_dim = int(config["HST_DIM"]["hst_dim"])
    hsc_dim = int(config["HSC_DIM"]["hsc_dim"])

    # Training parameters
    comet_tag = config["COMET_TAG"]["comet_tag"]
    comet_project = config.get("COMET_PROJECT", "comet_project", fallback="neo-rubin-lsst")
    batch_size = int(config["BATCH_SIZE"]["batch_size"])
    total_steps = int(config["GAN_STEPS"]["gan_steps"])
    save_steps = int(config["SAVE_STEPS"]["save_steps"])
    ckpt_dir = Path(os.path.expandvars(config.get("CHECKPOINT", "ckpt_dir", fallback="models")))
    latest_every = config.getint("CHECKPOINT", "latest_every", fallback=save_steps)
    num_workers = config.getint("DATALOADER", "num_workers", fallback=0)
    data_aug = eval(config["DATA_AUG"]["data_aug"])
    identifier = eval(config["IDENTIFIER"]["identifier"])
    display_step = eval(config["DISPLAY_STEPS"]["display_steps"])

    # Optimizer parameters
    lr = eval(config["LR"]["lr"])
    disc_lr = eval(config["DISC_LR"]["disc_lr"])

    # Loss weights
    lambda_recon = eval(config["LAMBDA_RECON"]["lambda_recon"])
    lambda_segmap = eval(config["LAMBDA_SEGMAP"]["lambda_segmap"])
    lambda_vgg = eval(config["LAMBDA_VGG"]["lambda_vgg"])
    lambda_adv = eval(config["LAMBDA_ADV"]["lambda_adv"])

    # Discriminator settings
    disc_update_freq = int(config["DISC_UPDATE_FREQ"]["disc_update_freq"])

    # Pretrained models
    pretrained_generator = config["PRETRAINED_GENERATOR"]["pretrained_generator"]
    pretrained_discriminator = config["PRETRAINED_DISCRIMINATOR"]["pretrained_discriminator"]
    vgg_loss_weights = eval(config["VGG_LOSS_WEIGHTS"]["vgg_loss_weights"])

    # Comet ML experiment tracking; logs locally when no API key is set
    api_key = os.environ.get('COMET_ML_ASTRO_API_KEY')
    comet_kwargs = dict(project_name=comet_project, workspace="samkahn-astro")
    if api_key:
        experiment = Experiment(api_key=api_key, **comet_kwargs)
    else:
        print("COMET_ML_ASTRO_API_KEY not set; logging offline to ./comet_offline")
        experiment = OfflineExperiment(offline_directory="comet_offline", **comet_kwargs)

    experiment.add_tag(comet_tag)
    experiment.log_parameter("batch_size", batch_size)
    experiment.log_parameter("total_steps", total_steps)
    experiment.log_parameter("save_steps", save_steps)
    experiment.log_parameter("data_aug", data_aug)
    experiment.log_parameter("display_step", display_step)
    experiment.log_parameter("lr", lr)
    experiment.log_parameter("disc_lr", disc_lr)
    experiment.log_parameter("lambda_recon", lambda_recon)
    experiment.log_parameter("lambda_vgg", lambda_vgg)
    experiment.log_parameter("lambda_segrecon", lambda_segmap)
    experiment.log_parameter("lambda_adv", lambda_adv)
    experiment.log_parameter("disc_update_freq", disc_update_freq)
    for i in range(5):
        experiment.log_parameter(f"vgg_layer_{i+1}", vgg_loss_weights[i])

    model_name = (
        f"gaussian_bcegan_{identifier}_global_lr={lr}_recon={lambda_recon}"
        f"_segrecon={lambda_segmap}_vgg={lambda_vgg}"
        f"_adv={lambda_adv}_discupdate={disc_update_freq}"
        f"_vgglayer_weights_{str(vgg_loss_weights)}"
    )
    print(model_name)

    # Create dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        SR_HST_HSC_Dataset(
            hst_path=hst_path_train, hsc_path=hsc_path_train,
            hr_size=[hst_dim, hst_dim], lr_size=[hsc_dim, hsc_dim],
            transform_type="ds9_scale", data_aug=data_aug, experiment=None,
        ),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=device == "cuda",
    )

    dataloader_val = torch.utils.data.DataLoader(
        SR_HST_HSC_Dataset(
            hst_path=hst_path_val, hsc_path=hsc_path_val,
            hr_size=[hst_dim, hst_dim], lr_size=[hsc_dim, hsc_dim],
            transform_type="ds9_scale", data_aug=data_aug, experiment=None,
        ),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=device == "cuda",
    )
    val_iter = iter(dataloader_val)

    def next_val_batch():
        nonlocal val_iter
        try:
            return next(val_iter)
        except StopIteration:
            val_iter = iter(dataloader_val)
            return next(val_iter)

    # Initialize model
    pix2pix = Pix2Pix(
        in_channels=1, out_channels=1, device=device,
        learning_rate=lr, disc_learning_rate=disc_lr,
        vgg_loss_weights=vgg_loss_weights, lambda_recon=lambda_recon,
        lambda_segmap=lambda_segmap, lambda_vgg=lambda_vgg,
        lambda_adv=lambda_adv,
        display_step=display_step, pretrained_generator=pretrained_generator,
        pretrained_discriminator=pretrained_discriminator,
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cur_step = 0
    latest = ckpt_dir / "latest.pt"
    if args.resume and latest.exists():
        cur_step = load_checkpoint(latest, pix2pix, device)
        print(f"resumed from {latest} at step {cur_step}")
    experiment.log_parameter("start_step", cur_step)

    # Training loop
    while cur_step < total_steps:
        for hr_real, lr, hsc_hr, seg_map_real in tqdm(dataloader_train, position=0):
            # Add channel dimension: (B, H, W) -> (B, 1, H, W)
            hr_real = hr_real.unsqueeze(1).to(device)
            hsc_hr = hsc_hr.unsqueeze(1).to(device)
            lr = lr.unsqueeze(1).to(device)
            seg_map_real = seg_map_real.unsqueeze(1).to(device)

            # Generator step
            losses = pix2pix.training_step(hr_real, lr, hsc_hr, seg_map_real, "generator")
            gen_loss = losses[0].item()
            adv_loss = losses[1].item()
            recon_loss = losses[2].item()
            vgg_loss = losses[3].item()
            segmap_loss = losses[4].item()

            # Discriminator step (at specified frequency)
            if cur_step % disc_update_freq == 0:
                disc_losses = pix2pix.training_step(hr_real, lr, hsc_hr, seg_map_real, "discriminator")
                disc_loss = disc_losses[0].item()
                fake_disc_logits = disc_losses[1]
                real_disc_logits = disc_losses[2]

            experiment.log_metrics({
                "Generator Loss": gen_loss,
                "Discriminator Loss": disc_loss,
                "VGG Loss": vgg_loss,
                "L1 Reconstruction Loss": recon_loss,
                "L1 Segmap Reconstruction Loss": segmap_loss,
                "L1 Segmap/L1 Recon Ratio": segmap_loss / recon_loss,
                "Adversarial Loss": adv_loss,
            }, step=cur_step)

            # Validation and logging
            if cur_step % display_step == 0 and cur_step > 0:
                hr_real_val, lr_val, hsc_hr_val, seg_map_real_val = next_val_batch()

                hr_real_val = hr_real_val.unsqueeze(1).to(device)
                hsc_hr_val = hsc_hr_val.unsqueeze(1).to(device)
                lr_val = lr_val.unsqueeze(1).to(device)
                seg_map_real_val = seg_map_real_val.unsqueeze(1).to(device)

                val_losses = pix2pix.validation_step(hr_real_val, lr_val, hsc_hr_val, seg_map_real_val, "generator")
                gen_val_loss = val_losses[0].item()
                adv_val_loss = val_losses[1].item()
                recon_val_loss = val_losses[2].item()
                vgg_val_loss = val_losses[3].item()
                segmap_val_loss = val_losses[4].item()

                disc_val_losses = pix2pix.validation_step(hr_real_val, lr_val, hsc_hr_val, seg_map_real_val, "discriminator")
                disc_val_loss = disc_val_losses[0].item()
                fake_disc_val_logits = disc_val_losses[1]
                real_disc_val_logits = disc_val_losses[2]

                fake_val_images = pix2pix.generate_fake_images(lr_val)
                print(f'Step: {cur_step}, Generator loss: {gen_val_loss:.5f}, Discriminator loss: {disc_val_loss:.5f}')

                # Extract single images for visualization
                hr_val = hr_real_val[0, :, :, :].squeeze(0).cpu()
                lr_val_img = lr_val[0, :, :, :].squeeze(0).cpu()
                fake_val = fake_val_images[0, 0, :, :].cpu().double()
                real_disc_val_map = real_disc_val_logits[0, 0, :, :].cpu()
                fake_disc_val_map = fake_disc_val_logits[0, 0, :, :].cpu()

                # Log visualization figures
                img_diff = CenterCrop(600)(fake_val - hr_val).cpu().detach().numpy()
                vmax = np.abs(img_diff).max()

                log_figure(CenterCrop(100)(lr_val_img).detach().numpy(), "100x100 Conditioned Val Image (LR)", experiment, step=cur_step)
                log_figure(CenterCrop(600)(fake_val).detach().numpy(), "600x600 Generated Val Image (SR)", experiment, step=cur_step)
                log_figure(CenterCrop(600)(hr_val).detach().numpy(), "600x600 Real Val Image (HST)", experiment, step=cur_step)
                log_figure(real_disc_val_map.detach().numpy(), "Real Disc Val Logits", experiment, step=cur_step)
                log_figure(fake_disc_val_map.detach().numpy(), "Fake Disc Val Logits", experiment, step=cur_step)
                log_figure(img_diff, "Paired Image Difference", experiment, cmap="bwr_r", set_lims=True, lims=[-vmax, vmax], step=cur_step)

                experiment.log_metrics({
                    "Generator Val Loss": gen_val_loss,
                    "Discriminator Val Loss": disc_val_loss,
                    "VGG Val Loss": vgg_val_loss,
                    "L1 Val Reconstruction Loss": recon_val_loss,
                    "L1 Val Segmap Reconstruction Loss": segmap_val_loss,
                    "L1 Val Segmap/L1 Recon Ratio": segmap_val_loss / recon_val_loss,
                    "Adversarial Val Loss": adv_val_loss,
                }, step=cur_step)

            cur_step += 1
            # cur_step now counts completed steps, so a resumed run continues exactly here
            if cur_step % save_steps == 0:
                save_checkpoint(ckpt_dir / f"step_{cur_step:08d}.pt", pix2pix, cur_step, model_name)
            if cur_step % latest_every == 0:
                save_checkpoint(latest, pix2pix, cur_step, model_name)
            if cur_step >= total_steps:
                break

    save_checkpoint(latest, pix2pix, cur_step, model_name)
    # Whole-model files for the inference snippet in the README
    torch.save(pix2pix.gen, ckpt_dir / f'gen_pix2pixsr_{model_name}_final.pt')
    torch.save(pix2pix.patch_gan, ckpt_dir / f'patchgan_pix2pixsr_{model_name}_final.pt')


if __name__ == "__main__":
    main()
