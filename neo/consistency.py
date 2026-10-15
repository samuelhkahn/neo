"""Flux consistency ("data consistency") loss: the SR image, pushed through the LSST forward
model, must reproduce the LSST input ([CONSISTENCY] section of a training config).

Physics: the LSST image is the HST sky convolved with the LSST PSF (FWHM ~0.85"; HST's ~0.1")
and binned 6 x 6, in other units. ForwardModel does this to a generator output, differentiably:
  SR (log space, B x 1 x 768 x 768)
  -> linear HR flux in the pairs' stored HR units (neo.noise.to_linear)
  -> x unit_factor = njy_per_px_hr / njy_per_px_lr: nJy per HR pixel, then stored LR units
     (NJYPERPX of the pair headers, neo.eval.postprocess.njy_per_px; 1 / 1 for nJy pairs,
     188.6 / 81.55 for --units paper pairs: read from the first training pair by default)
  -> Gaussian blur of FWHM sqrt(psf_lr^2 - psf_hr^2) in HR pixels (0.2" / 6), on the whole
     768 px output: its 84 px of reflected border is wider than the kernel radius, so no pixel of
     the central 600 px sees any padding of our own
  -> central 600 px, summed in 6 x 6 blocks: the predicted LR image on the central 100 x 100 px
     of the padded 128 px LR grid, in stored LR units.
ConsistencyLoss compares it with the batch's actual LR input, in linear stored units (the
dataset's log scaling inverted; what comes back is clip(LR, 0, p99.999) of the 100 px crop):
  excluded   LR pixels at their image's clip ceiling (the p99.999 clip puts the brightest pixel,
             and any saturated star's core, there), and edge_px pixels along each edge: the PSF
             carries light across the crop boundary that neither the reflected border nor the
             real LR agree on
  sky level  the per-image median of the sky pixels (LR pixels whose 6 x 6 HR block holds no
             segmap source) is subtracted from both before comparing, detached. The real LR's
             sky is not at zero (a small pedestal, and the clip at 0 drops its negative noise),
             and a generator imitating the clipped HST target carries that clip's positive sky
             mean too; neither is the generator's to fix, the L1 terms govern the sky.
  loss       Huber (delta 1) of (pred - lr) / sigma_lr, mean over the included pixels, with
             sigma_lr the LR sky noise ([CONSISTENCY] lr_noise_njy, in nJy per LR pixel so it is
             the same number for nJy and paper-unit pairs; 3.3 nJy on our LSST DP2 i pairs).
  ratio      median over the batch of sum(pred) / sum(lr) over the included source pixels, after
             the sky subtraction: a photometric bias monitor ("Flux Consistency Ratio").
The loss is added to the generator's total as learning_rate x lambda_consistency x loss, like the
other terms (neo/pix2pix.py). In log space its gradient scales with each pixel's linear flux, so
it acts on sources and hardly on sky.

Two defaults are empirical, fitted on real pairs rather than nominal (neo_outputs/scratch/
screening_arms/probe: fit_psf.txt, flux_ratio_check.txt, validate_forward_final.txt), and a
[CONSISTENCY] section without them gets these values (set both explicitly, as
neo/configs/lux_neo_relu_flux.ini does):
  PSF_LR_FWHM = 1.0"  the effective Gaussian that matches the LR pixel by pixel. The core measured
                      from source moments is 0.73-0.84" (it varies by patch) but the coadd PSF has
                      wings a Gaussian lacks; 1.0" on the paper-unit build, 1.15" on the nJy one.
  FLUX_SCALE = 0.87   the LR / model(HST) flux ratio of real sources (F814W is redder and wider
                      than LSST i, and the calibrations differ): 10-15% at every magnitude.
At the nominal 0.85" and flux_scale 1 the model over-predicts every source's LR flux by 13-31%
and the loss pulls all sources below HST, reinforcing the deficit the arm is meant to remove.
The bare ForwardModel keeps flux_scale 1 (it conserves flux); the calibration belongs to the
config path.
"""

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from astropy.io import fits
from torchvision.transforms import CenterCrop

from neo.eval.postprocess import HR_SIZE, LR_SIZE, njy_per_px
from neo.noise import to_linear

FACTOR = 6
LR_PIXEL_ARCSEC = 0.2
PSF_LR_FWHM = 1.0  # effective LSST DP2 i-band Gaussian, arcsec, fitted on real pairs (docstring)
PSF_HR_FWHM = 0.10  # ACS F814W, arcsec
FLUX_SCALE = 0.87  # LR / model(HST) flux ratio of real sources, the from_config default (docstring)
LR_NOISE_NJY = 3.3  # LSST sky sigma per 0.2" pixel on our pairs, nJy
EDGE_PX = 4  # the reflected border's error is < 0.12 sigma_lr (p99) from the 5th pixel in
LAMBDA = 1.0
WARMUP_STEPS = 2000
FWHM_TO_SIGMA = 1 / (2 * math.sqrt(2 * math.log(2)))


def pair_units(hst_dir, hsc_dir):
    """(njy_per_px_hr, njy_per_px_lr) from the headers of the first (sorted) training pair."""
    name = sorted(p.name for p in Path(hst_dir).glob("*.fits"))[0]
    hr = njy_per_px(fits.getheader(Path(hst_dir) / name))
    lr = njy_per_px(fits.getheader(Path(hsc_dir) / name))
    return hr, lr


def gaussian_kernel(sigma_px: float, device=None) -> torch.Tensor:
    """Normalised 1D Gaussian of standard deviation sigma_px, out to 4 sigma ([1] for no blur)."""
    if sigma_px <= 0:
        return torch.ones(1, dtype=torch.float32, device=device)
    radius = int(math.ceil(4 * sigma_px))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2)
    return k / k.sum()


class ForwardModel:
    """Generator output (log space, padded HR grid) -> predicted LR image in stored LR units."""

    def __init__(
        self,
        unit_factor: float,
        psf_lr_fwhm: float = PSF_LR_FWHM,
        psf_hr_fwhm: float = PSF_HR_FWHM,
        lr_pixel_arcsec: float = LR_PIXEL_ARCSEC,
        factor: int = FACTOR,
        device=None,
        flux_scale: float = 1.0,
    ):
        # flux_scale: the empirical LR / model(HST) flux ratio of real sources (F814W is not the
        # i band, and the two surveys' calibrations differ), folded into the unit factor; 1 here
        # so that the bare operator conserves flux, FLUX_SCALE on the config path (from_config)
        self.psf_lr_fwhm = float(psf_lr_fwhm)
        self.flux_scale = float(flux_scale)
        self.unit_factor = float(unit_factor) * self.flux_scale
        self.factor = factor
        hr_pixel = lr_pixel_arcsec / factor
        fwhm = math.sqrt(max(psf_lr_fwhm**2 - psf_hr_fwhm**2, 0.0))
        self.sigma_px = fwhm * FWHM_TO_SIGMA / hr_pixel
        self.kernel = gaussian_kernel(self.sigma_px, device)

    def blur(self, linear: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian blur with reflect padding, same shape."""
        k = self.kernel.to(linear.dtype)
        r = k.numel() // 2
        x = F.pad(linear, (r, r, r, r), mode="reflect")
        x = F.conv2d(x, k.view(1, 1, -1, 1))
        return F.conv2d(x, k.view(1, 1, 1, -1))

    def bin(self, hr: torch.Tensor) -> torch.Tensor:
        """Sum factor x factor blocks: (B, 1, H, W) -> (B, 1, H / factor, W / factor)."""
        b, c, h, w = hr.shape
        f = self.factor
        return hr.reshape(b, c, h // f, f, w // f, f).sum(dim=(3, 5))

    def __call__(self, sr_log: torch.Tensor, size: int = HR_SIZE) -> torch.Tensor:
        """Predicted LR image (B, 1, size / factor, size / factor) of the central `size` HR px."""
        linear = to_linear(sr_log) * self.unit_factor
        return self.bin(CenterCrop(size)(self.blur(linear)))


def masked_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-image median of values where mask is True, (B,); 0 for an image with no such pixel."""
    out = []
    for v, m in zip(values, mask, strict=True):
        pix = v[m]
        out.append(pix.median() if pix.numel() else torch.zeros((), device=v.device, dtype=v.dtype))
    return torch.stack(out)


class ConsistencyLoss:
    def __init__(
        self,
        model: ForwardModel,
        sigma_lr: float,
        weight: float = LAMBDA,
        edge_px: int = EDGE_PX,
        delta: float = 1.0,
        warmup_steps: int = WARMUP_STEPS,
    ):
        self.model = model
        self.sigma_lr = float(sigma_lr)
        self.lambda_consistency = float(weight)
        self.edge_px = int(edge_px)
        self.delta = float(delta)
        self.warmup_steps = int(warmup_steps)
        self.step = 0  # set by the training loop; the warm-up ramps the weight with it

    @property
    def weight(self) -> float:
        """lambda_consistency, ramped linearly from 0 over the first warmup_steps steps.

        At initialisation the output is far from the data in linear flux and the term's gradient
        is ~100x the L1 term's (its log-space gradient grows with the pixel's linear flux): at
        full weight from step 0 the generator did not learn the sky level in a local probe
        (neo_outputs/scratch/screening_arms/probe/dynamics_paper.txt), with the ramp it does.
        """
        if self.warmup_steps <= 0:
            return self.lambda_consistency
        return self.lambda_consistency * min(1.0, (self.step + 1) / self.warmup_steps)

    @classmethod
    def from_config(cls, config, device):
        """[CONSISTENCY]: lambda_consistency, warmup_steps, psf_lr_fwhm_arcsec,
        psf_hr_fwhm_arcsec, flux_scale, lr_noise_njy (nJy per LR pixel), njy_per_px_hr /
        njy_per_px_lr (default: the first training pair's NJYPERPX headers), edge_px, delta."""
        section = config["CONSISTENCY"]
        hr_njy = section.getfloat("njy_per_px_hr", fallback=None)
        lr_njy = section.getfloat("njy_per_px_lr", fallback=None)
        if hr_njy is None or lr_njy is None:
            hst_dir = os.path.expandvars(config["DEFAULT"]["hst_path_train"])
            hsc_dir = os.path.expandvars(config["DEFAULT"]["hsc_path_train"])
            header_hr, header_lr = pair_units(hst_dir, hsc_dir)
            hr_njy = header_hr if hr_njy is None else hr_njy
            lr_njy = header_lr if lr_njy is None else lr_njy
        model = ForwardModel(
            hr_njy / lr_njy,
            section.getfloat("psf_lr_fwhm_arcsec", fallback=PSF_LR_FWHM),
            section.getfloat("psf_hr_fwhm_arcsec", fallback=PSF_HR_FWHM),
            section.getfloat("lr_pixel_arcsec", fallback=LR_PIXEL_ARCSEC),
            device=device,
            flux_scale=section.getfloat("flux_scale", fallback=FLUX_SCALE),
        )
        return cls(
            model,
            sigma_lr=section.getfloat("lr_noise_njy", fallback=LR_NOISE_NJY) / lr_njy,
            weight=section.getfloat("lambda_consistency", fallback=LAMBDA),
            edge_px=section.getint("edge_px", fallback=EDGE_PX),
            delta=section.getfloat("delta", fallback=1.0),
            warmup_steps=section.getint("warmup_steps", fallback=WARMUP_STEPS),
        )

    @staticmethod
    def enabled(config) -> bool:
        return (
            config.has_section("CONSISTENCY")
            and config.getfloat("CONSISTENCY", "lambda_consistency", fallback=LAMBDA) > 0
        )

    def parameters(self) -> dict:
        """What to log for the run."""
        return {
            "lambda_consistency": self.lambda_consistency,
            "consistency_warmup_steps": self.warmup_steps,
            "consistency_psf_lr_fwhm_arcsec": self.model.psf_lr_fwhm,
            "consistency_flux_scale": self.model.flux_scale,
            "consistency_unit_factor": self.model.unit_factor,
            "consistency_blur_sigma_hr_px": self.model.sigma_px,
            "consistency_sigma_lr": self.sigma_lr,
            "consistency_edge_px": self.edge_px,
        }

    def terms(self, sr_log, lr_log, seg):
        """Everything the loss is made of, for logging and tests.

        Returns a dict: pred and lr (B, 1, 100, 100) sky-subtracted in stored LR units, valid and
        source masks, residual in sigma units (zero outside valid), the per-image sky levels.
        """
        pred = self.model(sr_log)
        lr = to_linear(CenterCrop(LR_SIZE)(lr_log))
        seg_hr = CenterCrop(HR_SIZE)(seg)
        source = self.model.bin(seg_hr) > 0  # LR pixels whose HR block holds a segmap source
        ceiling = lr.amax(dim=(2, 3), keepdim=True)
        valid = lr < ceiling
        e = self.edge_px
        if e > 0:
            edge = torch.zeros_like(valid)
            edge[..., e:-e, e:-e] = True
            valid = valid & edge
        sky = valid & ~source
        pred_sky = masked_median(pred.detach(), sky).view(-1, 1, 1, 1)
        lr_sky = masked_median(lr, sky).view(-1, 1, 1, 1)
        pred = pred - pred_sky
        lr = lr - lr_sky
        residual = torch.where(valid, (pred - lr) / self.sigma_lr, torch.zeros_like(pred))
        return {
            "pred": pred,
            "lr": lr,
            "valid": valid,
            "source": source,
            "residual": residual,
            "pred_sky": pred_sky.flatten(),
            "lr_sky": lr_sky.flatten(),
        }

    def __call__(self, sr_log, lr_log, seg):
        """(loss, ratio): Huber loss over the included pixels and the photometric ratio."""
        t = self.terms(sr_log, lr_log, seg)
        huber = F.huber_loss(
            t["residual"], torch.zeros_like(t["residual"]), reduction="none", delta=self.delta
        )
        n_valid = t["valid"].sum().clamp(min=1)
        loss = huber.sum() / n_valid
        src = (t["valid"] & t["source"]).to(t["pred"].dtype)
        flux_pred = (t["pred"] * src).sum(dim=(1, 2, 3))
        flux_lr = (t["lr"] * src).sum(dim=(1, 2, 3))
        has_source = src.sum(dim=(1, 2, 3)) > 0
        if has_source.any():
            ratio = (flux_pred[has_source] / flux_lr[has_source]).median().detach()
        else:
            ratio = torch.full((), float("nan"), device=loss.device, dtype=loss.dtype)
        return loss, ratio
