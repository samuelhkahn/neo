"""Conditional EDM diffusion (Karras et al. 2022, arXiv:2206.00364) for super-resolution.

SR3-style conditioning: the denoiser sees the noisy HR image concatenated with the 6x-upsampled
LR image (the dataset's `hsc_hr`, the same tensor NEO's discriminator is conditioned on). Images
stay in the training log space (the dataset's ds9 scaling); a per-channel affine normalization
measured on the training set maps them to std sigma_data. EDM preconditioning, log-normal
noise-level sampling and loss weighting; deterministic Heun sampling (EDM Algorithm 1).

Geometry: the model trains on random crops of, and samples, the 624 px window starting at
pixel 72 of the 768 px padded cutout. It covers the central 600 px that NEO is trained and
evaluated on with a 12 px margin; 72 is a multiple of 6, so the LR pixel grid has the same phase
in every crop, and 624 is a multiple of the U-Net's stride of 16.
"""

import copy

import torch
import torch.nn as nn

from neo.models.diffusion_unet import DiffusionUNet

WINDOW_START = 72
WINDOW = 624
GRID = 6
STATS = {
    "hr_mean": 0.0,
    "hr_std": 1.0,
    "cond_mean": 0.0,
    "cond_std": 1.0,
    "hr_min": float("-inf"),
    "hr_max": float("inf"),
}


class EDM(nn.Module):
    def __init__(self, unet: DiffusionUNet, sigma_data=0.5):
        super().__init__()
        self.unet = unet
        self.sigma_data = sigma_data
        # Log-space statistics of the training set (set_stats): mean/std of the HR targets and of
        # the conditioning images, and the HR range that samples are clamped to.
        for name, value in STATS.items():
            self.register_buffer(name, torch.tensor(value))

    def set_stats(self, **stats):
        for name, value in stats.items():
            getattr(self, name).fill_(float(value))

    def normalize(self, hr):
        return (hr - self.hr_mean) / self.hr_std * self.sigma_data

    def denormalize(self, x):
        return x / self.sigma_data * self.hr_std + self.hr_mean

    def normalize_cond(self, cond):
        return (cond - self.cond_mean) / self.cond_std * self.sigma_data

    def forward(self, x, sigma, cond):
        """Denoised estimate D(x; sigma | cond); x and cond normalized, sigma shape (B,)."""
        s = sigma.view(-1, 1, 1, 1)
        sd = self.sigma_data
        c_skip = sd**2 / (s**2 + sd**2)
        c_out = s * sd / (s**2 + sd**2).sqrt()
        c_in = 1 / (sd**2 + s**2).sqrt()
        f = self.unet(torch.cat([c_in * x, cond], dim=1), sigma.log() / 4)
        return c_skip * x + c_out * f


def edm_loss(model: EDM, y, cond, p_mean=-1.2, p_std=1.2):
    """Weighted denoising loss on normalized targets y and conditioning cond, both (B, 1, H, W)."""
    sigma = (torch.randn(y.shape[0], device=y.device) * p_std + p_mean).exp()
    noisy = y + torch.randn_like(y) * sigma.view(-1, 1, 1, 1)
    denoised = model(noisy, sigma, cond)
    sd = model.sigma_data
    weight = ((sigma**2 + sd**2) / (sigma * sd) ** 2).view(-1, 1, 1, 1)
    return (weight * (denoised - y) ** 2).mean()


def sigma_steps(steps, sigma_min=0.002, sigma_max=80.0, rho=7.0, device="cpu"):
    i = torch.arange(steps, dtype=torch.float64, device=device)
    lo, hi = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    t = (hi + i / max(steps - 1, 1) * (lo - hi)) ** rho
    return torch.cat([t, torch.zeros(1, dtype=torch.float64, device=device)])


@torch.no_grad()
def heun_sample(model: EDM, cond, steps=18, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """EDM Algorithm 1 (deterministic Heun, 2*steps-1 denoiser calls); returns a normalized sample.

    The ODE state is kept in float32: the MPS backend has no float64.
    """
    t = sigma_steps(steps, sigma_min, sigma_max, rho).tolist()
    x = torch.randn_like(cond) * t[0]
    for i in range(steps):
        t_cur, t_next = t[i], t[i + 1]
        d = (x - model(x, cond.new_full((x.shape[0],), t_cur), cond)) / t_cur
        x_next = x + (t_next - t_cur) * d
        if i < steps - 1:
            d2 = (x_next - model(x_next, cond.new_full((x.shape[0],), t_next), cond)) / t_next
            x_next = x + (t_next - t_cur) * (0.5 * d + 0.5 * d2)
        x = x_next
    return x


@torch.no_grad()
def super_resolve(model: EDM, cond, steps=18):
    """Sample HR images (B, 1, 768, 768) in log space from the dataset's `hsc_hr` (B, 1, 768, 768).

    Samples are clamped to the training targets' range (the usual final clip of image diffusion
    models; the log scaling turns any overshoot into an exponentially large flux). Only the
    sampling window is generated; outside it the output repeats the conditioning image (it lies
    outside the central 600 px that training losses and evaluation use).
    """
    span = slice(WINDOW_START, WINDOW_START + WINDOW)
    window = (..., span, span)
    sample = heun_sample(model, model.normalize_cond(cond[window]), steps=steps)
    out = cond.clone()
    out[window] = model.denormalize(sample).clamp(model.hr_min, model.hr_max)
    return out


def random_crops(hr, cond, size, generator=None):
    """Matching random crops of HR targets and conditioning images inside the sampling window.

    Offsets are multiples of the LR pixel (6 HR px), matching the window's phase at inference.
    """
    if size > WINDOW:
        raise ValueError(f"crop size {size} exceeds the {WINDOW} px sampling window")
    n_offsets = (WINDOW - size) // GRID + 1
    offsets = WINDOW_START + GRID * torch.randint(n_offsets, (hr.shape[0], 2), generator=generator)
    hr_crops, cond_crops = [], []
    for (y, x), h, c in zip(offsets.tolist(), hr, cond, strict=True):
        hr_crops.append(h[..., y : y + size, x : x + size])
        cond_crops.append(c[..., y : y + size, x : x + size])
    return torch.stack(hr_crops), torch.stack(cond_crops)


class EMA:
    """Exponential moving average of a module's parameters (buffers are copied)."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        decay = min(self.decay, (1 + step) / (10 + step))
        for e, p in zip(self.model.parameters(), model.parameters(), strict=True):
            e.lerp_(p, 1 - decay)
        for e, b in zip(self.model.buffers(), model.buffers(), strict=True):
            e.copy_(b)


def build(section) -> EDM:
    """EDM + U-Net from a config's [DIFFUSION] section (or a dict); missing keys use defaults."""
    unet = DiffusionUNet(
        base=int(section.get("base_channels", 64)),
        mults=tuple(int(m) for m in str(section.get("channel_mults", "1,1,2,3,4")).split(",")),
        blocks=int(section.get("blocks_per_level", 2)),
        dropout=float(section.get("dropout", 0.0)),
    )
    if WINDOW % unet.factor:
        raise ValueError(f"the {WINDOW} px sampling window must be a multiple of {unet.factor}")
    return EDM(unet, sigma_data=float(section.get("sigma_data", 0.5)))
