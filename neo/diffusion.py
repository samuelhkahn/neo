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

Source focus (optional, [DIFFUSION] object_mask / object_weight / source_crop_frac): the loss can
give extra weight to object pixels, NEO-style (its lambda_segmap term), and part of the training
crops can be centred on sources. Two masks: lr_source_mask, a detection on the LR image itself,
depends only on the conditioning and so leaves the denoiser's optimum E[HR | noisy HR, LR]
unchanged (it only moves capacity to sources); the dataset's HST segmentation map depends on
the target and so also tilts the learned distribution toward drawing sources.
"""

import copy
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from neo.eval.stack import to_linear
from neo.models.diffusion_unet import DiffusionUNet

WINDOW_START = 72
WINDOW = 624
GRID = 6
# The dataset centre-crops HR to 600 px (100 LR px) and reflect-pads it by 84 px to 768 px; the
# central 600 px of hsc_hr are the 100 x 100 LR pixels upsampled 6x (nearest).
PAD = 84
CENTER = 600
OBJECT_MASKS = ("none", "lr", "hst")
# lr_source_mask keeps a detected LR pixel only if at least this many of the 3 x 3 pixels around
# it (itself included) are detected: a real source covers several pixels after smoothing
MIN_CLUSTER = 3
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


def edm_loss(
    model: EDM, y, cond, p_mean=-1.2, p_std=1.2, mask=None, object_weight=0.0, parts=False
):
    """Weighted denoising loss on normalized targets y and conditioning cond, both (B, 1, H, W).

    With a binary object mask (same shape as y) the loss adds NEO's masked term (its
    lambda_segmap L1, here on the EDM-weighted squared error):
        mean(err) + object_weight * sum(err * mask) / sum(mask)
    summed over the whole batch, so object_weight = lambda_segmap / lambda_recon. A batch with no
    object pixels gets the first term only. parts=True returns (total, all-pixel term, object
    term), the object term being None without a mask and computed even when object_weight is 0
    (for logging). Without a mask or with object_weight 0 the total is exactly the plain loss.
    """
    sigma = (torch.randn(y.shape[0], device=y.device) * p_std + p_mean).exp()
    noisy = y + torch.randn_like(y) * sigma.view(-1, 1, 1, 1)
    denoised = model(noisy, sigma, cond)
    sd = model.sigma_data
    weight = ((sigma**2 + sd**2) / (sigma * sd) ** 2).view(-1, 1, 1, 1)
    err = weight * (denoised - y) ** 2
    loss_all = err.mean()
    loss_obj = None
    if mask is not None:
        # An empty mask makes the numerator exactly 0 (and its gradient too), so the clamp turns
        # 0 / 0 into 0 without a host sync
        loss_obj = (err * mask).sum() / mask.sum().clamp(min=torch.finfo(err.dtype).tiny)
    total = loss_all
    if loss_obj is not None and object_weight != 0:
        total = loss_all + object_weight * loss_obj
    return (total, loss_all, loss_obj) if parts else total


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


@torch.no_grad()
def draws(model: EDM, cond, k, steps=18, chunk=8):
    """k independent samples (k, 1, 768, 768) of one conditioning image cond (1, 1, 768, 768).

    Sampled `chunk` at a time (each a batch of copies of cond) so a large k fits in GPU memory,
    and returned on the CPU.
    """
    out = []
    for start in range(0, k, chunk):
        n = min(chunk, k - start)
        out.append(super_resolve(model, cond.repeat(n, 1, 1, 1), steps).cpu())
    return torch.cat(out)


def crop_offsets(n, size, mask=None, source_frac=0.0, generator=None):
    """(n, 2) top-left (y, x) offsets of size px crops inside the sampling window (CPU, int64).

    Offsets are multiples of the LR pixel (6 HR px) from the window start, matching the window's
    phase at inference. Every image first gets a uniformly random offset. With a one-channel mask
    (n, 1, H, W) covering the window and source_frac > 0, a fraction of the images instead get a
    crop centred on a uniformly random pixel of their mask: source_frac * n images, rounded up or
    down at random so that the mean is exact. Only pixels that some crop on the grid can cover are
    candidates, and the centred offset is snapped to the grid and clamped to the last grid offset,
    so the crop always contains its pixel. An image whose mask has no candidate keeps its uniform
    crop.

    The selection only uses the mask, so with an LR mask it depends on the conditioning alone.
    A generator must be a CPU one (and the mask then on the CPU); training passes none.
    """
    if size > WINDOW:
        raise ValueError(f"crop size {size} exceeds the {WINDOW} px sampling window")
    n_offsets = (WINDOW - size) // GRID + 1
    offsets = WINDOW_START + GRID * torch.randint(n_offsets, (n, 2), generator=generator)
    if mask is None or source_frac <= 0:
        return offsets

    lo = WINDOW_START
    last = lo + GRID * (n_offsets - 1)
    hi = last + size  # pixels in [lo, hi) are covered by at least one crop on the grid
    # The flat pixel index below is decoded as row and column, so one channel only
    if mask.dim() != 4 or mask.shape[0] != n or mask.shape[1] != 1 or min(mask.shape[2:]) < hi:
        raise ValueError(f"mask must be ({n}, 1, H, W) with H, W >= {hi}, not {tuple(mask.shape)}")
    expected = source_frac * n
    count = int(expected) + int(torch.rand((), generator=generator) < expected - int(expected))
    chosen = torch.randperm(n, generator=generator)[:count]
    if count == 0:
        return offsets
    candidates = mask[chosen.to(mask.device)][..., lo:hi, lo:hi].reshape(count, -1).float()
    found = candidates.sum(1) > 0
    # Images without a candidate draw from uniform weights; that draw is discarded below
    weights = torch.where(found[:, None], candidates, torch.ones_like(candidates))
    pixel = torch.multinomial(weights, 1, generator=generator)[:, 0]
    centre = torch.stack([pixel // (hi - lo), pixel % (hi - lo)], dim=1) + lo
    start = centre - size // 2
    snapped = lo + GRID * torch.div(start - lo + GRID // 2, GRID, rounding_mode="floor")
    snapped = snapped.clamp(lo, last).cpu()
    found = found.cpu()
    offsets[chosen[found]] = snapped[found]
    return offsets


def apply_crops(offsets, size, *images):
    """Crop each of images (B, ..., H, W) at the per-image (y, x) offsets; one stack per input.

    Each tensor stays on its own device, so CPU images and GPU masks can share one set of offsets.
    """
    crops = [[] for _ in images]
    for i, (y, x) in enumerate(offsets.tolist()):
        for out, image in zip(crops, images, strict=True):
            out.append(image[i, ..., y : y + size, x : x + size])
    return tuple(torch.stack(out) for out in crops)


def random_crops(hr, cond, size, generator=None):
    """Matching random crops of HR targets and conditioning images inside the sampling window.

    Offsets are multiples of the LR pixel (6 HR px), matching the window's phase at inference.
    """
    return apply_crops(crop_offsets(hr.shape[0], size, generator=generator), size, hr, cond)


def _gaussian_blur(x, sigma):
    """Separable Gaussian blur of (B, 1, H, W), reflect padding, kernel radius 4 sigma (scipy's)."""
    radius = max(1, int(4 * sigma + 0.5))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kernel.view(1, 1, -1, 1))


def _sky_threshold(flux, nsigma, exclude=None):
    """Per-image median + nsigma * (p75 - median) / 0.6745 of flux (B, 1, h, w), ignoring the
    pixels where exclude is set (an image with nothing left gets NaN)."""
    if exclude is not None:
        flux = flux.masked_fill(exclude, float("nan"))
    q = torch.nanquantile(flux.flatten(1), flux.new_tensor([0.5, 0.75]), dim=1)
    sky, p75 = q.view(2, -1, 1, 1, 1)
    return sky + nsigma * (p75 - sky) / 0.6745


@torch.no_grad()
def lr_detection(cond, nsigma=4.5, smooth=1.0):
    """The detection behind lr_source_mask: the smoothed linear LR grid (B, 1, 100, 100) and each
    image's threshold (B, 1, 1, 1); LR pixels above their image's threshold are sources.

    The sky level and noise are estimated twice: on the whole image, then again without the
    pixels within 2 LR px of what that first threshold detects, so bright sources do not inflate
    the noise estimate. Kept separate so that a threshold can be applied to another image (tests,
    real-data checks).
    """
    grid = cond[..., PAD : PAD + CENTER : GRID, PAD : PAD + CENTER : GRID]
    # to_linear computes in float64 and nanquantile is missing on MPS: compute there on the CPU
    work = grid.cpu() if grid.device.type == "mps" else grid
    flux = to_linear(work).float()
    if smooth > 0:
        flux = _gaussian_blur(flux, smooth)
    first = _sky_threshold(flux, nsigma)
    near_source = F.max_pool2d((flux > first).float(), 5, stride=1, padding=2) > 0
    threshold = _sky_threshold(flux, nsigma, exclude=near_source)
    threshold = torch.where(torch.isnan(threshold), first, threshold)
    return flux.to(cond.device), threshold.to(cond.device)


@torch.no_grad()
def lr_source_mask(cond, nsigma=4.5, smooth=1.0, dilate=2):
    """Binary (B, 1, 768, 768) float32 mask of the sources detected in the LR image.

    cond is the dataset's hsc_hr (B, 1, 768, 768) in the training log space. A cheap detection
    that runs on every training batch on the GPU (lr_detection): the LR pixels (every 6th pixel
    of the central 600) go back to linear flux and are smoothed by a Gaussian of `smooth` LR px;
    a pixel is a source if it lies more than nsigma times the sky noise above the sky level. Both
    are measured per image on the smoothed image as the median and (p75 - median) / 0.6745:
    estimators that only see the upper half of the sky distribution, because the dataset clipped
    the LR at 0 before the log scaling and so piled the lower half up at exactly 0. They are
    measured again without the first pass's detections. A detected pixel is kept only if at least
    MIN_CLUSTER pixels of its 3 x 3 neighbourhood are detected (isolated pixels are noise). The
    mask is dilated by `dilate` LR px (a square), upsampled 6x (nearest) and placed on the central
    600 px, so it is aligned with the LR pixel grid and the padding is never masked.

    On 50 real training cutouts at the defaults the threshold is 3.5-5.7 sigma of the true
    smoothed sky noise (smoothing the clipped sky narrows the estimate, hence nsigma 4.5 for an
    effective ~4). The mask covers 12.5% of the central 600 px (the HST segmentation map 1.8%) and
    97% of the segmentation map's pixels. Noise alone (the raw LR reflected about its sky level,
    2 * sky - LR, which clips the sources away and keeps the noise's distribution) passes the
    same thresholds on 0.03% of the area. The earlier single-pass nsigma 3 version without the
    cluster cut flagged 2.3% (about 16% of its mask), at 3-7 sigma depending on the field.
    """
    flux, threshold = lr_detection(cond, nsigma, smooth)
    detected = (flux > threshold).float()
    count = F.avg_pool2d(detected, 3, stride=1, padding=1, count_include_pad=True) * 9
    detected = detected * (count > MIN_CLUSTER - 0.5)
    if dilate > 0:
        detected = F.max_pool2d(detected, 2 * dilate + 1, stride=1, padding=dilate)
    mask = torch.zeros(cond.shape, dtype=torch.float32, device=cond.device)
    upsampled = detected.repeat_interleave(GRID, dim=-2).repeat_interleave(GRID, dim=-1)
    mask[..., PAD : PAD + CENTER, PAD : PAD + CENTER] = upsampled
    return mask


def central(mask):
    """mask with everything outside the central 600 px (the padding) set to 0."""
    span = slice(PAD, PAD + CENTER)
    out = torch.zeros_like(mask)
    out[..., span, span] = mask[..., span, span]
    return out


@dataclass(frozen=True)
class SourceFocus:
    """[DIFFUSION] options that focus training on sources; the defaults are plain training.

    object_mask       none | lr (lr_source_mask) | hst (the dataset's HST segmentation map,
                      restricted to the central 600 px as in NEO's loss): the mask of the loss's
                      object term (edm_loss), also logged with object_weight 0
    object_weight     weight of the object term (NEO: lambda_segmap / lambda_recon)
    source_crop_frac  fraction of training crops centred on LR-detected sources (crop_offsets);
                      always the LR mask, so arms that differ in object_mask see the same crops
    mask_nsigma, mask_smooth, mask_dilate   lr_source_mask's nsigma, smooth and dilate
    """

    object_mask: str = "none"
    object_weight: float = 0.0
    source_crop_frac: float = 0.0
    mask_nsigma: float = 4.5
    mask_smooth: float = 1.0
    mask_dilate: int = 2

    def __post_init__(self):
        if self.object_mask not in OBJECT_MASKS:
            raise ValueError(
                f"unknown object_mask {self.object_mask!r}; choose from {list(OBJECT_MASKS)}"
            )
        if self.object_weight < 0:
            raise ValueError(f"object_weight must be >= 0, not {self.object_weight}")
        if self.object_weight > 0 and self.object_mask == "none":
            raise ValueError("object_weight > 0 needs object_mask = lr or hst")
        if not 0 <= self.source_crop_frac <= 1:
            raise ValueError(f"source_crop_frac must be in [0, 1], not {self.source_crop_frac}")
        if self.mask_nsigma <= 0 or self.mask_smooth < 0 or self.mask_dilate < 0:
            raise ValueError("need mask_nsigma > 0, mask_smooth >= 0 and mask_dilate >= 0")
        # _gaussian_blur's reflect padding (4 sigma) must stay inside the 100 px LR grid; caught
        # here rather than on the first training batch, after the run's setup
        if int(4 * self.mask_smooth + 0.5) >= CENTER // GRID:
            raise ValueError(f"mask_smooth must be < 24.875 LR px, not {self.mask_smooth}")

    @classmethod
    def from_section(cls, section):
        """From a config's [DIFFUSION] section (or a dict); missing keys use the defaults."""
        d = cls()
        return cls(
            object_mask=str(section.get("object_mask", d.object_mask)).strip(),
            object_weight=float(section.get("object_weight", d.object_weight)),
            source_crop_frac=float(section.get("source_crop_frac", d.source_crop_frac)),
            mask_nsigma=float(section.get("mask_nsigma", d.mask_nsigma)),
            mask_smooth=float(section.get("mask_smooth", d.mask_smooth)),
            mask_dilate=int(section.get("mask_dilate", d.mask_dilate)),
        )

    @property
    def enabled(self):
        return self.object_mask != "none" or self.source_crop_frac > 0

    @property
    def needs_lr_mask(self):
        return self.object_mask == "lr" or self.source_crop_frac > 0

    def name(self):
        """'_option=value' for every option that differs from its default ('' at defaults)."""
        default = SourceFocus()
        values = ((f.name, getattr(self, f.name)) for f in fields(self))
        return "".join(f"_{k}={v}" for k, v in values if v != getattr(default, k))

    def lr_mask(self, cond):
        return lr_source_mask(cond, self.mask_nsigma, self.mask_smooth, self.mask_dilate)

    def weight_mask(self, lr_mask, hst_segmap):
        """The object mask (B, 1, 768, 768) for edm_loss, or None; hst_segmap as the dataset's
        4th element with a channel axis."""
        if self.object_mask == "lr":
            return lr_mask
        if self.object_mask == "hst":
            return central(hst_segmap.float())
        return None


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
