"""Stack several draws of a stochastic model (diffusion, or a GAN with dropout on) per cutout.

A diffusion model draws a different HR image from p(HR | LR) each time it samples. A feature that
appears in one draw but not the others (a "ghost") was invented by that draw rather than required
by the LR image. Stacking K draws suppresses such features:
  single  the first draw, i.e. what one sample looks like
  mean    pixelwise mean in linear flux (a log-space mean would be a geometric mean, biased low
          on sources): the posterior-mean flux, unbiased for photometry but smoother
  median  pixelwise median, the same in log and linear space since the scaling is monotonic; for
          even K the two middle draws are averaged in linear flux. An L1-trained generator such as
          NEO estimates this per-pixel median, so it is the like-for-like comparison with NEO
  std     pixelwise standard deviation in linear flux: where the draws disagree
Draws and the single/mean/median stacks are in the training log space (neo.data.dataset ds9
scaling); std is in linear stored HR units.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.visualization import simple_norm

from neo.eval.postprocess import HR_SIZE, LOG_SCALE_A, LOG_SCALE_OFFSET, LR_SIZE, center_crop

IMAGES = ("single", "mean", "median")  # log-space images, written and scored like any prediction
KINDS = (*IMAGES, "std")


def to_linear(x: torch.Tensor) -> torch.Tensor:
    """Training log space -> linear stored units (neo.data.dataset ds9_unscaling)."""
    return torch.expm1((x.double() + LOG_SCALE_OFFSET) * math.log(LOG_SCALE_A + 1)) / LOG_SCALE_A


def to_log(v: torch.Tensor) -> torch.Tensor:
    """Linear stored units -> training log space (neo.data.dataset ds9_scaling)."""
    log = torch.log10(LOG_SCALE_A * v.double() + 1) / math.log10(LOG_SCALE_A + 1)
    return log - LOG_SCALE_OFFSET


def stack(draws: torch.Tensor) -> dict:
    """K draws (K, ...) in log space -> {single, mean, median (log space), std (linear)}."""
    k = draws.shape[0]
    linear = to_linear(draws)
    ordered = draws.sort(dim=0).values
    if k % 2:
        median = ordered[k // 2]
    else:
        middle = to_linear(ordered[k // 2 - 1 : k // 2 + 1]).mean(0)
        median = to_log(middle).to(draws.dtype)
    return {
        "single": draws[0],
        "mean": to_log(linear.mean(0)).to(draws.dtype),
        "median": median,
        "std": linear.std(0, unbiased=k > 1).to(draws.dtype),
    }


def l1(image, hr) -> float:
    """Log-space L1 against the HR target over the central 600 px (predict.py's L1LOG)."""
    return float(np.mean(np.abs(center_crop(image, HR_SIZE) - center_crop(hr, HR_SIZE))))


def stack_figure(name, lr, hr, stacked, k):
    """Single draw vs stacks of k draws, central 600 px, numpy arrays in the training log space.

    Top row:    LR | single | mean | median | std of the draws (linear flux, asinh stretch)
    Bottom row: HST | single - HST | mean - HST | median - HST | single - median
    SR images share the HST colour scale (LR has its own); differences share one symmetric scale,
    so a ghost shows up in single - median and in single - HST but not in median - HST.
    """
    crop = {kind: center_crop(np.asarray(v, np.float64), HR_SIZE) for kind, v in stacked.items()}
    hst = center_crop(np.asarray(hr, dtype=np.float64), HR_SIZE)
    vmin, vmax = np.percentile(hst, [1, 99.8])
    diffs = {f"{kind} - HST": crop[kind] - hst for kind in IMAGES}
    diffs["single - median"] = crop["single"] - crop["median"]
    lim = max(np.percentile(np.abs(d), 99.5) for d in diffs.values()) or 1.0

    fig, axes = plt.subplots(2, 5, figsize=(22, 9.6), layout="constrained")
    lr = center_crop(np.asarray(lr, dtype=np.float64), LR_SIZE)
    axes[0, 0].imshow(lr, origin="lower", cmap="plasma", vmin=np.percentile(lr, 1), vmax=lr.max())
    axes[0, 0].set_title("LR")
    for ax, kind in zip(axes[0, 1:4], IMAGES, strict=True):
        im = ax.imshow(crop[kind], origin="lower", cmap="plasma", vmin=vmin, vmax=vmax)
        label = kind if kind == "single" else f"{kind} of {k}"
        ax.set_title(f"{label}   L1 {np.mean(np.abs(crop[kind] - hst)):.4f}")
    fig.colorbar(im, ax=axes[0, 1:4], shrink=0.8, label="log-scaled flux")
    std = crop["std"]
    im = axes[0, 4].imshow(std, origin="lower", cmap="magma", norm=simple_norm(std, "asinh"))
    axes[0, 4].set_title(f"std of {k} draws (linear flux)")
    fig.colorbar(im, ax=axes[0, 4], shrink=0.8)

    axes[1, 0].imshow(hst, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("HST")
    for ax, (title, d) in zip(axes[1, 1:], diffs.items(), strict=True):
        im = ax.imshow(d, origin="lower", cmap="bwr_r", vmin=-lim, vmax=lim)
        ax.set_title(title)
    fig.colorbar(im, ax=axes[1, 1:], shrink=0.8, label="log-space difference")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{name}: one draw vs stacks of {k} draws")
    return fig
