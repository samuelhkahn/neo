"""Turn a model's output back into a physical image; identical for every model under comparison.

Inverse of the training preprocessing in neo.data.dataset (center crop -> clip -> log scale -> pad):
the 768 px output is cropped to its central 600 px (the 84 px reflection pad is removed) and the
fixed log scaling (alpha = 1000, b = 1) is inverted, giving the HR image in the pairs' stored units.
NJYPERPX in each cutout header (neo.preprocess.pairs) converts stored units to nJy per pixel:
e-/s per 0.03" pixel (HR) and HSC counts per 0.168" pixel (LR) for --units paper pairs, 1.0 for
nJy pairs (pairs built before the card existed are nJy).
"""

import numpy as np
import sep

from neo.data.dataset import SR_HST_HSC_Dataset

HR_SIZE = 600  # dataset center-crops HR cutouts to 600 px and pads to 768
LR_SIZE = 100  # and LR cutouts to 100 px, padded to 128
LOG_SCALE_A = 1000
LOG_SCALE_OFFSET = 1


def center_crop(a: np.ndarray, size: int) -> np.ndarray:
    y0 = (a.shape[-2] - size) // 2
    x0 = (a.shape[-1] - size) // 2
    return a[..., y0 : y0 + size, x0 : x0 + size]


def to_physical(output: np.ndarray) -> np.ndarray:
    """(…, 768, 768) model output in log space -> (…, 600, 600) image in the HR pair units."""
    cropped = center_crop(np.asarray(output, dtype=np.float64), HR_SIZE)
    return SR_HST_HSC_Dataset.ds9_unscaling(cropped, a=LOG_SCALE_A, offset=LOG_SCALE_OFFSET)


def njy_per_px(header) -> float:
    """nJy per pixel per stored unit of a pair cutout (NJYPERPX; absent on nJy pairs)."""
    return float(header.get("NJYPERPX", 1.0))


def paper_clip(image: np.ndarray) -> np.ndarray:
    """clip(x, 0, p99.999) of a centre crop, as the training dataset does before log scaling.

    The paper measured HST and LR after the dataset round trip unscale(scale(clip(x))); the round
    trip itself only adds float32 rounding (under 0.01 sky sigma per pixel on our pairs), so the
    clip is applied directly.
    """
    data = np.asarray(image, dtype=np.float32)  # the dataset clips float32 cutouts
    return SR_HST_HSC_Dataset.clip(data, use_data=False)[0].astype(np.float64)


def subtract_background(image: np.ndarray) -> np.ndarray:
    """SEP mesh background subtraction (default 64 px mesh), as applied before cataloging."""
    data = np.ascontiguousarray(image, dtype=np.float64)
    return data - sep.Background(data).back()


def balance_noise(
    image: np.ndarray, tolerance: float = 0.1, rng=None, max_iter: int = 100
) -> np.ndarray:
    """Refill exact zeros with negated positive noise samples.

    Port of `balance_noise` (paper repo, figure_scripts/segmap_photometry/catalog_matching.ipynb):
    clipping at zero during preprocessing removes negative sky noise; this restores it by sampling
    from the iteratively sigma-clipped positive noise and flipping the sign. The original compared
    every iteration's std with the *initial* std and so never terminated unless the first clip was
    already within tolerance; here the clip iterates until the std converges (max_iter cap).
    """
    rng = np.random.default_rng() if rng is None else rng
    image = np.array(image, dtype=np.float64)
    pixel_vals = image.flatten()
    mask = np.ones_like(pixel_vals, dtype=bool)
    curr_std = np.std(image)
    for _ in range(max_iter):
        masked = pixel_vals[mask]
        new_mask = pixel_vals < 3 * np.std(masked)
        new_std = np.std(pixel_vals[new_mask])
        mask = np.logical_and(mask, new_mask)
        if np.abs(new_std - curr_std) < tolerance:
            break
        curr_std = new_std
    negative_noise_samples = pixel_vals[np.logical_and(mask, pixel_vals > 0)] * -1
    ys, xs = np.where(image == 0)
    if len(ys) and len(negative_noise_samples):
        image[ys, xs] = rng.choice(negative_noise_samples, size=ys.shape)
    return image
