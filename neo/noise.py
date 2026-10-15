"""HST-like sky noise for the noise-separation arm ([NOISE] section of a training config).

The generator predicts a clean (noise-free) HR image s, and a noise realisation n is added before
the discriminator (and, by default, the VGG loss), so the generator no longer has to synthesise
sky noise itself. The L1 terms compare the clean s with the noisy HST target: their optimum is the
clean conditional median, which is what we want. At inference the clean image is the output;
predict.py --noise hst adds a realisation at the config default sigma for realism comparisons.

Noise model, in the pairs' stored linear HR units (neo.eval.postprocess.njy_per_px):
  template   the azimuthally averaged noise power spectrum of source-free HST sky, measured once
             from the first n_files training HR cutouts (sorted names; sources masked with the
             dataset's own SEP extraction, dilated), as the masked autocorrelation up to MAX_LAG
             pixels (FFT-based, so every sky pixel counts), symmetrised and transformed. Drizzled
             HST noise is correlated (lag-1 autocorrelation ~0.7 on our pairs), so white noise
             would not look like HST sky. The template is cached as noise_template.npz in the
             run's [CHECKPOINT] ckpt_dir (or given as [NOISE] template = <file>); it also records
             each file's sky sigma, whose median is the inference default ([NOISE] sigma). Only
             training measures one: the evaluation tools read the cache or the named file and
             stop without it, as a template is in the stored units of the pairs it came from.
  sigma      per image, the HST target's own sky sigma, so the fake gets the depth of its target.
             The target is clipped at 0 before log scaling (neo.data.dataset), so about half of
             its sky pixels are exactly 0 and a MAD would read ~0: clipped_sky_sigma() fits a
             clipped Gaussian to the sky pixels outside the segmap instead (zero fraction and
             median of the positive pixels). A target without a measurable sky (its segmap covers
             the crop) gets the batch's median sigma, never a nan (NoiseModel.target_sigma).
             The noise is stationary at that sky sigma: real HST sources carry photon noise on
             top of it (1.4-1.5x the sky scatter at 10-30 sigma, more on bright cores), which
             the fake lacks; the discriminator columns of the sweep inherit that difference.
  sample     white Gaussian noise shaped by the template: convolution with the real kernel whose
             power spectrum is the template (FFT shaping in the spatial domain, which runs on
             every device), scaled to sigma.
  add_noise  to_linear(s) + n, clipped at 0 as the dataset clips HST, back to log space;
             differentiable with respect to s (the clip has zero gradient where it acts, as the
             target's clipped pixels carry no information either).
"""

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from scipy.ndimage import binary_dilation

from neo.eval.postprocess import HR_SIZE, LOG_SCALE_A, LOG_SCALE_OFFSET, center_crop

LN_A1 = math.log(LOG_SCALE_A + 1)
MAX_LAG = 15  # autocorrelation lags measured (the HST ACF is ~0 beyond 6 px)
GRID = 64  # FFT grid the spectrum and the shaping kernel are built on
KERNEL = 31  # shaping kernel size (odd); the kernel is ~0 beyond MAX_LAG
N_FILES = 32
MASK_DILATION = 3  # px around each SEP source left out of the sky
TEMPLATE_FILE = "noise_template.npz"


def to_linear(x: torch.Tensor) -> torch.Tensor:
    """Training log space -> linear stored units, in x's dtype (stack.to_linear is double)."""
    return torch.expm1((x + LOG_SCALE_OFFSET) * LN_A1) / LOG_SCALE_A


def to_log(v: torch.Tensor) -> torch.Tensor:
    """Linear stored units -> training log space, in v's dtype (stack.to_log is double)."""
    return torch.log1p(LOG_SCALE_A * v) / LN_A1 - LOG_SCALE_OFFSET


# ---------------------------------------------------------------------------------------------
# template: measured once from training HST cutouts


def sky_mask(data: np.ndarray) -> np.ndarray:
    """True on sky: the dataset's SEP extraction (3 sigma, segmentation map) dilated."""
    import sep

    data = np.ascontiguousarray(data, dtype=np.float32)
    bkg = sep.Background(data)
    seg = sep.extract(data, 3, err=bkg.globalrms, segmentation_map=True)[1]
    return ~binary_dilation(seg > 0, iterations=MASK_DILATION)


def sky_autocorrelation(data: np.ndarray, mask: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """Normalised autocorrelation of the sky (mask True) for lags -max_lag..max_lag per axis.

    Masked estimate: sum of products over pairs of sky pixels divided by the number of such
    pairs, both via FFT on a zero-padded grid (so lags never wrap around the image).
    """
    data = np.asarray(data, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    x = np.where(mask, data - np.median(data[mask]), 0.0)
    m = mask.astype(np.float64)
    shape = (data.shape[0] + max_lag, data.shape[1] + max_lag)
    num = np.fft.irfft2(np.abs(np.fft.rfft2(x, s=shape)) ** 2, s=shape)
    den = np.fft.irfft2(np.abs(np.fft.rfft2(m, s=shape)) ** 2, s=shape)
    lags = np.arange(-max_lag, max_lag + 1)  # negative indices read the wrapped (negative) lags
    acf = num[np.ix_(lags, lags)] / np.maximum(den[np.ix_(lags, lags)], 1.0)
    return acf / acf[max_lag, max_lag]


def symmetrise(acf: np.ndarray) -> np.ndarray:
    """Average over the 8 symmetries of the pixel grid (flips and transposition)."""
    a = (acf + acf[::-1] + acf[:, ::-1] + acf[::-1, ::-1]) / 4
    return (a + a.T) / 2


def radial_power(acf: np.ndarray, grid: int = GRID):
    """(k, P): azimuthally averaged power spectrum of an autocorrelation, k in cycles per pixel.

    The ACF is placed at the origin of a grid x grid array and Fourier transformed (real, as it is
    symmetric); the 2D spectrum is averaged in rings of width 1 / grid in |k|. Negative values
    (truncation of the ACF) are clipped: a power spectrum is nonnegative.
    """
    lag = acf.shape[0] // 2
    padded = np.zeros((grid, grid))
    idx = np.arange(-lag, lag + 1)
    padded[np.ix_(idx, idx)] = acf  # wrapped: lag 0 at [0, 0], negative lags at the far end
    power = np.clip(np.fft.fft2(padded).real, 0.0, None)
    freq = np.fft.fftfreq(grid)
    k = np.hypot(freq[:, None], freq[None, :])
    bins = np.round(k * grid).astype(int)
    n_bins = bins.max() + 1
    counts = np.bincount(bins.ravel(), minlength=n_bins)
    sums = np.bincount(bins.ravel(), weights=power.ravel(), minlength=n_bins)
    keep = counts > 0
    return np.arange(n_bins)[keep] / grid, sums[keep] / counts[keep]


def measure_template(paths, n_files: int = N_FILES, size: int = HR_SIZE) -> dict:
    """Noise template from the first n_files HST cutouts (sorted by name): the central `size` px
    of each raw (unclipped) cutout, sources masked, sky sigma as 1.4826 MAD."""
    paths = sorted(Path(p) for p in paths)[:n_files]
    if not paths:
        raise ValueError("no HST cutouts to measure the noise template from")
    acfs, sigmas = [], []
    for path in paths:
        data = center_crop(fits.getdata(path).astype(np.float64), size)
        mask = sky_mask(data)
        sky = data[mask]
        sigmas.append(1.4826 * np.median(np.abs(sky - np.median(sky))))
        acfs.append(sky_autocorrelation(data, mask))
    acf = symmetrise(np.mean(acfs, axis=0))
    k, power = radial_power(acf)
    sigmas = np.array(sigmas)
    return {
        "k": k,
        "power": power,
        "acf": acf,
        "sigma": sigmas,
        "sigma_median": float(np.median(sigmas)),
        "files": np.array([p.name for p in paths]),
    }


def save_template(path, template: dict) -> None:
    np.savez(path, **template)


def load_template(path) -> dict:
    with np.load(path) as data:
        template = {key: data[key] for key in data.files}
    template["sigma_median"] = float(template["sigma_median"])
    return template


def shaping_kernel(template: dict, size: int = KERNEL, grid: int = GRID) -> np.ndarray:
    """Real, symmetric kernel h with |FFT(h)|^2 = the template spectrum, sum(h^2) = 1.

    White noise convolved with h has unit variance and the template's autocorrelation.
    """
    freq = np.fft.fftfreq(grid)
    k = np.hypot(freq[:, None], freq[None, :])
    amplitude = np.sqrt(np.interp(k, template["k"], template["power"]))
    h = np.fft.fftshift(np.fft.ifft2(amplitude).real)
    c = grid // 2
    r = size // 2
    h = h[c - r : c + r + 1, c - r : c + r + 1]
    return h / np.sqrt(np.sum(h**2))


# ---------------------------------------------------------------------------------------------
# per-image sky sigma of clipped targets


def clipped_sky_sigma(linear: torch.Tensor, sky: torch.Tensor, min_clipped=0.02) -> torch.Tensor:
    """Per-image sky sigma (B,) of images clipped at 0, from the sky pixels (sky True).

    For sky ~ N(mu, sigma) clipped at 0, the fraction of zeros is z = Phi(-mu / sigma) and the
    median m of the positive pixels satisfies Phi((m - mu) / sigma) = (1 + z) / 2, so
    sigma = m / (ndtri((1 + z) / 2) - ndtri(z)) (= m / 0.6745 for mu = 0). Images with fewer
    than min_clipped zeros are unclipped data: 1.4826 x MAD of their sky.

    An image whose sigma cannot be measured gets nan: no sky pixel outside its segmap, no
    positive sky pixel, or fewer than 2% positive ones (the fit is meaningless there; a median of
    an empty tensor would also be nan, silently). NoiseModel.target_sigma replaces those, a nan
    must never reach the noise. Cutouts built by neo.preprocess.pairs hold no blank mosaic
    pixel, so the zero fraction is the clip's alone.
    """
    values = []
    nan = float("nan")
    for v, m in zip(linear, sky, strict=True):
        pix = v[m]
        if pix.numel() == 0:
            values.append(torch.full((), nan, device=v.device, dtype=v.dtype))
            continue
        z = float((pix <= 0).to(v.dtype).mean().cpu())  # ndtri on the CPU: a scalar per image
        if z < min_clipped:
            values.append(1.4826 * (pix - pix.median()).abs().median())
            continue
        positive = pix[pix > 0]
        if z > 0.98 or positive.numel() == 0:
            values.append(torch.full((), nan, device=v.device, dtype=v.dtype))
            continue
        denominator = torch.special.ndtri(torch.tensor([(1 + z) / 2, z], dtype=torch.float64))
        values.append(positive.median() / float(denominator[0] - denominator[1]))
    return torch.stack(values)


# ---------------------------------------------------------------------------------------------


class NoiseModel:
    """Sample and add HST-like noise, in the stored linear HR units of the training pairs.

    Args:
        template: measure_template() / load_template() result.
        device: where the noise is drawn.
        sigma_default: sky sigma used at inference (stored HR units).
        vgg_on: "noisy" (the VGG loss compares s + n with HST, like the discriminator) or "clean".
    """

    def __init__(self, template: dict, device, sigma_default=None, vgg_on="noisy"):
        if vgg_on not in ("noisy", "clean"):
            raise ValueError(f"[NOISE] vgg_on must be noisy or clean, not {vgg_on!r}")
        self.template = template
        self.device = device
        self.vgg_on = vgg_on
        self.sigma_default = template["sigma_median"] if sigma_default is None else sigma_default
        self.n_fallbacks = 0  # targets whose sky sigma could not be measured (target_sigma)
        kernel = torch.as_tensor(shaping_kernel(template), dtype=torch.float32)
        self.kernel = kernel.view(1, 1, *kernel.shape).to(device)

    @classmethod
    def from_config(cls, config, device, measure=True):
        """[NOISE]: template (file; default: <ckpt_dir>/noise_template.npz, measured from the
        first n_files of [DEFAULT] hst_path_train when missing), sigma (default: the template's
        median training sky sigma), vgg_on (default noisy).

        measure=False (the evaluation tools) stops instead of measuring: a template is in the
        stored units of the pairs it was measured from and its spectrum differs between pair
        builds, so a checkpoint directory copied without its cache must not get one measured
        from whatever pairs are local; the cache travels with the checkpoints.
        """
        section = config["NOISE"]
        path = section.get("template", fallback="")
        if path:
            template = load_template(os.path.expandvars(path))
        else:
            ckpt_dir = config.get("CHECKPOINT", "ckpt_dir", fallback="models")
            ckpt_dir = Path(os.path.expandvars(ckpt_dir))
            cache = ckpt_dir / TEMPLATE_FILE
            if cache.exists():
                template = load_template(cache)
            elif not measure:
                raise FileNotFoundError(
                    f"{cache} is missing: copy it with the checkpoints (training writes it), or "
                    "name the run's template with [NOISE] template = <file>"
                )
            else:
                hst_dir = Path(os.path.expandvars(config["DEFAULT"]["hst_path_train"]))
                n_files = section.getint("n_files", fallback=N_FILES)
                print(f"measuring the HST noise template from {n_files} cutouts in {hst_dir}")
                template = measure_template(hst_dir.glob("*.fits"), n_files)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_template(cache, template)
                print(f"noise template written to {cache}")
        sigma = section.getfloat("sigma", fallback=None)
        return cls(template, device, sigma, section.get("vgg_on", fallback="noisy"))

    @staticmethod
    def enabled(config) -> bool:
        """[NOISE] present with add_noise = true (the default when the section exists)."""
        if not config.has_section("NOISE"):
            return False
        return config.getboolean("NOISE", "add_noise", fallback=True)

    def target_sigma(self, real_log: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """Sky sigma (B,) of HST targets in log space, outside their segmentation maps.

        A target without a measurable sky (clipped_sky_sigma gives nan: its segmap covers the
        crop, say) gets the batch's median sigma, or sigma_default when it is alone: a nan would
        turn its whole fake, then every weight, to nan without stopping the run.
        """
        sigma = clipped_sky_sigma(to_linear(real_log), seg <= 0)
        finite = torch.isfinite(sigma)
        if not finite.all():
            missing = int((~finite).sum())
            if finite.any():
                fallback = sigma[finite].median()
            else:
                fallback = sigma.new_tensor(self.sigma_default)
            sigma = torch.where(finite, sigma, fallback)
            self.n_fallbacks += missing
            if self.n_fallbacks <= 5:  # say so the first few times, not 300k times
                print(
                    f"noise: no measurable sky sigma for {missing} of {len(sigma)} targets "
                    f"(segmap over the whole crop?); using {fallback.item():.4g}"
                )
            if not torch.isfinite(sigma).all():
                raise RuntimeError("noise: non-finite target sky sigma (is [NOISE] sigma set?)")
        return sigma

    def sample(self, shape, sigma) -> torch.Tensor:
        """Noise of the template's correlation, unit variance scaled by sigma (float or (B,))."""
        white = torch.randn(shape, device=self.device, dtype=torch.float32)
        r = self.kernel.shape[-1] // 2
        shaped = F.conv2d(F.pad(white, (r, r, r, r), mode="circular"), self.kernel)
        if torch.is_tensor(sigma):
            sigma = sigma.to(shaped.dtype).view(-1, *([1] * (shaped.ndim - 1)))
        return shaped * sigma

    def add_noise(self, s_log: torch.Tensor, sigma) -> torch.Tensor:
        """to_linear(s) + n, clipped at 0 as the dataset clips HST, back in log space."""
        noisy = to_linear(s_log) + self.sample(s_log.shape, sigma).to(s_log.dtype)
        return to_log(noisy.clamp(min=0))
