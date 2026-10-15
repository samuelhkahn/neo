"""How much of HST's fine structure an SR image reproduces, per spatial frequency, inside sources.

For each bright HST source (SEP detection on the HST cutout), a 64 px (2.1") stamp centred on it is
cut from both HST and SR, Hann-windowed and Fourier transformed, and divided by the source's HST
flux so every source weighs the same. Summed over sources and images, per annulus in spatial
frequency k (cycles / arcsec):

    T(k)   = sum Re(SR HST*) / sum (|HST|^2 - N)                 transfer: true structure kept
    rho(k) = sum Re(SR HST*) / sqrt(sum |SR|^2 sum (|HST|^2 - N))  coherence: phase agreement
    R(k)   = sum |SR|^2 / sum (|HST|^2 - N)                       power ratio: true or invented

N is HST's own noise power, measured on source-free stamps of the same cutout; the SR image is made
from the LR image and carries none of HST's noise, so the cross-power is unbiased by it. A model
that blurs has T, rho and R falling at high k; one that invents texture keeps R near 1 while T and
rho fall. Summaries: k50, where T drops to 0.5 (1 / k50 is an effective resolution in arcsec), and
band means of T, rho and R over FINE_BAND, between the LR and HST resolutions, using only the bins
where HST's signal power exceeds twice its noise.

Two kinds of source are counted instead of summed, because in linear flux one of them outweighs
every real source in the pooled sums:
  saturated  the SR stamp holds a pixel at or above `saturation` (sweep.py passes the flux of the
             generator's tanh ceiling: "the generator saturated here"); saturated_fraction
  blown      the SR stamp is far brighter than the HST stamp without reaching the ceiling: its
             peak exceeds PEAK_MAX times the HST stamp's peak, or its fine-band power exceeds R_MAX
             times the HST stamp's noise-subtracted fine-band power (floored at the noise power,
             so a smooth source whose HST fine-band power is all noise cannot trip the rule on a
             near-zero denominator); blown_fraction. Both tests use only that source's own stamps.
             The peak rule names the failure directly (a blow-up is a brightness error: one source
             rendered 10x too bright has 100x the power); the power rule catches the same failure
             in the metric's own currency when it comes as ringing or texture rather than one
             bright peak. A model that invents texture at the level of HST's own fine structure
             (R ~ 1-3) is untouched by either, so R_band still measures invented texture.
Sources inside an exclusion mask (HstSources exclude=..., e.g. the Gaia star masks of
neo.eval.stars: stars moved between the epochs) are skipped before any of this and counted as
n_excluded.

Error bars: the per-source binned spectra are kept, and result() resamples sources with
replacement (N_BOOT draws) to give a 68% interval (16th-84th percentiles) for T_band, rho_band,
R_band and k50 (*_lo, *_hi): the spread the metric would show over other draws of the same kind
of sources, not over other dropout draws of the generator.

Ported from the 2026-10-12 fine-structure analysis (neo_outputs/scratch/curves_fine_structure).
HST measurements depend only on the HST cutout, so HstSources computes them once and every SR image
(e.g. every checkpoint of a run) reuses them.
"""

import numpy as np
import sep
from scipy.ndimage import gaussian_filter

HR_ARCSEC = 0.2 / 6
STAMP = 64
FINE_BAND = (1.5, 8.0)  # cycles/arcsec: ~0.67" (ground-based PSF scale) to ~0.125" (HST)
MIN_SNR = 20  # white-noise S/N of the HST detection for a source to be used
R_MAX = 10.0  # blown: SR fine-band power over this many times HST's (noise-subtracted) power
PEAK_MAX = 10.0  # blown: SR stamp peak over this many times the HST stamp peak
N_BOOT = 200
BOOT_PERCENTILES = (16, 84)  # a 68% interval


def hann2d(n):
    w = np.hanning(n)
    return np.outer(w, w)


def kgrid(n, pix):
    f = np.fft.fftfreq(n, d=pix)
    return np.hypot(*np.meshgrid(f, f, indexing="ij"))


def _stamp(img, x, y, n=STAMP):
    h = n // 2
    xi, yi = int(round(x)), int(round(y))
    if yi - h < 0 or xi - h < 0 or yi + h > img.shape[0] or xi + h > img.shape[1]:
        return None
    return img[yi - h : yi + h, xi - h : xi + h]


def _empty_positions(mask, n=STAMP, tries=400, want=20, seed=0):
    rng = np.random.default_rng(seed)
    grown = gaussian_filter(mask.astype(float), 4) > 0.01
    h = n // 2
    pos = []
    for _ in range(tries):
        x, y = rng.integers(h, mask.shape[1] - h), rng.integers(h, mask.shape[0] - h)
        if not grown[y - h : y + h, x - h : x + h].any():
            pos.append((x, y))
            if len(pos) >= want:
                break
    return pos


class Spectra:
    """Annulus binning of 2-D power spectra of Hann-windowed STAMP x STAMP stamps."""

    def __init__(self, n=STAMP, pix=HR_ARCSEC, nbins=24):
        self.window = hann2d(n)
        self.edges = np.linspace(0, 1 / (2 * pix), nbins + 1)
        self.k = 0.5 * (self.edges[1:] + self.edges[:-1])
        idx = np.digitize(kgrid(n, pix).ravel(), self.edges) - 1
        self.ok = (idx >= 0) & (idx < nbins)
        self.idx = idx[self.ok]
        self.nbins = nbins

    def bin(self, a):
        return np.bincount(self.idx, weights=a.ravel()[self.ok], minlength=self.nbins)

    def fft(self, stamp):
        return np.fft.fft2(stamp * self.window)


class HstSources:
    """The HST side of the metric for one cutout: source stamps, fluxes, peaks and the noise
    spectrum. Sources whose position falls inside `exclude` (a boolean image) are skipped and
    counted in n_excluded."""

    def __init__(self, hst, spectra, thresh=3.0, minarea=10, exclude=None):
        data = np.ascontiguousarray(hst, dtype=np.float64)
        bkg = sep.Background(data)
        sub = data - bkg.back()
        objs, seg = sep.extract(
            sub, thresh, err=bkg.globalrms, minarea=minarea, segmentation_map=True
        )
        self.mask = seg > 0
        self.positions, self.fluxes, self.ffts, self.peaks = [], [], [], []
        self.noise = np.zeros(spectra.nbins)
        self.n_excluded = 0
        if len(objs) == 0:
            return
        empty = [_stamp(sub, x, y) for x, y in _empty_positions(self.mask)]
        empty = [s for s in empty if s is not None]
        if empty:
            self.noise = np.mean(
                [spectra.bin(np.abs(spectra.fft(s - np.median(s))) ** 2) for s in empty], axis=0
            )
        snr = objs["flux"] / (bkg.globalrms * np.sqrt(objs["npix"]))
        for obj in objs[snr > MIN_SNR]:
            stamp = _stamp(sub, obj["x"], obj["y"])
            if stamp is None or obj["flux"] <= 0:
                continue
            if exclude is not None and exclude[int(round(obj["y"])), int(round(obj["x"]))]:
                self.n_excluded += 1
                continue
            self.positions.append((obj["x"], obj["y"]))
            self.fluxes.append(float(obj["flux"]))
            self.ffts.append(spectra.fft(stamp))
            self.peaks.append(float(stamp.max()))


def _band_mean(a, inband):
    """Mean of `a` over the bins `inband` marks, along the last axis; NaN where none is marked.

    A 1-D input (the point estimate) is averaged as np.mean(a[inband]), exactly the arithmetic
    the metric used before the bootstrap existed, so a rescored checkpoint reproduces its old
    rho_band / R_band to the last digit; the padded sum below rounds differently by an ulp."""
    if a.ndim == 1:
        return np.mean(a[inband]) if inband.any() else np.nan
    count = inband.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, np.where(inband, a, 0.0).sum(axis=-1) / count, np.nan)


def _k50(transfer, k):
    """k where T first drops below 0.5 (beyond 0.3 cycles/arcsec), by linear interpolation
    between the bracketing bins, along the last axis; NaN where it never does, or does so in the
    first bin."""
    below = (transfer < 0.5) & (k > 0.3)
    i = np.argmax(below, axis=-1)
    ok = below.any(axis=-1) & (i > 0)
    i = np.where(ok, i, 1)
    prev = np.take_along_axis(transfer, (i - 1)[..., None], axis=-1)[..., 0]
    this = np.take_along_axis(transfer, i[..., None], axis=-1)[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        step = (prev - 0.5) / (prev - this)
        k50 = k[i - 1] + step * (k[i] - k[i - 1])
    return np.where(ok, k50, np.nan)


def _curves(cross, p_sr, p_hst, p_noise, k, band, subtract_noise=True):
    """T, rho, R per bin, the usable bins, k50 and the band means, for sums with any leading
    (bootstrap) axes."""
    noise = p_noise if subtract_noise else np.zeros_like(p_noise)
    sig = np.clip(p_hst - noise, 1e-30, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        transfer = cross / sig
        rho = cross / np.sqrt(p_sr * sig)
        ratio = p_sr / sig
    usable = p_hst > 2 * p_noise  # HST signal power beats its noise
    inband = (k >= band[0]) & (k <= band[1]) & usable
    return {
        "T": transfer,
        "rho": rho,
        "R": ratio,
        "usable": usable,
        "k50": _k50(transfer, k),
        "T_band": _band_mean(transfer, inband),
        "rho_band": _band_mean(rho, inband),
        "R_band": _band_mean(ratio, inband),
    }


class Accumulator:
    """Sums of cross- and auto-power over the sources of many (HST, SR) cutout pairs, plus the
    per-source binned spectra behind them (for the bootstrap)."""

    def __init__(self, spectra):
        self.s = spectra
        self.cross = np.zeros(spectra.nbins)
        self.p_sr = np.zeros(spectra.nbins)
        self.p_hst = np.zeros(spectra.nbins)
        self.p_noise = np.zeros(spectra.nbins)
        self.n_sources = 0
        self.n_saturated = 0
        self.n_blown = 0
        self.n_excluded = 0
        self.band = (spectra.k >= FINE_BAND[0]) & (spectra.k <= FINE_BAND[1])
        self.per_source = []  # (4, nbins) per kept source: cross, |SR|^2, |HST|^2, noise

    def blown(self, peak_sr, peak_hst, p_sr, p_hst, p_noise, r_max=R_MAX, peak_max=PEAK_MAX):
        """The per-source blow-up rule (module docstring), on this source's own binned spectra
        (already divided by its HST flux) and stamp peaks."""
        hst_band = max((p_hst - p_noise)[self.band].sum(), p_noise[self.band].sum())
        if p_sr[self.band].sum() > r_max * hst_band:
            return True
        return peak_hst > 0 and peak_sr > peak_max * peak_hst

    def add(self, hst: HstSources, sr, saturation=np.inf, r_max=R_MAX, peak_max=PEAK_MAX):
        """Add one SR cutout (same grid and units as the HST cutout behind hst); sources whose SR
        stamp reaches `saturation` are counted in n_saturated, blown ones in n_blown, and both are
        skipped."""
        self.n_excluded += hst.n_excluded
        if not hst.positions:
            return
        sr = np.asarray(sr, dtype=np.float64)
        sky = np.median(sr[~hst.mask]) if (~hst.mask).any() else 0.0
        sources = zip(hst.positions, hst.fluxes, hst.ffts, hst.peaks, strict=True)
        for (x, y), flux, fh, peak_hst in sources:
            stamp = _stamp(sr, x, y)
            if stamp.max() >= saturation:
                self.n_saturated += 1
                continue
            centred = stamp - sky
            fs = self.s.fft(centred) / flux
            fh = fh / flux
            cross = self.s.bin((fs * np.conj(fh)).real)
            p_sr = self.s.bin(np.abs(fs) ** 2)
            p_hst = self.s.bin(np.abs(fh) ** 2)
            p_noise = hst.noise / flux**2
            if self.blown(centred.max(), peak_hst, p_sr, p_hst, p_noise, r_max, peak_max):
                self.n_blown += 1
                continue
            self.cross += cross
            self.p_sr += p_sr
            self.p_hst += p_hst
            self.p_noise += p_noise
            self.per_source.append(np.stack([cross, p_sr, p_hst, p_noise]))
            self.n_sources += 1

    def bootstrap(self, band=FINE_BAND, subtract_noise=True, n_boot=N_BOOT, seed=0):
        """(lo, hi) per summary (T_band, rho_band, R_band, k50) from resampling sources with
        replacement; NaN with fewer than two sources or n_boot 0."""
        keys = ("T_band", "rho_band", "R_band", "k50")
        if n_boot < 1 or len(self.per_source) < 2:
            return {key: (np.nan, np.nan) for key in keys}
        per_source = np.array(self.per_source)  # (N, 4, nbins)
        n = len(per_source)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n, size=(n_boot, n))
        weights = np.zeros((n_boot, n))
        np.add.at(weights, (np.arange(n_boot)[:, None], draws), 1)
        sums = np.einsum("bn,nkj->bkj", weights, per_source)  # (n_boot, 4, nbins)
        curves = _curves(
            sums[:, 0], sums[:, 1], sums[:, 2], sums[:, 3], self.s.k, band, subtract_noise
        )
        out = {}
        for key in keys:
            values = curves[key][np.isfinite(curves[key])]
            if len(values):
                lo, hi = np.percentile(values, BOOT_PERCENTILES)
                out[key] = (float(lo), float(hi))
            else:
                out[key] = (np.nan, np.nan)
        return out

    def result(self, band=FINE_BAND, subtract_noise=True, n_boot=N_BOOT, seed=0):
        curves = _curves(
            self.cross, self.p_sr, self.p_hst, self.p_noise, self.s.k, band, subtract_noise
        )
        k50 = float(curves["k50"])
        boot = self.bootstrap(band, subtract_noise, n_boot, seed)
        total = max(self.n_sources + self.n_saturated + self.n_blown, 1)
        return {
            "k50": k50,
            "theta50_arcsec": 1 / k50 if np.isfinite(k50) and k50 > 0 else float("nan"),
            "T_band": float(curves["T_band"]),
            "rho_band": float(curves["rho_band"]),
            "R_band": float(curves["R_band"]),
            "n_sources": self.n_sources,
            "n_saturated": self.n_saturated,
            "saturated_fraction": self.n_saturated / total,
            "n_blown": self.n_blown,
            "blown_fraction": self.n_blown / total,
            "n_excluded": self.n_excluded,
            "n_boot": n_boot if len(self.per_source) >= 2 else 0,
            **{f"{key}_lo": lo for key, (lo, hi) in boot.items()},
            **{f"{key}_hi": hi for key, (lo, hi) in boot.items()},
            "k": self.s.k.tolist(),
            "T": curves["T"].tolist(),
            "rho": curves["rho"].tolist(),
            "R": curves["R"].tolist(),
            "usable": curves["usable"].tolist(),
        }
