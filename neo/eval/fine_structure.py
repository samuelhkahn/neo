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
    """The HST side of the metric for one cutout: source stamps, fluxes and the noise spectrum."""

    def __init__(self, hst, spectra, thresh=3.0, minarea=10):
        data = np.ascontiguousarray(hst, dtype=np.float64)
        bkg = sep.Background(data)
        sub = data - bkg.back()
        objs, seg = sep.extract(
            sub, thresh, err=bkg.globalrms, minarea=minarea, segmentation_map=True
        )
        self.mask = seg > 0
        self.positions, self.fluxes, self.ffts = [], [], []
        self.noise = np.zeros(spectra.nbins)
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
            if stamp is not None and obj["flux"] > 0:
                self.positions.append((obj["x"], obj["y"]))
                self.fluxes.append(float(obj["flux"]))
                self.ffts.append(spectra.fft(stamp))


class Accumulator:
    """Sums of cross- and auto-power over the sources of many (HST, SR) cutout pairs."""

    def __init__(self, spectra):
        self.s = spectra
        self.cross = np.zeros(spectra.nbins)
        self.p_sr = np.zeros(spectra.nbins)
        self.p_hst = np.zeros(spectra.nbins)
        self.p_noise = np.zeros(spectra.nbins)
        self.n_sources = 0

    def add(self, hst: HstSources, sr):
        """Add one SR cutout (same grid and units as the HST cutout behind hst)."""
        if not hst.positions:
            return
        sr = np.asarray(sr, dtype=np.float64)
        sr = sr - np.median(sr[~hst.mask]) if (~hst.mask).any() else sr
        for (x, y), flux, fh in zip(hst.positions, hst.fluxes, hst.ffts, strict=True):
            fs = self.s.fft(_stamp(sr, x, y)) / flux
            fh = fh / flux
            self.cross += self.s.bin((fs * np.conj(fh)).real)
            self.p_sr += self.s.bin(np.abs(fs) ** 2)
            self.p_hst += self.s.bin(np.abs(fh) ** 2)
            self.p_noise += hst.noise / flux**2
            self.n_sources += 1

    def result(self, band=FINE_BAND, subtract_noise=True):
        noise = self.p_noise if subtract_noise else np.zeros_like(self.p_noise)
        sig = np.clip(self.p_hst - noise, 1e-30, None)
        k = self.s.k
        with np.errstate(divide="ignore", invalid="ignore"):
            transfer = self.cross / sig
            rho = self.cross / np.sqrt(self.p_sr * sig)
            ratio = self.p_sr / sig
        usable = self.p_hst > 2 * self.p_noise  # HST signal power beats its noise
        below = np.where((transfer < 0.5) & (k > 0.3))[0]
        if below.size and below[0] > 0:  # linear interpolation between the bracketing bins
            i = below[0]
            step = (transfer[i - 1] - 0.5) / (transfer[i - 1] - transfer[i])
            k50 = float(k[i - 1] + step * (k[i] - k[i - 1]))
        else:
            k50 = float("nan")
        inband = (k >= band[0]) & (k <= band[1]) & usable

        def band_mean(a):
            return float(np.mean(a[inband])) if inband.any() else float("nan")

        return {
            "k50": k50,
            "theta50_arcsec": 1 / k50 if np.isfinite(k50) and k50 > 0 else float("nan"),
            "T_band": band_mean(transfer),
            "rho_band": band_mean(rho),
            "R_band": band_mean(ratio),
            "n_sources": self.n_sources,
            "k": k.tolist(),
            "T": transfer.tolist(),
            "rho": rho.tolist(),
            "R": ratio.tolist(),
            "usable": usable.tolist(),
        }
