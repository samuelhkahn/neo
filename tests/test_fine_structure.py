"""The fine-structure metric (neo/eval/fine_structure.py) on synthetic galaxies with known blur."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from neo.eval import fine_structure as fs


def truth_and_hst(seed, n=600):
    """Noise-free sky with compact, clumpy sources, and the same plus HST-like white noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    truth = np.zeros((n, n))
    for _ in range(12):
        y, x = rng.uniform(60, n - 60, 2)
        a, b = rng.uniform(1.0, 4.0, 2)
        truth += rng.uniform(20, 200) * np.exp(-0.5 * (((xx - x) / a) ** 2 + ((yy - y) / b) ** 2))
    return truth, truth + rng.normal(0, 0.3, (n, n))


def measure(pairs, subtract_noise=True):
    spectra = fs.Spectra()
    acc = fs.Accumulator(spectra)
    for hst, sr in pairs:
        acc.add(fs.HstSources(hst, spectra), sr)
    return acc.result(subtract_noise=subtract_noise)


@pytest.fixture(scope="module")
def images():
    return [truth_and_hst(seed) for seed in range(4)]


def test_hst_against_itself_is_exactly_one(images):
    r = measure([(hst, hst) for _, hst in images], subtract_noise=False)
    assert r["n_sources"] > 20
    usable = np.array(r["usable"])
    for key in ("T", "rho", "R"):
        # equal up to the background: HST stamps use SEP's background map, SR its sky median
        np.testing.assert_allclose(np.array(r[key])[usable], 1.0, rtol=1e-3)


def test_noise_free_truth_is_reproduced(images):
    """SR = the noise-free sky: HST's noise is subtracted from its power, so T, rho, R ~ 1."""
    r = measure([(hst, truth) for truth, hst in images])
    assert r["T_band"] == pytest.approx(1.0, abs=0.1)
    assert r["rho_band"] == pytest.approx(1.0, abs=0.1)
    assert r["R_band"] == pytest.approx(1.0, abs=0.15)


def test_blur_lowers_transfer_and_k50(images):
    sharp = measure([(hst, truth) for truth, hst in images])
    blurred = measure([(hst, gaussian_filter(truth, 2.0)) for truth, hst in images])
    more = measure([(hst, gaussian_filter(truth, 4.0)) for truth, hst in images])
    assert more["T_band"] < blurred["T_band"] < 0.8 * sharp["T_band"]
    assert more["k50"] < blurred["k50"]
    assert blurred["R_band"] < sharp["R_band"]


def test_empty_cutout_contributes_nothing():
    spectra = fs.Spectra()
    acc = fs.Accumulator(spectra)
    empty = np.random.default_rng(0).normal(0, 1, (600, 600))
    acc.add(fs.HstSources(empty, spectra), empty)
    assert acc.n_sources == 0
