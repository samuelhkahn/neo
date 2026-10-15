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


def measure(pairs, subtract_noise=True, **add_options):
    spectra = fs.Spectra()
    acc = fs.Accumulator(spectra)
    for hst, sr in pairs:
        acc.add(fs.HstSources(hst, spectra), sr, **add_options)
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
    assert r["n_blown"] == 0 and r["blown_fraction"] == 0 and r["n_excluded"] == 0


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
    r = acc.result()
    assert np.isnan(r["T_band"]) and np.isnan(r["T_band_lo"]) and r["n_boot"] == 0


def test_saturated_sources_are_counted_and_skipped(images):
    """One source saturated at the generator's ceiling would dominate every pooled spectrum."""
    truth, hst = images[0]
    spectra = fs.Spectra()
    srcs = fs.HstSources(hst, spectra)
    x, y = srcs.positions[0]
    blown = truth.copy()
    blown[int(y) - 2 : int(y) + 3, int(x) - 2 : int(x) + 3] = 1e6
    acc = fs.Accumulator(spectra)
    acc.add(srcs, blown, saturation=1e5)
    r = acc.result()
    assert acc.n_saturated == 1 and r["n_sources"] == len(srcs.positions) - 1
    assert r["saturated_fraction"] == pytest.approx(1 / len(srcs.positions))
    assert r["n_blown"] == 0  # saturation is checked first: not counted twice
    assert r["T_band"] == pytest.approx(1.0, abs=0.1) and r["R_band"] < 1.5
    # without either guard the one source wrecks the pooled power ratio
    unguarded = fs.Accumulator(spectra)
    unguarded.add(srcs, blown, r_max=np.inf, peak_max=np.inf)
    assert unguarded.result()["R_band"] > 100


def test_a_blown_up_source_is_counted_and_skipped(images):
    """A source rendered far too bright (but under the ceiling) is skipped by the per-source
    rule, and the pooled T of an otherwise perfect SR stays ~1; without the rule it does not."""
    truth, hst = images[0]
    spectra = fs.Spectra()
    srcs = fs.HstSources(hst, spectra)
    x, y = srcs.positions[0]
    blown = truth.copy()
    yy, xx = np.mgrid[:600, :600]
    near = (xx - x) ** 2 + (yy - y) ** 2 <= 10**2
    blown[near] *= 40  # 40 x brighter: peak 40 x HST's, fine-band power 1600 x
    assert blown.max() < 1e5
    acc = fs.Accumulator(spectra)
    acc.add(srcs, blown, saturation=1e5)
    r = acc.result()
    assert r["n_blown"] == 1 and r["n_saturated"] == 0
    assert r["n_sources"] == len(srcs.positions) - 1
    assert r["blown_fraction"] == pytest.approx(1 / len(srcs.positions))
    assert r["T_band"] == pytest.approx(1.0, abs=0.1) and r["R_band"] < 1.5
    unguarded = fs.Accumulator(spectra)
    unguarded.add(srcs, blown, saturation=1e5, r_max=np.inf, peak_max=np.inf)
    assert unguarded.result()["R_band"] > 10 and unguarded.result()["n_blown"] == 0
    # each rule alone catches it: 40 x the peak, and 1600 x the power
    for options in ({"r_max": np.inf}, {"peak_max": np.inf}):
        one = fs.Accumulator(spectra)
        one.add(srcs, blown, saturation=1e5, **options)
        assert one.n_blown == 1, options
    # a source only 3 x too bright (a flux bias, not a blow-up) passes both rules
    mild = truth.copy()
    mild[near] *= 3
    ok = fs.Accumulator(spectra)
    ok.add(srcs, mild, saturation=1e5)
    assert ok.n_blown == 0 and ok.n_sources == len(srcs.positions)


def test_the_blow_up_rule_is_decided_per_source(images):
    """The rule compares each SR stamp with its own HST stamp: a bright source next to faint
    ones is not blown, and the per-source test passes a uniformly scaled copy of HST."""
    truth, hst = images[1]
    spectra = fs.Spectra()
    srcs = fs.HstSources(hst, spectra)
    acc = fs.Accumulator(spectra)
    acc.add(srcs, 2 * truth)  # every source 2 x: under both thresholds
    assert acc.n_blown == 0 and acc.n_sources == len(srcs.positions)
    assert acc.result()["R_band"] == pytest.approx(4.0, rel=0.3)


def test_bootstrap_interval_contains_the_estimate_and_shrinks_with_more_sources(images):
    # blurred and re-noised, so the sources differ in how much fine structure they keep
    rng = np.random.default_rng(7)
    blurred = [
        (hst, gaussian_filter(truth, 1.5) + rng.normal(0, 0.5, hst.shape)) for truth, hst in images
    ]
    few = measure(blurred[:1])
    many = measure(blurred)
    assert many["n_sources"] > 2 * few["n_sources"] >= 4
    for key in ("T_band", "rho_band", "R_band", "k50"):
        for r in (few, many):
            assert r[f"{key}_lo"] <= r[key] <= r[f"{key}_hi"], key
            assert r["n_boot"] == fs.N_BOOT
        assert many[f"{key}_hi"] - many[f"{key}_lo"] < few[f"{key}_hi"] - few[f"{key}_lo"], key
    # the interval is a bootstrap over sources: the same draws give the same interval
    again = measure(blurred)
    assert (again["T_band_lo"], again["T_band_hi"]) == (many["T_band_lo"], many["T_band_hi"])
    # the sums behind the point estimate are the sums of the stored per-source spectra
    spectra = fs.Spectra()
    acc = fs.Accumulator(spectra)
    for hst, sr in blurred:
        acc.add(fs.HstSources(hst, spectra), sr)
    per_source = np.array(acc.per_source)
    np.testing.assert_allclose(per_source[:, 0].sum(0), acc.cross)
    np.testing.assert_allclose(per_source[:, 1].sum(0), acc.p_sr)
    assert len(per_source) == acc.n_sources
    # the point estimate is the plain mean over the usable band bins, bit for bit as the metric
    # computed it before the bootstrap existed (a rescored checkpoint reproduces its old value)
    r = acc.result()
    usable = np.array(r["usable"])
    k = np.array(r["k"])
    inband = (k >= fs.FINE_BAND[0]) & (k <= fs.FINE_BAND[1]) & usable
    for key in ("T", "rho", "R"):
        assert r[f"{key}_band"] == float(np.mean(np.array(r[key])[inband])), key


def test_result_keeps_the_old_keys():
    spectra = fs.Spectra()
    r = fs.Accumulator(spectra).result()
    for key in ("k50", "theta50_arcsec", "T_band", "rho_band", "R_band", "n_sources"):
        assert key in r
    for key in ("n_saturated", "saturated_fraction", "k", "T", "rho", "R", "usable"):
        assert key in r
    assert len(r["k"]) == spectra.nbins
