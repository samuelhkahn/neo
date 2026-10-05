"""NEO paper comparison metrics (Eqs. relative_bias, abs_bias, sim_diff) and their summaries.

Relative bias (M_X - M_HST) / M_HST for R_e, FWHM and C75/25; absolute bias M_X - M_HST for the axis
ratio q = b/a = 1 / elongation; orientation similarity S = 1 - |cos(theta_X - theta_HST)|. LR sizes
are measured in coarse pixels and multiplied by the scale factor (6) before comparison, as in the
paper's gain plots. Table 4 reports the median over the test set.

The code behind the paper's Table 4 measured q differently from its text: q was photutils
ellipticity (1 - b/a), and the printed q entry is the 68th percentile of |B - median(B)| of the
absolute bias B, with +/- the std (the paper's Table-4 statistics notebook).
ellipticity_bias and paper_q_statistic reproduce that; our q bias equals -median(B).
"""

import numpy as np

# name -> (catalog column, kind, scale LR values by the pixel factor)
PARAMETERS = {
    "R_e": ("half_light_radius", "relative", True),
    "FWHM": ("fwhm", "relative", True),
    "q": ("q", "absolute", False),
    "C75/25": ("flux_concentration_75_25", "relative", False),
    "orientation": ("orientation", "similarity", False),
    "flux": ("segment_flux", "relative", False),
}


def axis_ratio(tbl) -> np.ndarray:
    return 1.0 / np.asarray(tbl["elongation"], dtype=float)


def orientation_similarity(theta_x_deg, theta_hst_deg) -> np.ndarray:
    """1 - |cosine similarity| of the unit vectors at the two position angles."""
    return 1.0 - np.abs(np.cos(np.radians(np.asarray(theta_x_deg) - np.asarray(theta_hst_deg))))


def values(tbl, column: str, scale: float = 1.0) -> np.ndarray:
    if column == "q":
        return axis_ratio(tbl)
    return np.asarray(tbl[column], dtype=float) * scale


def per_source(hst_tbl, x_tbl, factor: float = 1.0) -> dict:
    """Per-source metric values for one image set X against HST (factor = LR->HR pixel scale)."""
    if not np.array_equal(np.asarray(hst_tbl["label"]), np.asarray(x_tbl["label"])):
        raise ValueError("catalog rows are not aligned by segment label")
    out = {}
    for name, (column, kind, scaled) in PARAMETERS.items():
        truth = values(hst_tbl, column)
        meas = values(x_tbl, column, factor if scaled else 1.0)
        if kind == "relative":
            out[name] = (meas - truth) / truth
        elif kind == "absolute":
            out[name] = meas - truth
        else:
            out[name] = orientation_similarity(meas, truth)
    return out


def errors(hst_tbl, x_tbl, factor: float = 1.0) -> dict:
    """|X - HST| per source in each parameter's native units (for paired model comparisons)."""
    out = {}
    for name, (column, kind, scaled) in PARAMETERS.items():
        truth = values(hst_tbl, column)
        meas = values(x_tbl, column, factor if scaled else 1.0)
        out[name] = (
            orientation_similarity(meas, truth) if kind == "similarity" else np.abs(meas - truth)
        )
    return out


def ellipticity_bias(hst_tbl, x_tbl) -> np.ndarray:
    """Absolute bias of photutils ellipticity (1 - b/a), X - HST: the paper code's q bias."""
    if not np.array_equal(np.asarray(hst_tbl["label"]), np.asarray(x_tbl["label"])):
        raise ValueError("catalog rows are not aligned by segment label")
    return np.asarray(x_tbl["ellipticity"], dtype=float) - np.asarray(
        hst_tbl["ellipticity"], dtype=float
    )


def paper_q_statistic(bias: np.ndarray) -> dict:
    """The paper's Table 4 q entry: 68th percentile of |B - median(B)|, with median and std of B."""
    b = np.asarray(bias, dtype=float)
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return {"n": 0}
    med = np.median(b)
    return {
        "n": len(b),
        "q68": float(np.quantile(np.abs(b - med), 0.68)),
        "median": float(med),
        "std": float(b.std()),
    }


def summarize(samples: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Median (paper's Table 4 statistic) with NMAD and a bootstrap 95% CI, plus mean and std."""
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    med = np.median(x)
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return {
        "n": len(x),
        "median": med,
        "nmad": 1.4826 * np.median(np.abs(x - med)),
        "median_ci_lo": np.percentile(boot, 2.5),
        "median_ci_hi": np.percentile(boot, 97.5),
        "mean": x.mean(),
        "std": x.std(),
    }


def gain(err_baseline: np.ndarray, err_model: np.ndarray) -> dict:
    """Paper's performance gain log10(|HST - baseline| / |HST - model|): share > 0 and mean."""
    g = np.log10(np.asarray(err_baseline, dtype=float) / np.asarray(err_model, dtype=float))
    g = g[np.isfinite(g)]
    if len(g) == 0:
        return {"n": 0}
    return {"n": len(g), "frac_improved": float(np.mean(g > 0)), "mean_log_gain": float(g.mean())}
