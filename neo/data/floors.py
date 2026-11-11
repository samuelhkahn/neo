"""Fixed sky floors for the training clip, so a model can learn a sky that averages zero.

The paper's preprocessing clips every HST target and LR input at 0 before the ds9 log scale
(neo.data.dataset: clip(x, 0, p99.999), then log10(1000 x + 1) / log10(1001) - 1). The clipped
target sky is max(0, noise), whose mean is about +0.4 sigma, and the generator's tanh output at -1
is exactly 0 flux, so it can never predict a negative pixel: the SR sky sits on a positive floor
(sky_floor_check: COSMOS SR sky mean 0.138 nJy per HR px, clipped-HST target 0.151, true sky
-0.008; GOODS-S SR 0.135, so the same floor out of sample).

[DATASET] hr_floor and lr_floor (stored units, <= 0; default 0, the paper's clip) move that clip
down to [floor, p99.999], and the log scale becomes its signed (odd) extension:
  s(x) = sign(x) log10(1 + a|x|) / log10(1 + a),   a = 1000 as before
  v    = -1 + (s(x) - s(floor)) / (1 - s(floor))
so the floor maps to -1 (the generator's tanh minimum) and x = 1 to 0, as in the paper's scale.
The ds9 scale is there to control dynamic range: its knee at x ~ 1/a puts sky noise and faint
sources on the steep part of the curve and compresses bright cores. The signed form keeps that
exactly for x >= 0 (only rescaled by 1 / (1 - s(floor)): 0.74 for HR at -0.01, 0.57 for LR at
-0.2) and mirrors it for x < 0, so the sky stays where the curve is steepest (x = 0) and is
smooth there (s is C1 at 0). Shifting x by the floor before the plain log instead would put the
sky at a |floor| = 10 (HR) or 200 (LR) times the knee, where the curve is flat: 3-sigma sources
would lose 4x (HR) to 10x (LR) of their contrast in log space.
A floor must be one fixed value per survey, not measured per image: inference has no target to
measure it from, and a per-image shift would make the output's zero point unknowable.

Every checkpoint records its floors; check() refuses to evaluate or resume one with a config whose
floors differ (a checkpoint without the record was trained with the paper's clip, floors 0).
"""

import math

import numpy as np

KEYS = ("hr_floor", "lr_floor")
A = 1000  # the paper's ds9 scale parameter (neo.eval.postprocess.LOG_SCALE_A)
LOG_A1 = math.log10(A + 1)


def _signed_log(x, lib):
    return lib.sign(x) * lib.log10(1 + A * lib.abs(x)) / LOG_A1


def s_floor(floor: float) -> float:
    """s(floor) of the signed log (module docstring); 0 for a floor of 0."""
    return -math.log10(1 - A * floor) / LOG_A1


def scale(x, floor: float):
    """Stored units (already clipped at floor) -> log space, floor -> -1 (numpy or torch)."""
    lib = np if isinstance(x, np.ndarray) else _torch()
    sf = s_floor(floor)
    return -1 + (_signed_log(x, lib) - sf) / (1 - sf)


def unscale(v, floor: float):
    """Log space -> stored units; the inverse of scale (numpy or torch)."""
    lib = np if isinstance(v, np.ndarray) else _torch()
    sf = s_floor(floor)
    s = sf + (v + 1) * (1 - sf)
    return lib.sign(s) * lib.expm1(lib.abs(s) * math.log(A + 1)) / A


def _torch():
    import torch

    return torch


def floors(config) -> dict:
    """{"hr_floor": float, "lr_floor": float} from [DATASET] (default 0.0 each), each <= 0."""
    out = {}
    for key in KEYS:
        value = float(config.get("DATASET", key, fallback="0") or 0)
        if not value <= 0:  # also refuses nan
            raise ValueError(f"[DATASET] {key} must be <= 0 (stored units), not {value}")
        out[key] = value
    return out


def any_floor(values: dict) -> bool:
    return any(values[key] != 0 for key in KEYS)


def check(state, config, source="checkpoint") -> None:
    """Refuse a checkpoint trained with other floors than the config's."""
    want = floors(config)
    saved = {key: float(state.get(key, 0.0)) for key in KEYS}
    if saved != want:
        raise ValueError(
            f"{source} was trained with sky floors {saved}, but the config has {want} "
            "([DATASET] hr_floor / lr_floor): use the run's own config"
        )


def unsupported_with(values: dict, config) -> None:
    """The [CONSISTENCY] and [NOISE] arms convert log space to flux assuming floors of 0."""
    if not any_floor(values):
        return
    for section in ("CONSISTENCY", "NOISE"):
        if config.has_section(section):
            raise ValueError(
                f"[DATASET] sky floors are not supported with [{section}] (neo/consistency.py and "
                "neo/noise.py map log space to flux with the paper's zero floor)"
            )
