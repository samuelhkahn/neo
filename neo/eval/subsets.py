"""Which val pairs are used for what, and which sky a cutout shows.

Val is split deterministically into a select subset (20%: checkpoint choice, training-time
validation curves) and a report subset (80%: the numbers that get reported, touched only once a
model is chosen). With <val>/groups.csv (written by neo.preprocess.leakage) whole groups of
overlapping cutouts are assigned together; otherwise each pair is assigned by its name.
"""

import csv
import hashlib
from pathlib import Path

SUBSETS = ("select", "report", "all")


def load_groups(split: Path) -> dict:
    path = Path(split) / "groups.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {row["name"]: row["group"] for row in csv.DictReader(f)}


def in_subset(name: str, subset: str, groups=None) -> bool:
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, not {subset!r}")
    if subset == "all":
        return True
    key = (groups or {}).get(name, name)
    selected = int(hashlib.md5(key.encode()).hexdigest(), 16) % 5 == 0
    return selected if subset == "select" else not selected


def pair_id(header) -> str:
    """The sky a pair shows: its LSST image and LR pixel origin (pairs.py cards), else its WCS."""
    if "LRFILE" in header:
        return f"{header['LRFILE']}:{header['LRX0']}:{header['LRY0']}"
    return ",".join(f"{header.get(k, '')}" for k in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"))
