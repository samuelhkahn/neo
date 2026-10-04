"""Identify a pairs build, and stop training from mixing builds.

neo.preprocess.leakage --apply writes <pairs>/manifest.json, whose build id hashes every pair's
split, name and sky (LSST image + LR pixel origin). `guard` runs before every Lux training job:
it records the build id in the checkpoint directory on the first run and refuses to start when
that directory holds checkpoints from a different build. Otherwise --resume would quietly
continue a model trained on other pairs, e.g. on sky that is now val.

Usage:  python -m neo.preprocess.manifest guard <training .ini>
"""

import argparse
import configparser
import hashlib
import json
import os
import sys
from pathlib import Path

MANIFEST = "manifest.json"
STAMP = "pairs_build.txt"


def build_id(entries) -> str:
    """Hash of (split, name, pair id) for every pair, independent of order."""
    digest = hashlib.sha256()
    for entry in sorted(entries):
        digest.update(("\t".join(entry) + "\n").encode())
    return digest.hexdigest()[:16]


def write_manifest(pairs: Path, entries, **info) -> str:
    build = build_id(entries)
    counts = {}
    for split, _, _ in entries:
        counts[split] = counts.get(split, 0) + 1
    record = {"build": build, "counts": counts, **info}
    (Path(pairs) / MANIFEST).write_text(json.dumps(record, indent=1) + "\n")
    return build


def read_build(pairs: Path):
    path = Path(pairs) / MANIFEST
    return json.loads(path.read_text())["build"] if path.exists() else None


def pairs_root(config) -> Path:
    """The pairs directory a training config reads (parent of train/hr)."""
    return Path(os.path.expandvars(config["DEFAULT"]["hst_path_train"])).parent.parent


def guard(config_path) -> int:
    config = configparser.ConfigParser()
    if not config.read(config_path):
        print(f"guard: cannot read {config_path}")
        return 1
    pairs = pairs_root(config)
    build = read_build(pairs)
    if build is None:
        print(f"guard: {pairs / MANIFEST} is missing; run neo.preprocess.leakage --apply first")
        return 1
    ckpt_dir = Path(os.path.expandvars(config.get("CHECKPOINT", "ckpt_dir", fallback="models")))
    stamp = ckpt_dir / STAMP
    recorded = stamp.read_text().strip() if stamp.exists() else None
    has_checkpoints = ckpt_dir.is_dir() and any(ckpt_dir.glob("*.pt"))
    if recorded == build:
        print(f"guard: {ckpt_dir} matches pairs build {build}")
        return 0
    if has_checkpoints:
        print(
            f"guard: {ckpt_dir} holds checkpoints from pairs build {recorded or 'unknown'}, but "
            f"{pairs} is build {build}. Move that directory aside (or set another ckpt_dir) so a "
            "model trained on other pairs is not resumed."
        )
        return 1
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stamp.write_text(build + "\n")
    print(f"guard: new run in {ckpt_dir} on pairs build {build}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "guard", help="check a training config's checkpoints against its pairs"
    ).add_argument("config")
    args = parser.parse_args(argv)
    return guard(args.config)


if __name__ == "__main__":
    sys.exit(main())
