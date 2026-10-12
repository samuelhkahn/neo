"""Training checkpoints that survive a node crash, and resuming from the newest one that loads.

save() writes <name>.tmp, forces it to disk (fsync), renames it over <name> and syncs the
directory. A rename alone is atomic for running processes, but after a node crash on a network
filesystem such as Lustre the rename can survive while the data does not, leaving a truncated
file: lux_diffusion_lrfocus's latest.pt was lost that way at step 600,000 (2026-10-10), and the
requeued job then failed with "PytorchStreamReader failed reading zip archive".

load_latest() returns the newest checkpoint in a directory that loads: latest.pt first (it is
written at least as often as the numbered step_*.pt files, since latest_every divides
save_steps), then step_*.pt from the highest step down. An unreadable file is reported and
skipped, so a requeued job resumes from the last good state instead of failing.
"""

import os
from pathlib import Path

import torch

LATEST = "latest.pt"


def save(state, path) -> None:
    """torch.save(state) to path so that, even after a crash, path is the old or the new file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)  # make the rename itself durable
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def candidates(ckpt_dir) -> list:
    """latest.pt, then step_*.pt newest first (zero-padded step numbers sort by name)."""
    ckpt_dir = Path(ckpt_dir)
    return [ckpt_dir / LATEST, *sorted(ckpt_dir.glob("step_*.pt"), reverse=True)]


def load_latest(ckpt_dir, map_location="cpu"):
    """(path, state) of the newest checkpoint in ckpt_dir that loads, or (None, None)."""
    for path in candidates(ckpt_dir):
        if not path.is_file():
            continue
        try:
            return path, torch.load(path, map_location=map_location, weights_only=True)
        except Exception as exc:  # noqa: BLE001 - a truncated or corrupt file fails in many ways
            print(f"skipping unreadable checkpoint {path}: {type(exc).__name__}: {exc}")
    return None, None
