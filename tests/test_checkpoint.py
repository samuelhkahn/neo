"""Crash-safe checkpoints and resuming from the newest readable one (neo/checkpoint.py)."""

import torch

from neo import checkpoint


def state(step):
    return {"step": step, "w": torch.full((3,), float(step))}


def test_save_round_trip_leaves_no_temp_file(tmp_path):
    checkpoint.save(state(7), tmp_path / "latest.pt")
    assert [p.name for p in tmp_path.iterdir()] == ["latest.pt"]
    path, loaded = checkpoint.load_latest(tmp_path)
    assert path == tmp_path / "latest.pt" and loaded["step"] == 7
    assert torch.equal(loaded["w"], state(7)["w"])


def test_latest_is_preferred_over_numbered_checkpoints(tmp_path):
    checkpoint.save(state(25000), tmp_path / "step_00025000.pt")
    checkpoint.save(state(26000), tmp_path / "latest.pt")
    assert checkpoint.load_latest(tmp_path)[1]["step"] == 26000


def test_corrupt_files_fall_back_to_the_newest_readable(tmp_path, capsys):
    """As on Lux: latest.pt and the newest step file truncated by a node crash."""
    for step in (25000, 100000, 575000):
        checkpoint.save(state(step), tmp_path / f"step_{step:08d}.pt")
    good = (tmp_path / "step_00575000.pt").read_bytes()
    (tmp_path / "step_00600000.pt").write_bytes(good[: len(good) // 2])  # truncated
    (tmp_path / "latest.pt").write_bytes(b"")  # empty
    path, loaded = checkpoint.load_latest(tmp_path)
    assert path.name == "step_00575000.pt" and loaded["step"] == 575000
    out = capsys.readouterr().out
    assert "latest.pt" in out and "step_00600000.pt" in out


def test_nothing_to_resume(tmp_path):
    assert checkpoint.load_latest(tmp_path) == (None, None)
    (tmp_path / "latest.pt").write_bytes(b"not a checkpoint")
    assert checkpoint.load_latest(tmp_path) == (None, None)
