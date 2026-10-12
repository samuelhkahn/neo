"""The fixed-set checkpoint sweep (neo/eval/sweep.py) on tiny local checkpoints."""

import configparser
import csv

import pytest
import torch
from test_eval_units import write_pair

from neo.eval import sweep
from neo.models.generator import Pix2PixGenerator
from neo.models.patchgan import PatchGAN


class CheapVGG:
    """Stands in for the VGG perceptual loss so the test needs no pretrained weights."""

    def __init__(self, device, weights):
        pass

    def __call__(self, x, y):
        return ((x - y) ** 2).mean()


def write_run(tmp_path, steps):
    split = write_pair(tmp_path / "val", "paper")
    write_pair(tmp_path / "val", "paper", name="deep_coadd_test_val_00001.fits")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for step in steps:
        torch.manual_seed(step)
        torch.save(
            {
                "step": step,
                "gen": Pix2PixGenerator(1, 1).state_dict(),
                "patch_gan": PatchGAN(2).state_dict(),
            },
            ckpt / f"step_{step:08d}.pt",
        )
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "DEFAULT": {"hst_path_val": str(split / "hr"), "hsc_path_val": str(split / "lr")},
            "CHECKPOINT": {"ckpt_dir": str(ckpt)},
            "VGG_LOSS_WEIGHTS": {"vgg_loss_weights": "[0.0, 1.0, 1.0, 0.0, 0.0]"},
            "IDENTIFIER": {"identifier": '"test_run"'},
            "DATASET": {"hsc_hr_8bit": "wrap"},
        }
    )
    path = tmp_path / "run.ini"
    with open(path, "w") as f:
        config.write(f)
    return path


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_sweep_scores_checkpoints_and_averages(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "VGGLoss", CheapVGG)
    config = write_run(tmp_path, [25000, 50000, 75000])
    out = tmp_path / "out"
    args = ["--config", str(config), "--out", str(out), "--device", "cpu", "--no-comet"]
    sweep.main([*args, "--average", "50000:75000"])
    summary = rows(out / "sweep.csv")
    assert [r["checkpoint"] for r in summary] == [
        "step_00025000",
        "step_00050000",
        "step_00075000",
        "avg_50000-75000",
    ]
    assert all(int(r["n_images"]) == 2 for r in summary)
    assert len(rows(out / "per_image.csv")) == 8
    assert (out / "sweep.png").exists() and (out / "fine_structure.json").exists()

    # a rerun scores only what is new, and refuses other settings in the same directory
    torch.save(
        torch.load(out.parent / "ckpt" / "step_00075000.pt"),
        out.parent / "ckpt" / "step_00100000.pt",
    )
    sweep.main(args)
    summary = rows(out / "sweep.csv")
    assert summary[-1]["checkpoint"] == "step_00100000"
    # step 100000 is a copy of 75000: same images, same dropout masks, same numbers
    by_name = {r["checkpoint"]: r for r in summary}
    for key in ("l1", "segmap_l1", "vgg", "adversarial", "discriminator"):
        assert float(by_name["step_00100000"][key]) == pytest.approx(
            float(by_name["step_00075000"][key]), rel=1e-6
        )
    with pytest.raises(SystemExit, match="other settings"):
        sweep.main([*args, "--seed", "1"])


def test_average_of_identical_checkpoints_is_the_checkpoint(tmp_path):
    state = {"gen": Pix2PixGenerator(1, 1).state_dict(), "patch_gan": PatchGAN(2).state_dict()}
    for name in ("a.pt", "b.pt"):
        torch.save(state, tmp_path / name)
    avg = sweep.average_states([tmp_path / "a.pt", tmp_path / "b.pt"])
    for part in ("gen", "patch_gan"):
        for key, value in state[part].items():
            assert torch.allclose(avg[part][key].to(value.dtype), value, atol=1e-6), key
