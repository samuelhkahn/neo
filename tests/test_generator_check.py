"""Loaders refuse a checkpoint written by another generator than the config builds.

neo and neo_relu (NEO with the decoder ReLU) have the same state_dict keys, so a strict
load_state_dict takes either's weights: without neo/models/registry.py check_generator an Arm A
checkpoint scored with lux.ini would silently run through the linear-decoder forward (or the
reverse). train.py records the generator in every checkpoint; older ones have no record.
"""

import configparser

import pytest
import torch
from test_sweep import CheapVGG, write_run

from neo.eval import sweep
from neo.eval.predictors import build_predictor
from neo.models.generator import Pix2PixGenerator
from neo.models.registry import check_generator


@pytest.fixture(autouse=True)
def drop_checkpoints(tmp_path):
    """Delete the checkpoints a test wrote (168 MB of generator weights each), which pytest
    would otherwise keep in its retained temp directories."""
    yield
    for path in tmp_path.rglob("*.pt"):
        path.unlink()


def config_for(name=None):
    config = configparser.ConfigParser()
    if name is not None:
        config["MODEL"] = {"generator": name}
    return config


def state_by(generator):
    """A checkpoint's top level, with the generator record when `generator` is given."""
    return {} if generator is None else {"generator": generator}


@pytest.mark.parametrize(
    ("saved", "builds"),
    [
        ("neo", None),  # no [MODEL] section: the NEO U-Net
        ("neo", "neo"),
        ("neo_relu", "neo_relu"),
        (None, "neo"),  # parity checkpoints written before the record existed
        (None, "swinir"),  # other layouts are left to the strict load, which catches a mismatch
        (None, "neo_lrtrunk"),
    ],
)
def test_matching_or_unrecorded_checkpoints_pass(saved, builds):
    check_generator(state_by(saved), config_for(builds), "ckpt.pt")


@pytest.mark.parametrize(
    ("saved", "builds"),
    [
        ("neo_relu", None),  # Arm A scored with lux.ini: the case that would go unnoticed
        ("neo_relu", "neo"),
        ("neo", "neo_relu"),  # the reverse
        (None, "neo_relu"),  # unrecorded with the neo layout = parity: Arm A always records itself
        ("neo_lrtrunk", "neo_relu"),
        ("neo_relu", "swinir"),
    ],
)
def test_a_checkpoint_of_another_generator_is_refused(saved, builds):
    with pytest.raises(ValueError, match="ckpt.pt was trained with generator"):
        check_generator(state_by(saved), config_for(builds), "ckpt.pt")


@pytest.mark.parametrize("saved", [None, "neo", "neo_relu"])
def test_predictor_loads_only_a_checkpoint_of_the_configs_generator(tmp_path, saved):
    path = tmp_path / "step.pt"
    torch.save({**state_by(saved), "gen": Pix2PixGenerator(1, 1).state_dict()}, path)
    if saved == "neo_relu":
        with pytest.raises(ValueError, match="generator 'neo_relu'.*builds 'neo'"):
            build_predictor(config_for(), path, "cpu")
    else:
        predict = build_predictor(config_for(), path, "cpu")
        assert predict(torch.zeros(1, 1, 128, 128), None).shape == (1, 1, 768, 768)


def record(path, generator):
    state = torch.load(path, weights_only=True)
    torch.save({**state, "generator": generator}, path)


def test_sweep_checks_every_checkpoint_including_each_averaged_one(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "VGGLoss", CheapVGG)
    config = write_run(tmp_path, [25000, 50000])  # lux.ini-like: no [MODEL], so neo
    ckpt = tmp_path / "ckpt"
    record(ckpt / "step_00025000.pt", "neo")
    record(ckpt / "step_00050000.pt", "neo_relu")
    args = ["--config", str(config), "--device", "cpu", "--no-comet", "--limit", "1"]
    with pytest.raises(ValueError, match="step_00050000.pt was trained with generator"):
        sweep.main([*args, "--out", str(tmp_path / "single"), "--min-step", "50000"])
    # the averaged state keeps only gen and patch_gan, so the member files are what is checked
    with pytest.raises(ValueError, match="step_00050000.pt was trained with generator"):
        sweep.main(
            [*args, "--out", str(tmp_path / "avg"), "--max-step", "0", "--average", "0:99999999"]
        )
