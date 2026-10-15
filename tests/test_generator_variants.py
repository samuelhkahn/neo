"""The generator ablation arms (neo/models/generator_variants.py), their registry names and configs.

Arm A (neo_relu) = NEO with the decoder ReLU the paper specifies; Arm B (neo_lrtrunk) = Arm A + a
trunk at the 128 px LR grid. The parity generator (Pix2PixGenerator, lux.ini) must stay as it is.
"""

import configparser
import importlib.util
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torchvision

import neo.models.vgg19
from neo.eval.predictors import build_predictor
from neo.models.conv_pixel_shuffle import ConvPixelShuffle
from neo.models.generator import Pix2PixGenerator
from neo.models.generator_variants import (
    RCAB,
    NeoLRTrunkGenerator,
    NeoReLUGenerator,
    ResBlock,
    icnr_,
)
from neo.models.registry import check_generator, generator_name, make_generator
from neo.pix2pix import Pix2Pix

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "neo" / "configs"
DECODERS = [f"d{i}" for i in range(1, 8)]
# Pinned so that an accidental architecture change shows up; NEO's decoder alone is 26.5 M.
N_PARAMS = {"neo": 42_011_171, "neo_relu": 42_011_171, "neo_lrtrunk": 42_652_505}


@pytest.fixture(autouse=True)
def drop_checkpoints(tmp_path):
    """Delete the checkpoints a test wrote (168 MB of generator weights each), which pytest
    would otherwise keep in its retained temp directories."""
    yield
    for path in tmp_path.rglob("*.pt"):
        path.unlink()


@pytest.fixture
def random_vgg(monkeypatch):
    """VGGLoss with random VGG19 weights: Pix2Pix builds without the pretrained download."""
    vgg19 = torchvision.models.vgg19
    monkeypatch.setattr(neo.models.vgg19.models, "vgg19", lambda pretrained=True: vgg19())


def config_for(name):
    config = configparser.ConfigParser()
    config["MODEL"] = {"generator": name}
    return config


def linear_decoder_forward(gen, x):
    """Pix2PixGenerator.forward as written when the arms were added: no decoder ReLU."""

    def block(decoder, x):
        x = decoder.bn(decoder.deconv(x))
        return decoder.drop(x) if decoder.dropout else x

    skips = []
    for encoder in gen.encoders:
        x = encoder(x)
        skips.append(x)
    for decoder, skip in zip(gen.decoders[:-1], reversed(skips[:-1]), strict=True):
        x = torch.cat((block(decoder, x), skip), axis=1)
    x = block(gen.decoders[-1], x)
    return gen.tanh(gen.final_conv(gen.ps_blocks(x)))


def decoder_outputs(gen, x):
    """Each decoder block's output as the next stage receives it, from forward pre-hooks on that
    stage: the next block's transposed conv (input = cat(output, skip)), then for d7 the upsampler
    or, in the trunk arm, the merge conv (input = cat(head(lr), d7))."""
    seen = {}

    def keep(name, channels):
        return lambda module, inputs: seen.__setitem__(name, inputs[0][:, channels])

    hooks = [
        after.deconv.register_forward_pre_hook(keep(f"d{i + 1}", slice(block.deconv.out_channels)))
        for i, (block, after) in enumerate(zip(gen.decoders[:-1], gen.decoders[1:], strict=True))
    ]
    if isinstance(gen, NeoLRTrunkGenerator):
        hooks.append(
            gen.merge.register_forward_pre_hook(keep("d7", slice(gen.head.out_channels, None)))
        )
    else:
        hooks.append(gen.ps_blocks.register_forward_pre_hook(keep("d7", slice(None))))
    with torch.no_grad():
        gen(x)
    for hook in hooks:
        hook.remove()
    return seen


@pytest.mark.parametrize("name", ["neo", "neo_relu", "neo_lrtrunk"])
def test_maps_padded_lr_to_hr_in_the_tanh_range(name):
    torch.manual_seed(0)
    gen = make_generator(config_for(name))
    with torch.no_grad():
        y = gen(torch.randn(2, 1, 128, 128))
    assert y.shape == (2, 1, 768, 768) and y.abs().max() < 1
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"{name}: {n_params:,} parameters")
    assert n_params == N_PARAMS[name]


def test_registry_names():
    assert generator_name(configparser.ConfigParser()) == "neo"
    assert type(make_generator(config_for("neo"))) is Pix2PixGenerator  # parity: not a subclass
    assert type(make_generator(config_for("neo_relu"))) is NeoReLUGenerator
    assert type(make_generator(config_for("neo_lrtrunk"))) is NeoLRTrunkGenerator
    with pytest.raises(ValueError, match="unknown generator"):
        make_generator(config_for("neo_nope"))


def test_pix2pix_keeps_the_neo_generator_unless_one_is_passed(random_vgg):
    assert type(Pix2Pix(1, 1, "cpu").gen) is Pix2PixGenerator
    arm = NeoLRTrunkGenerator()
    p = Pix2Pix(1, 1, "cpu", generator=arm)
    assert p.gen is arm
    in_optimizer = {id(q) for group in p.gen_opt.param_groups for q in group["params"]}
    assert in_optimizer == {id(q) for q in arm.parameters()}


def test_parity_generator_forward_is_unchanged_and_its_decoder_stays_linear():
    torch.manual_seed(1)
    gen = Pix2PixGenerator(1, 1)
    x = torch.randn(2, 1, 128, 128)
    for train in (True, False):  # train mode with the same dropout masks, then eval mode
        gen.train(train)
        torch.manual_seed(2)
        with torch.no_grad():
            y = gen(x)
        torch.manual_seed(2)
        with torch.no_grad():
            expected = linear_decoder_forward(gen, x)
        assert torch.equal(y, expected)
    gen.train()
    assert all((out < 0).any() for out in decoder_outputs(gen, x).values())


def test_relu_arm_has_the_parity_modules_and_a_nonnegative_decoder():
    torch.manual_seed(3)
    parity = Pix2PixGenerator(1, 1)
    arm = NeoReLUGenerator(1, 1)
    assert list(arm.state_dict()) == list(parity.state_dict())
    arm.load_state_dict(parity.state_dict())  # strict: the same keys and shapes
    x = torch.randn(2, 1, 128, 128)
    outputs = decoder_outputs(arm, x)
    assert sorted(outputs) == sorted(DECODERS)
    assert all((out >= 0).all() and (out > 0).any() for out in outputs.values())
    with torch.no_grad():
        torch.manual_seed(4)
        y_arm = arm(x)
        torch.manual_seed(4)
        y_parity = parity(x)
    assert not torch.allclose(y_arm, y_parity)  # the ReLU changes the output for the same weights


def test_relu_is_applied_after_batch_norm_and_dropout():
    """d1-d3 have Dropout2d: the arm's output there is ReLU(dropout(BN(deconv(x))))."""
    torch.manual_seed(5)
    arm = NeoReLUGenerator(1, 1).train()
    block = arm.decoders[0]
    x = torch.randn(2, 512, 1, 1)
    torch.manual_seed(6)
    out = arm._decode("d1", block, x)
    torch.manual_seed(6)
    expected = torch.relu(block.drop(block.bn(block.deconv(x))))
    assert torch.equal(out, expected)


def test_trunk_arm_drops_the_parent_upsampler_and_uses_every_parameter():
    torch.manual_seed(7)
    gen = NeoLRTrunkGenerator(1, 1)
    keys = list(gen.state_dict())
    assert not any(k.startswith(("ps_blocks.", "final_conv.")) for k in keys)
    parity_keys = [k for k in Pix2PixGenerator(1, 1).state_dict() if k.startswith(("enc", "dec"))]
    assert [k for k in keys if k.startswith(("enc", "dec"))] == parity_keys
    gen(torch.randn(2, 1, 128, 128)).mean().backward()
    unused = [n for n, p in gen.named_parameters() if p.grad is None]
    assert unused == []
    outputs = decoder_outputs(gen, torch.randn(2, 1, 128, 128))
    assert sorted(outputs) == sorted(DECODERS) and all((o >= 0).all() for o in outputs.values())


@pytest.mark.parametrize("r", [2, 3])
def test_icnr_makes_the_initial_shuffle_a_nearest_neighbour_upsampling(r):
    torch.manual_seed(8)
    up = ConvPixelShuffle(16, 8, upscale_factor=r)
    icnr_(up.convolution, r)
    w = up.convolution.weight.view(8, r * r, 16, 3, 3)
    assert torch.equal(w, w[:, :1].expand_as(w))  # all r**2 sub-filters of a channel are equal
    assert (up.convolution.bias == 0).all()
    with torch.no_grad():
        y = up(torch.randn(2, 16, 10, 10))
    # constant within each r x r block, for any input
    coarse = y[..., ::r, ::r]
    assert torch.equal(y, coarse.repeat_interleave(r, dim=-2).repeat_interleave(r, dim=-1))
    with pytest.raises(ValueError, match="multiple"):
        icnr_(nn.Conv2d(4, 10, 3), 2)


def test_trunk_arm_upsampler_starts_from_icnr():
    torch.manual_seed(9)
    gen = NeoLRTrunkGenerator(1, 1)
    convs = [m.convolution for m in gen.upsampler if isinstance(m, ConvPixelShuffle)]
    for conv, r in zip(convs, (3, 2), strict=True):
        w = conv.weight.view(conv.out_channels // r**2, r * r, *conv.weight.shape[1:])
        assert torch.equal(w, w[:, :1].expand_as(w))
    # without the PReLU/ResBlock in between, the 3x shuffle of any feature map is blockwise constant
    with torch.no_grad():
        y = gen.upsampler[0](torch.randn(1, 64, 8, 8))
    assert torch.equal(y, y[..., ::3, ::3].repeat_interleave(3, -2).repeat_interleave(3, -1))


def test_rcab_and_resblock_keep_the_shape_and_are_residual():
    torch.manual_seed(10)
    x = torch.randn(2, 64, 12, 12)
    rcab, res = RCAB(64, reduction=16), ResBlock(64)
    assert rcab.attention[1].out_channels == 4  # squeeze to 64 / 16 channels
    with torch.no_grad():
        assert rcab(x).shape == x.shape and res(x).shape == x.shape
        attention = rcab.attention(rcab.body(x))
        assert attention.shape == (2, 64, 1, 1) and ((attention > 0) & (attention < 1)).all()
        assert torch.allclose(rcab(x), x + rcab.body(x) * attention)
        # with the body's last conv at zero both blocks are the identity
        for block in (rcab, res):
            nn.init.zeros_(block.body[2].weight)
            nn.init.zeros_(block.body[2].bias)
            assert torch.equal(block(x), x)


@pytest.mark.parametrize("cls", [NeoReLUGenerator, NeoLRTrunkGenerator])
def test_activation_stats_only_when_recording(cls):
    torch.manual_seed(11)
    gen = cls(1, 1).train()
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        gen(x)
    assert gen.activation_stats() == {}  # off by default: training steps compute nothing

    # the recorded fraction is that of each block's batch-norm output <= 0 (before dropout)
    expected = {}
    hooks = [
        d.bn.register_forward_hook(
            lambda m, i, out, name=f"d{k + 1}": expected.__setitem__(
                name, (out <= 0).float().mean().item()
            )
        )
        for k, d in enumerate(gen.decoders)
    ]
    gen.record_activations = True
    with torch.no_grad():
        gen(x)
    gen.record_activations = False
    for h in hooks:
        h.remove()
    stats = gen.activation_stats()
    assert list(stats) == DECODERS
    assert all(0 <= v <= 1 for v in stats.values())
    assert stats == pytest.approx(expected, abs=1e-6)

    with torch.no_grad():
        gen(torch.randn(2, 1, 128, 128))
    assert gen.activation_stats() == stats  # not recomputed while recording is off


def test_parity_generator_has_no_activation_stats():
    """train.py logs ReLU statistics only for generators that have activation_stats."""
    assert not hasattr(Pix2PixGenerator(1, 1), "activation_stats")


def read_config(path):
    """{section: {key: raw value}}, with [DEFAULT] read as a section of its own (configparser
    would otherwise copy its keys into every section)."""
    config = configparser.ConfigParser(interpolation=None, default_section="(none)")
    config.read(path)
    return {section: dict(config[section]) for section in config.sections()}


@pytest.mark.parametrize("arm", ["neo_relu", "neo_lrtrunk"])
def test_arm_configs_differ_from_lux_only_in_name_checkpoints_tag_and_generator(arm):
    lux = read_config(CONFIGS / "lux.ini")
    cfg = read_config(CONFIGS / f"lux_{arm}.ini")
    identifier = f"lux_cosmosweb_i_{arm}"
    assert cfg["IDENTIFIER"]["identifier"] == f'"{identifier}"'
    assert cfg["CHECKPOINT"]["ckpt_dir"] == "${NEO_DATA}/checkpoints/" + identifier
    assert cfg["COMET_TAG"]["comet_tag"].strip('"').endswith(f" {arm}")
    assert cfg["MODEL"]["generator"] == arm
    differ = {
        (section, key)
        for section in lux.keys() | cfg.keys()
        for key in lux.get(section, {}).keys() | cfg.get(section, {}).keys()
        if lux.get(section, {}).get(key) != cfg.get(section, {}).get(key)
    }
    assert differ == {
        ("IDENTIFIER", "identifier"),
        ("CHECKPOINT", "ckpt_dir"),
        ("COMET_TAG", "comet_tag"),
        ("MODEL", "generator"),
    }
    assert lux["GAN_STEPS"]["gan_steps"] == cfg["GAN_STEPS"]["gan_steps"] == "2000000"
    assert (
        type(make_generator(configparser_of(CONFIGS / f"lux_{arm}.ini")))
        is {
            "neo_relu": NeoReLUGenerator,
            "neo_lrtrunk": NeoLRTrunkGenerator,
        }[arm]
    )


def configparser_of(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config


def load_train_module():
    """train.py as a module (it is a script at the repo root, not part of the neo package)."""
    spec = importlib.util.spec_from_file_location("train", ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_records_the_generator_in_every_checkpoint(tmp_path):
    train = load_train_module()
    gen, disc = NeoReLUGenerator(1, 1), nn.Linear(2, 2)
    pix2pix = types.SimpleNamespace(
        gen=gen,
        patch_gan=disc,
        gen_opt=torch.optim.Adam(gen.parameters()),
        disc_opt=torch.optim.Adam(disc.parameters()),
        generator_name="neo_relu",  # set by train.py's main() from [MODEL] generator
    )
    train.save_checkpoint(tmp_path / "latest.pt", pix2pix, 7, "run")
    state = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert state["generator"] == "neo_relu" and state["step"] == 7
    check_generator(state, config_for("neo_relu"))
    with pytest.raises(ValueError, match="trained with generator 'neo_relu'"):
        check_generator(state, configparser.ConfigParser())  # lux.ini: the parity generator


def test_relu_arm_predictor_refuses_parity_weights_and_runs_its_own(tmp_path):
    """An unrecorded checkpoint with the neo layout is a parity run: under a neo_relu config it
    would load (same keys) and be scored as Arm A, so it is refused."""
    torch.manual_seed(12)
    weights = NeoReLUGenerator(1, 1).state_dict()
    unrecorded, recorded = tmp_path / "old.pt", tmp_path / "arm.pt"
    torch.save({"gen": weights}, unrecorded)
    torch.save({"gen": weights, "generator": "neo_relu"}, recorded)
    with pytest.raises(ValueError, match="generator 'neo'.*builds 'neo_relu'"):
        build_predictor(config_for("neo_relu"), unrecorded, "cpu")
    x = torch.randn(1, 1, 128, 128)
    out = build_predictor(config_for("neo_relu"), recorded, "cpu")(x, None)
    reference = NeoReLUGenerator(1, 1)
    reference.load_state_dict(weights)
    with torch.no_grad():
        assert torch.equal(out, reference.eval()(x))


def test_sweep_scores_and_averages_relu_arm_checkpoints(monkeypatch, tmp_path):
    """Each averaged member is checked, not the average (which keeps no record), so averages
    of an Arm A run are scored under its own config."""
    from test_sweep import CheapVGG, rows, write_run

    from neo.eval import sweep

    monkeypatch.setattr(sweep, "VGGLoss", CheapVGG)
    path = write_run(tmp_path, [25000, 50000])
    config = configparser_of(path)
    config["MODEL"] = {"generator": "neo_relu"}
    with open(path, "w") as f:
        config.write(f)
    for step in (25000, 50000):
        file = tmp_path / "ckpt" / f"step_{step:08d}.pt"
        torch.save({**torch.load(file, weights_only=True), "generator": "neo_relu"}, file)
    out = tmp_path / "out"
    sweep.main(
        [*("--config", str(path), "--out", str(out), "--device", "cpu", "--no-comet")]
        + ["--limit", "1", "--average", "25000:50000"]
    )
    assert [r["checkpoint"] for r in rows(out / "sweep.csv")] == [
        "step_00025000",
        "step_00050000",
        "avg_25000-50000",
    ]


@pytest.mark.parametrize(
    "arm, knob",
    [
        ("neo_relu_d4", ("DATA_AUG", "augment", "shift_d4")),
        ("neo_relu_cleand", ("DATASET", "hsc_hr_8bit", "none")),
    ],
)
def test_screening_configs_differ_from_arm_a_only_in_their_knob_and_length(arm, knob):
    """Each screening arm is Arm A plus one change, trained for 300k steps."""
    base = read_config(CONFIGS / "lux_neo_relu.ini")
    cfg = read_config(CONFIGS / f"lux_{arm}.ini")
    identifier = f"lux_cosmosweb_i_{arm}"
    assert cfg["IDENTIFIER"]["identifier"] == f'"{identifier}"'
    assert cfg["CHECKPOINT"]["ckpt_dir"] == "${NEO_DATA}/checkpoints/" + identifier
    assert cfg["COMET_TAG"]["comet_tag"].strip('"').endswith(f" {arm}")
    assert cfg["MODEL"]["generator"] == "neo_relu" and cfg["GAN_STEPS"]["gan_steps"] == "300000"
    section, key, value = knob
    assert cfg[section][key] == value
    differ = {
        (s, k)
        for s in base.keys() | cfg.keys()
        for k in base.get(s, {}).keys() | cfg.get(s, {}).keys()
        if base.get(s, {}).get(k) != cfg.get(s, {}).get(k)
    }
    assert differ == {
        ("IDENTIFIER", "identifier"),
        ("CHECKPOINT", "ckpt_dir"),
        ("COMET_TAG", "comet_tag"),
        ("GAN_STEPS", "gan_steps"),
        (section, key),
    }
