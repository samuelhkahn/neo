import configparser

import torch

from neo.eval.predictors import build_predictor
from neo.models.generator import Pix2PixGenerator
from neo.models.registry import generator_name, make_generator
from neo.models.swinir import SwinIR, shift_mask, window_partition, window_reverse
from neo.pix2pix import Pix2Pix


def tiny():
    return SwinIR(embed_dim=12, depths=(2,), num_heads=(2,), window_size=8, num_feat=8)


def config_for(name, **swinir):
    config = configparser.ConfigParser()
    config["MODEL"] = {"generator": name}
    if swinir:
        config["SWINIR"] = {k: repr(v) for k, v in swinir.items()}
    return config


def test_maps_padded_lr_to_hr_in_the_tanh_range_like_the_neo_generator():
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        y = tiny()(x)
    assert y.shape == (2, 1, 768, 768) and y.abs().max() < 1


def test_inputs_that_are_not_window_multiples_are_padded_then_cropped():
    with torch.no_grad():
        y = tiny()(torch.randn(1, 1, 20, 28))
    assert y.shape == (1, 1, 120, 168)


def test_window_partition_round_trip_and_shift_mask():
    x = torch.randn(2, 16, 24, 5)
    assert torch.equal(window_reverse(window_partition(x, 8), 8, 16, 24), x)
    mask = shift_mask(16, 16, 8, 4, "cpu")
    assert mask.shape == (4, 64, 64)
    assert (mask[0] == 0).all() and (mask[-1] != 0).any()


def test_gradient_checkpointing_gives_the_same_gradients():
    torch.manual_seed(0)
    a = SwinIR(embed_dim=12, depths=(2, 2), num_heads=(2, 2), num_feat=8)
    b = SwinIR(embed_dim=12, depths=(2, 2), num_heads=(2, 2), num_feat=8, use_checkpoint=True)
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 1, 32, 32)
    a(x).sum().backward()
    b(x).sum().backward()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5)


def test_registry_builds_each_generator():
    assert generator_name(configparser.ConfigParser()) == "neo"
    assert isinstance(make_generator(config_for("neo")), Pix2PixGenerator)
    model = make_generator(config_for("swinir", embed_dim=12, depths=(2,), num_heads=(2,)))
    assert isinstance(model, SwinIR) and model.conv_first.out_channels == 12


def test_pix2pix_keeps_the_neo_generator_unless_one_is_passed():
    assert isinstance(Pix2Pix(1, 1, "cpu").gen, Pix2PixGenerator)
    swin = tiny()
    assert Pix2Pix(1, 1, "cpu", generator=swin).gen is swin


def test_predictor_loads_a_swinir_checkpoint(tmp_path):
    config = config_for("swinir", embed_dim=12, depths=(2,), num_heads=(2,), num_feat=8)
    model = make_generator(config)
    torch.save({"gen": model.state_dict()}, tmp_path / "ckpt.pt")
    predict = build_predictor(config, tmp_path / "ckpt.pt", "cpu")
    lr = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        expected = model.eval()(lr)
    assert torch.allclose(predict(lr, None), expected)
