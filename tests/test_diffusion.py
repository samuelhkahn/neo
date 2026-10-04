"""Tests for the conditional EDM diffusion experiment (neo/diffusion.py, train_diffusion.py)."""

import configparser

import pytest
import torch

from neo import diffusion
from neo.eval.predictors import build_predictor
from neo.models.diffusion_unet import DiffusionUNet

SMALL = {"base_channels": "16", "channel_mults": "1,2", "blocks_per_level": "1"}


def small_model():
    torch.manual_seed(0)
    return diffusion.build(SMALL)


def test_unet_shapes_and_stride_check():
    unet = DiffusionUNet(base=16, mults=(1, 2, 2), blocks=1)
    out = unet(torch.randn(2, 2, 32, 48), torch.zeros(2))
    assert out.shape == (2, 1, 32, 48)
    with pytest.raises(ValueError, match="multiple of 4"):
        unet(torch.randn(1, 2, 30, 32), torch.zeros(1))


def test_default_model_matches_window():
    model = diffusion.build({})
    assert diffusion.WINDOW % model.unet.factor == 0
    assert diffusion.WINDOW_START % diffusion.GRID == 0
    # the window covers the central 600 px of the 768 px padded cutout
    assert diffusion.WINDOW_START <= 84 and diffusion.WINDOW_START + diffusion.WINDOW >= 684


def test_preconditioning_limits():
    """D(x; sigma) -> x as sigma -> 0 (c_skip -> 1, c_out -> 0)."""
    model = small_model()
    x, cond = torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        assert torch.allclose(model(x, torch.full((1,), 1e-6), cond), x, atol=1e-4)


def test_normalization_round_trip():
    model = small_model()
    model.set_stats(hr_mean=-0.6, hr_std=0.4, cond_mean=-0.35, cond_std=0.57)
    hr = torch.randn(2, 1, 8, 8)
    assert torch.allclose(model.denormalize(model.normalize(hr)), hr, atol=1e-6)
    assert torch.isclose(model.normalize(torch.full((1,), -0.2)), torch.tensor(0.5)).all()


def test_sigma_schedule():
    t = diffusion.sigma_steps(18)
    assert len(t) == 19 and t[-1] == 0
    assert t[0].item() == pytest.approx(80.0) and t[-2].item() == pytest.approx(0.002)
    assert (t[:-1].diff() < 0).all()


def test_random_crops_alignment():
    hr = torch.arange(768 * 768, dtype=torch.float32).view(1, 1, 768, 768).repeat(16, 1, 1, 1)
    hr_c, cond_c = diffusion.random_crops(
        hr, hr + 1, 256, generator=torch.Generator().manual_seed(1)
    )
    assert hr_c.shape == cond_c.shape == (16, 1, 256, 256)
    assert torch.equal(cond_c, hr_c + 1)  # HR and conditioning crops match
    y0 = (hr_c[:, 0, 0, 0] // 768).long()
    x0 = (hr_c[:, 0, 0, 0] % 768).long()
    for o in (y0, x0):
        assert ((o - diffusion.WINDOW_START) % diffusion.GRID == 0).all()
        assert (o >= diffusion.WINDOW_START).all()
        assert (o + 256 <= diffusion.WINDOW_START + diffusion.WINDOW).all()
    with pytest.raises(ValueError):
        diffusion.random_crops(hr, hr, 640)


def test_loss_decreases_on_one_batch():
    model = small_model()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    torch.manual_seed(1)
    cond = torch.randn(4, 1, 16, 16) * 0.5
    y = cond.flip(-1)  # a deterministic function of the conditioning
    losses = []
    for _ in range(60):
        torch.manual_seed(2)  # same noise draws each step
        loss = diffusion.edm_loss(model, y, cond)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_super_resolve_window_and_determinism():
    model = small_model()
    cond = torch.rand(1, 1, 768, 768) * 2 - 1
    torch.manual_seed(3)
    a = diffusion.super_resolve(model, cond, steps=2)
    torch.manual_seed(3)
    b = diffusion.super_resolve(model, cond, steps=2)
    assert a.shape == cond.shape and torch.equal(a, b)
    w0, w1 = diffusion.WINDOW_START, diffusion.WINDOW_START + diffusion.WINDOW
    outside = torch.ones_like(cond, dtype=torch.bool)
    outside[..., w0:w1, w0:w1] = False
    assert torch.equal(a[outside], cond[outside])
    assert not torch.equal(a[..., w0:w1, w0:w1], cond[..., w0:w1, w0:w1])
    assert torch.isfinite(a).all()


def test_super_resolve_clamps_to_training_range():
    model = small_model()
    model.set_stats(hr_min=-1.0, hr_max=-0.5)
    with torch.no_grad():
        model.unet.conv_out.bias.fill_(50.0)  # force a wild sample
    out = diffusion.super_resolve(model, torch.zeros(1, 1, 768, 768), steps=2)
    w0, w1 = diffusion.WINDOW_START, diffusion.WINDOW_START + diffusion.WINDOW
    assert out[..., w0:w1, w0:w1].min() >= -1.0 and out[..., w0:w1, w0:w1].max() <= -0.5


def test_ema_tracks_parameters():
    model = small_model()
    ema = diffusion.EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    model.set_stats(hr_mean=1.0, hr_std=2.0)
    before = [p.clone() for p in ema.model.parameters()]
    ema.update(model, step=1000)
    for e, b, p in zip(ema.model.parameters(), before, model.parameters(), strict=True):
        assert torch.allclose(e, 0.5 * b + 0.5 * p)
    assert ema.model.hr_std.item() == 2.0


def test_predictor_loads_checkpoint(tmp_path):
    model = small_model()
    model.set_stats(hr_mean=-0.6, hr_std=0.4, cond_mean=-0.35, cond_std=0.57)
    ema = diffusion.EMA(model)
    torch.save(
        {"step": 7, "model": model.state_dict(), "ema": ema.model.state_dict()}, tmp_path / "c.pt"
    )
    config = configparser.ConfigParser()
    config.read_dict(
        {"MODEL": {"generator": "diffusion"}, "DIFFUSION": {**SMALL, "sample_steps": "2"}}
    )
    predict = build_predictor(config, tmp_path / "c.pt", "cpu", mode="eval")
    cond = torch.rand(2, 1, 768, 768) * 2 - 1
    out = predict(torch.zeros(2, 1, 128, 128), cond)
    assert out.shape == (2, 1, 768, 768) and torch.isfinite(out).all()
