"""Tests for the conditional EDM diffusion experiment (neo/diffusion.py, train_diffusion.py)."""

import configparser
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from neo import diffusion
from neo.data.dataset import SR_HST_HSC_Dataset
from neo.eval.predictors import build_predictor
from neo.models.diffusion_unet import DiffusionUNet

SMALL = {"base_channels": "16", "channel_mults": "1,2", "blocks_per_level": "1"}
CONFIGS = Path(__file__).resolve().parent.parent / "neo" / "configs"


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


def hsc_hr_from_lr(lr):
    """(B, 100, 100) linear LR images -> (B, 1, 768, 768) hsc_hr as the dataset builds it: clip
    to [0, p99.999], ds9 log scale, 6x nearest upsampling, 84 px reflect padding."""
    out = []
    for image in lr:
        clipped = SR_HST_HSC_Dataset.clip(np.asarray(image, np.float32), use_data=False)[0]
        log = torch.from_numpy(SR_HST_HSC_Dataset.ds9_scaling(clipped, offset=1).astype(np.float32))
        up = log.repeat_interleave(6, 0).repeat_interleave(6, 1)
        out.append(F.pad(up[None, None], (84, 84, 84, 84), mode="reflect")[0])
    return torch.stack(out)


def sky(n, seed=0):
    """LSST-like LR sky: Gaussian noise of 3.35 stored units about 0.3 (real cutouts' values)."""
    return np.random.default_rng(seed).normal(0.3, 3.35, (n, 100, 100))


def test_lr_source_mask_finds_source_not_noise():
    yy, xx = np.mgrid[:100, :100]
    source = 150 * np.exp(-0.5 * ((yy - 40) ** 2 + (xx - 60) ** 2) / 1.5**2)
    noise = sky(4)
    with_source = diffusion.lr_source_mask(hsc_hr_from_lr(noise + source))
    lr_px = with_source[:, 0, 84:684:6, 84:684:6]
    assert (lr_px[:, 40, 60] == 1).all() and (lr_px[:, 38:43, 58:63] == 1).all()

    # Pure clipped noise: the threshold follows the noise, so few pixels pass (about 0.5% before
    # dilation; the clipped sky's upper tail is heavier than a Gaussian's), none at 6 sigma
    pure = hsc_hr_from_lr(noise)
    undilated = diffusion.lr_source_mask(pure, dilate=0)[..., 84:684, 84:684]
    assert (undilated.flatten(1).mean(1) < 0.02).all()
    assert diffusion.lr_source_mask(pure, nsigma=6.0, dilate=0).sum() == 0
    assert diffusion.lr_source_mask(hsc_hr_from_lr(noise + source), nsigma=6.0)[
        :, 0, 84 + 40 * 6, 84 + 60 * 6
    ].all()
    # an empty LR image masks nothing
    assert diffusion.lr_source_mask(hsc_hr_from_lr(np.zeros((1, 100, 100)))).sum() == 0


def real_like_sky(n, seed=0, n_sources=12):
    """Real-like LR cutouts (n, 100, 100) and their sky level: the real train cutouts' positive
    sky (about 0.4 stored units), noise (3.4) and neighbour-pixel noise correlation (0.25), with
    n_sources Gaussian sources each (peaks 5-200, sigma 0.8-2.5 LR px), which give about the
    real 14% mask coverage."""
    level, std, corr = 0.4, 3.4, 0.25
    rng = np.random.default_rng(seed)
    a = (1 - np.sqrt(1 - 2 * corr**2)) / (2 * corr)  # [a, 1, a] has lag-1 correlation corr
    white = rng.normal(0, std / (1 + 2 * a * a), (n, 102, 102))
    noise = a * white[:, :, :-2] + white[:, :, 1:-1] + a * white[:, :, 2:]
    noise = a * noise[:, :-2] + noise[:, 1:-1] + a * noise[:, 2:]
    y, x = rng.uniform(5, 95, (2, n, n_sources, 1, 1))
    peak = np.exp(rng.uniform(np.log(5), np.log(200), (n, n_sources, 1, 1)))
    width = rng.uniform(0.8, 2.5, (n, n_sources, 1, 1))
    yy, xx = np.mgrid[:100, :100]
    sources = peak * np.exp(-0.5 * ((yy - y) ** 2 + (xx - x) ** 2) / width**2)
    return level, level + noise + sources.sum(1)


def test_lr_source_mask_noise_rate():
    """lr_source_mask's documented noise rate (2.3% of the area after dilation on 50 real train
    cutouts) on a real-like sky: each image's threshold applied to the image reflected about its
    sky level, which clips the sources away and keeps the noise's distribution. White noise about
    0 is not enough: the positive sky and the correlated noise both raise the rate."""
    level, lr = real_like_sky(24)
    real = hsc_hr_from_lr(lr)
    _, threshold = diffusion.lr_detection(real)
    noise, _ = diffusion.lr_detection(hsc_hr_from_lr(2 * level - lr))
    flagged = F.max_pool2d((noise > threshold).float(), 5, stride=1, padding=2)  # dilate = 2
    coverage = diffusion.lr_source_mask(real)[..., 84:684, 84:684].mean()
    assert 0.10 < coverage < 0.20  # as on real cutouts (14%), so the threshold is comparable
    assert flagged.mean() < 0.025  # 1.5% here; thresholds 10% lower already exceed the bound


def test_lr_source_mask_registration():
    """The mask sits exactly on the LR pixels it was detected from: on a zero sky (p50 = p75 = 0)
    only the bright pixels pass; a 3-pixel source is kept and an isolated pixel is dropped."""
    source = [(40, 61), (40, 62), (41, 61)]
    lr = np.zeros((1, 100, 100))
    for r, c in source:
        lr[0, r, c] = 50.0
    lr[0, 10, 10] = 50.0  # fewer than MIN_CLUSTER detected pixels in its 3 x 3 neighbourhood
    cond = hsc_hr_from_lr(lr)
    for dilate in (0, 1):
        grid = torch.zeros(1, 1, 100, 100)
        for r, c in source:
            grid[..., r - dilate : r + dilate + 1, c - dilate : c + dilate + 1] = 1
        expected = torch.zeros_like(cond)
        expected[..., 84:684, 84:684] = grid.repeat_interleave(6, -2).repeat_interleave(6, -1)
        assert torch.equal(diffusion.lr_source_mask(cond, smooth=0, dilate=dilate), expected)


def test_lr_source_mask_geometry():
    yy, xx = np.mgrid[:100, :100]
    source = 100 * np.exp(-0.5 * ((yy - 2) ** 2 + (xx - 97) ** 2) / 1.0**2)  # at an edge
    cond = hsc_hr_from_lr(sky(2, seed=1) + source)
    mask = diffusion.lr_source_mask(cond)
    assert mask.shape == cond.shape and mask.dtype == torch.float32
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    # zero in the padding, constant over each 6 x 6 block of an LR pixel
    outside = torch.ones_like(mask, dtype=torch.bool)
    outside[..., 84:684, 84:684] = False
    assert mask[outside].sum() == 0
    blocks = mask[..., 84:684, 84:684].reshape(2, 1, 100, 6, 100, 6)
    assert torch.equal(blocks, blocks[:, :, :, :1, :, :1].expand_as(blocks))
    # dilation only grows the mask
    small = diffusion.lr_source_mask(cond, dilate=0)
    large = diffusion.lr_source_mask(cond, dilate=3)
    assert (small <= mask).all() and (mask <= large).all()
    assert small.sum() < mask.sum() < large.sum()


def old_random_crops(hr, cond, size, generator=None):
    """random_crops as it was before crop_offsets / apply_crops existed."""
    n_offsets = (diffusion.WINDOW - size) // diffusion.GRID + 1
    offsets = diffusion.WINDOW_START + diffusion.GRID * torch.randint(
        n_offsets, (hr.shape[0], 2), generator=generator
    )
    hr_crops, cond_crops = [], []
    for (y, x), h, c in zip(offsets.tolist(), hr, cond, strict=True):
        hr_crops.append(h[..., y : y + size, x : x + size])
        cond_crops.append(c[..., y : y + size, x : x + size])
    return torch.stack(hr_crops), torch.stack(cond_crops)


def test_random_crops_unchanged():
    hr, cond = torch.randn(2, 6, 1, 768, 768).unbind(0)
    for seed, size in [(0, 256), (5, 128), (9, 624)]:
        new = diffusion.random_crops(hr, cond, size, torch.Generator().manual_seed(seed))
        old = old_random_crops(hr, cond, size, torch.Generator().manual_seed(seed))
        assert all(torch.equal(a, b) for a, b in zip(new, old, strict=True))
    torch.manual_seed(3)  # the global generator, as in training
    new = diffusion.random_crops(hr, cond, 256)
    torch.manual_seed(3)
    old = old_random_crops(hr, cond, 256)
    assert all(torch.equal(a, b) for a, b in zip(new, old, strict=True))


def contains(offsets, size, pixels):
    """Whether each crop at offsets (n, 2) contains its pixel (n, 2)."""
    return ((pixels >= offsets) & (pixels < offsets + size)).all(1)


def test_source_crops_contain_a_mask_pixel():
    n, size = 64, 256
    w0, w1 = diffusion.WINDOW_START, diffusion.WINDOW_START + diffusion.WINDOW
    last = w0 + diffusion.GRID * ((diffusion.WINDOW - size) // diffusion.GRID)  # 438
    gen = torch.Generator().manual_seed(0)
    # one mask pixel per image, anywhere a crop on the grid can reach (up to last + size - 1)
    pixels = torch.randint(w0, last + size, (n, 2), generator=gen)
    pixels[0] = torch.tensor([w0, last + size - 1])  # the reachable corners
    pixels[1] = torch.tensor([last + size - 1, last + size - 1])
    mask = torch.zeros(n, 1, 768, 768)
    mask[torch.arange(n), 0, pixels[:, 0], pixels[:, 1]] = 1
    offsets = diffusion.crop_offsets(n, size, mask, 1.0, gen)
    assert contains(offsets, size, pixels).all()
    assert ((offsets - w0) % diffusion.GRID == 0).all()
    assert (offsets >= w0).all() and (offsets + size <= w1).all() and (offsets <= last).all()
    crops = diffusion.apply_crops(offsets, size, mask)[0]
    assert (crops.flatten(1).sum(1) == 1).all()
    # centred on the pixel up to the grid snap, unless the centred crop leaves the window
    free = ((pixels - size // 2 >= w0) & (pixels - size // 2 <= last)).all(1)
    assert free.sum() > n // 4
    off_centre = (offsets + size // 2 - pixels)[free].abs()
    assert (off_centre <= diffusion.GRID // 2).all()

    # A fraction: frac * n images get source crops (16 px crops, single pixels: a uniform crop
    # almost never contains its image's pixel)
    for frac in (0.25, 0.5):
        offsets = diffusion.crop_offsets(n, 16, mask, frac, torch.Generator().manual_seed(2))
        assert contains(offsets, 16, pixels).sum() == frac * n


def test_source_crops_choose_mask_pixels_uniformly():
    # two mask pixels too far apart for one 64 px crop: each is chosen about half of the time
    n, size = 400, 64
    pixels = torch.tensor([[100, 100], [600, 600]])
    mask = torch.zeros(n, 1, 768, 768)
    mask[:, 0, pixels[:, 0], pixels[:, 1]] = 1
    offsets = diffusion.crop_offsets(n, size, mask, 1.0, torch.Generator().manual_seed(5))
    first = contains(offsets, size, pixels[:1].expand(n, 2))
    assert (first ^ contains(offsets, size, pixels[1:].expand(n, 2))).all()
    assert 0.4 < first.float().mean() < 0.6


def test_source_crops_need_a_one_channel_mask():
    gen = torch.Generator().manual_seed(0)
    for shape in [(4, 2, 768, 768), (4, 768, 768), (3, 1, 768, 768), (4, 1, 600, 600)]:
        with pytest.raises(ValueError, match="mask must be"):
            diffusion.crop_offsets(4, 256, torch.ones(shape), 1.0, gen)


def test_source_crops_fall_back_to_uniform():
    n, size = 8, 256
    mask = torch.zeros(n, 1, 768, 768)
    mask[1, 0, 300, 300] = 1  # the only image with a mask pixel
    mask[2, 0, 695, 695] = 1  # inside the window, but no crop on the grid reaches it
    mask[3, 0, 50, 50] = 1  # outside the window
    uniform = diffusion.crop_offsets(n, size, generator=torch.Generator().manual_seed(4))
    offsets = diffusion.crop_offsets(n, size, mask, 1.0, torch.Generator().manual_seed(4))
    keep = torch.arange(n) != 1
    assert torch.equal(offsets[keep], uniform[keep])
    assert contains(offsets[1:2], size, torch.tensor([[300, 300]])).all()
    # no mask or source_frac = 0: the uniform offsets, from the same random draws
    assert torch.equal(
        diffusion.crop_offsets(n, size, mask, 0.0, torch.Generator().manual_seed(4)), uniform
    )


def old_edm_loss(model, y, cond, p_mean=-1.2, p_std=1.2):
    """edm_loss as it was before the object term."""
    sigma = (torch.randn(y.shape[0], device=y.device) * p_std + p_mean).exp()
    noisy = y + torch.randn_like(y) * sigma.view(-1, 1, 1, 1)
    denoised = model(noisy, sigma, cond)
    sd = model.sigma_data
    weight = ((sigma**2 + sd**2) / (sigma * sd) ** 2).view(-1, 1, 1, 1)
    return (weight * (denoised - y) ** 2).mean()


def test_edm_loss_object_term():
    model = small_model()
    y, cond = torch.randn(2, 3, 1, 16, 16).unbind(0)

    def loss(fn, *args, **kwargs):
        torch.manual_seed(7)  # the same noise draws for every call
        return fn(model, y, cond, *args, **kwargs)

    base = loss(old_edm_loss)
    assert torch.equal(loss(diffusion.edm_loss), base)
    ones, zeros = torch.ones_like(y), torch.zeros_like(y)
    assert torch.equal(loss(diffusion.edm_loss, mask=ones, object_weight=0.0), base)
    # all pixels are object pixels: the object term is the mean again
    total, loss_all, loss_obj = loss(diffusion.edm_loss, mask=ones, object_weight=0.7, parts=True)
    assert torch.equal(loss_all, base)
    assert torch.allclose(loss_obj, base) and torch.allclose(total, 1.7 * base)
    # no object pixels: the object term is 0 and the total the plain loss, gradients finite
    total, _, loss_obj = loss(diffusion.edm_loss, mask=zeros, object_weight=1.0, parts=True)
    assert torch.equal(total, base) and loss_obj.item() == 0
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # a partial mask: the masked mean over the whole batch, so the means over a mask and over its
    # complement, weighted by their pixel counts, add up to the plain sum
    part = torch.zeros_like(y)
    part[0, ..., :8, :] = 1
    part[2, ..., :, :3] = 1
    _, _, inside = loss(diffusion.edm_loss, mask=part, parts=True)
    _, _, outside = loss(diffusion.edm_loss, mask=1 - part, parts=True)
    n_in, n_out = part.sum(), (1 - part).sum()
    assert torch.allclose(inside * n_in + outside * n_out, base * y.numel())
    _, loss_all, none = loss(diffusion.edm_loss, parts=True)
    assert none is None and torch.equal(loss_all, base)


def test_source_focus_options():
    default = diffusion.SourceFocus.from_section({})
    assert default == diffusion.SourceFocus() and not default.enabled and default.name() == ""
    lr = diffusion.SourceFocus.from_section(
        {"object_mask": "lr", "object_weight": "1.0", "source_crop_frac": "0.5"}
    )
    assert lr.enabled and lr.needs_lr_mask
    assert lr.name() == "_object_mask=lr_object_weight=1.0_source_crop_frac=0.5"
    hst = diffusion.SourceFocus(object_mask="hst", object_weight=1.0)
    assert hst.enabled and not hst.needs_lr_mask
    assert diffusion.SourceFocus(source_crop_frac=0.5).needs_lr_mask
    bad = [
        ({"object_mask": "segmap"}, "unknown object_mask"),
        ({"object_weight": "1.0"}, "needs object_mask"),
        ({"object_mask": "lr", "object_weight": "-1"}, "object_weight"),
        ({"source_crop_frac": "1.5"}, "source_crop_frac"),
        ({"source_crop_frac": "-0.1"}, "source_crop_frac"),
        ({"mask_dilate": "-1"}, "mask_dilate"),
        # its blur's reflect padding would not fit the 100 px LR grid
        ({"object_mask": "lr", "mask_smooth": "24.875"}, "mask_smooth"),
    ]
    for section, match in bad:
        with pytest.raises(ValueError, match=match):
            diffusion.SourceFocus.from_section(section)
    # the largest smoothing that validates also runs
    widest = diffusion.SourceFocus(object_mask="lr", mask_smooth=24.87)
    assert widest.lr_mask(torch.rand(2, 1, 768, 768) - 1).shape == (2, 1, 768, 768)


def test_hst_weight_mask_is_central():
    segmap = torch.ones(1, 1, 768, 768)
    mask = diffusion.SourceFocus(object_mask="hst", object_weight=1.0).weight_mask(None, segmap)
    assert mask.sum() == 600 * 600 and mask[..., 84:684, 84:684].all()
    assert diffusion.SourceFocus().weight_mask(None, segmap) is None


@pytest.mark.parametrize("arm, mask", [("lrfocus", "lr"), ("hstfocus", "hst")])
def test_focus_configs_differ_from_baseline_only_in_focus(arm, mask):
    def read(name):
        config = configparser.ConfigParser(interpolation=None)
        config.read(CONFIGS / name)
        return {(s, k): v for s in config.sections() for k, v in config.items(s, raw=True)}

    base, focus = read("lux_diffusion.ini"), read(f"lux_diffusion_{arm}.ini")
    identifier = f"lux_cosmosweb_i_diffusion_{arm}"
    expected = {
        **base,
        ("IDENTIFIER", "identifier"): f'"{identifier}"',
        ("CHECKPOINT", "ckpt_dir"): "${NEO_DATA}/checkpoints/" + identifier,
        ("DIFFUSION", "object_mask"): mask,
        ("DIFFUSION", "object_weight"): "1.0",
        ("DIFFUSION", "source_crop_frac"): "0.5",
        ("DIFFUSION", "mask_nsigma"): "4.5",
        ("DIFFUSION", "mask_smooth"): "1.0",
        ("DIFFUSION", "mask_dilate"): "2",
    }
    tag = focus.pop(("COMET_TAG", "comet_tag"))
    expected.pop(("COMET_TAG", "comet_tag"))
    assert tag.strip('"').endswith(arm)
    assert focus == expected
    section = configparser.ConfigParser()
    section.read(CONFIGS / f"lux_diffusion_{arm}.ini")
    focus_options = diffusion.SourceFocus.from_section(section["DIFFUSION"])
    assert focus_options.object_mask == mask and focus_options.source_crop_frac == 0.5
