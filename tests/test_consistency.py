"""The LSST forward model and the flux-consistency loss (neo/consistency.py, [CONSISTENCY])."""

import configparser
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from test_eval_units import NJY_HR, NJY_LR, write_pair

from neo import consistency as cs
from neo.noise import to_linear, to_log

HR_PIX = 0.2 / 6


def padded_log(linear_600, pad=84):
    """(B, 1, 600, 600) linear HR image -> the dataset's padded log tensor (B, 1, 768, 768)."""
    return to_log(F.pad(linear_600, (pad, pad, pad, pad), mode="reflect"))


def lr_tensor(linear_100, pad=14):
    """(B, 1, 100, 100) linear LR image -> the dataset's padded log tensor (B, 1, 128, 128)."""
    return to_log(F.pad(linear_100.clamp(min=0), (pad, pad, pad, pad), mode="reflect"))


def gaussian(size, cy, cx, sigma, amp):
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return amp * torch.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / sigma**2)


def moments(img):
    """Flux-weighted centroid and second moment (sigma, px) of a (H, W) image."""
    yy, xx = torch.meshgrid(torch.arange(img.shape[0]), torch.arange(img.shape[1]), indexing="ij")
    w = img.double()
    total = w.sum()
    cy, cx = (w * yy).sum() / total, (w * xx).sum() / total
    var = ((w * (yy - cy) ** 2).sum() + (w * (xx - cx) ** 2).sum()) / (2 * total)
    return cy.item(), cx.item(), math.sqrt(var.item())


def test_blur_sigma_is_the_quadrature_psf_difference_in_hr_pixels():
    model = cs.ForwardModel(1.0, psf_lr_fwhm=0.85, psf_hr_fwhm=0.10)
    expected = math.sqrt(0.85**2 - 0.10**2) / (2 * math.sqrt(2 * math.log(2))) / HR_PIX
    assert model.sigma_px == pytest.approx(expected, rel=1e-6)
    assert model.kernel.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert model.kernel.numel() == 2 * math.ceil(4 * expected) + 1
    none = cs.ForwardModel(1.0, psf_lr_fwhm=0.10, psf_hr_fwhm=0.10)  # no blur at all
    assert none.sigma_px == 0 and none.kernel.tolist() == [1.0]
    x = torch.rand(1, 1, 12, 12)
    assert torch.equal(none.blur(x), x)


def test_point_source_conserves_flux_and_the_unit_factor_scales_it():
    hr = torch.zeros(1, 1, 600, 600)
    hr[0, 0, 299, 302] = 50.0
    for unit in (1.0, NJY_HR / NJY_LR):
        pred = cs.ForwardModel(unit)(padded_log(hr))
        assert pred.shape == (1, 1, 100, 100)
        assert pred.sum().item() == pytest.approx(50.0 * unit, rel=1e-4)
        assert pred.argmax().item() == 49 * 100 + 50  # the block holding the source
        assert (pred > 0.01 * pred.max()).sum() > 9  # spread by the PSF over several LR pixels


def test_gaussian_source_gets_the_expected_width_after_blur_and_binning():
    sigma_source = 9.0  # HR px
    hr = gaussian(600, 300.0, 300.0, sigma_source, 20.0).view(1, 1, 600, 600)
    model = cs.ForwardModel(1.0, psf_lr_fwhm=0.85, psf_hr_fwhm=0.10)
    pred = model(padded_log(hr))[0, 0]
    cy, cx, sigma_lr = moments(pred)
    # blur adds the PSF in quadrature; summing 6 x 6 blocks adds the box variance 1 / 12 LR px^2
    expected = math.sqrt((sigma_source**2 + model.sigma_px**2) / 36 + 1 / 12)
    assert sigma_lr == pytest.approx(expected, rel=0.02)
    assert (cy, cx) == pytest.approx((49.5 + 0.5 / 6, 49.5 + 0.5 / 6), abs=0.05)
    assert pred.sum().item() == pytest.approx(hr.sum().item(), rel=1e-3)


def synthetic_pair(seed=0, noise=0.0, unit=1.0, sky=0.0):
    """HR sky + sources (linear, B=2) and the LR the forward model predicts for them (+ noise)."""
    g = torch.Generator().manual_seed(seed)
    hr = torch.rand(2, 1, 600, 600, generator=g) * 0.2  # faint positive sky texture
    hr[0, 0] += gaussian(600, 200.0, 250.0, 12.0, 40.0) + gaussian(600, 420.0, 400.0, 6.0, 80.0)
    hr[1, 0] += gaussian(600, 300.0, 300.0, 20.0, 15.0)
    seg = (hr > 2.0).float()
    model = cs.ForwardModel(unit)
    lr = model(padded_log(hr)) + sky
    lr = lr + noise * torch.randn(lr.shape, generator=g)
    return hr, seg, lr, model


def test_zero_loss_when_the_sr_is_the_target_pushed_through_the_model():
    hr, seg, lr, model = synthetic_pair(unit=NJY_HR / NJY_LR)
    loss_fn = cs.ConsistencyLoss(model, sigma_lr=0.04, weight=1.0, edge_px=2)
    loss, ratio = loss_fn(padded_log(hr), lr_tensor(lr), F.pad(seg, (84,) * 4))
    assert loss.item() < 1e-6 and ratio.item() == pytest.approx(1.0, abs=1e-4)


def test_noise_in_the_lr_gives_the_huber_floor_and_a_unit_ratio():
    sigma = 3.3
    hr, seg, lr, model = synthetic_pair(noise=sigma, sky=30.0)  # a sky well above 0: no clipping
    loss_fn = cs.ConsistencyLoss(model, sigma_lr=sigma, weight=1.0, edge_px=2)
    loss, ratio = loss_fn(padded_log(hr), lr_tensor(lr), F.pad(seg, (84,) * 4))
    # E[huber(n / sigma)] for unit Gaussian n, delta 1: 0.5 E[n^2; |n|<1] + E[|n| - 0.5; |n|>1]
    n = torch.randn(1_000_000, generator=torch.Generator().manual_seed(1))
    floor = F.huber_loss(n, torch.zeros_like(n), delta=1.0).item()
    assert loss.item() == pytest.approx(floor, rel=0.05)
    # the dataset clips the LR at 0: with the sky near 0 the clipped noise trims the floor
    hr, seg, lr, model = synthetic_pair(noise=sigma)
    clipped, _ = loss_fn(padded_log(hr), lr_tensor(lr), F.pad(seg, (84,) * 4))
    assert 0.75 * floor < clipped.item() < floor
    assert ratio.item() == pytest.approx(1.0, abs=0.03)
    # a 20% flux deficit in the SR raises the loss and shows in the ratio
    deficit = padded_log(hr * 0.8)
    loss_d, ratio_d = loss_fn(deficit, lr_tensor(lr), F.pad(seg, (84,) * 4))
    assert loss_d > loss and ratio_d.item() == pytest.approx(0.8, abs=0.03)


def test_sky_level_pedestals_are_subtracted_before_comparing():
    hr, seg, lr, model = synthetic_pair()
    loss_fn = cs.ConsistencyLoss(model, sigma_lr=1.0, weight=1.0, edge_px=0)
    args = (lr_tensor(lr + 5.0), F.pad(seg, (84,) * 4))  # the LR has a sky pedestal
    base, _ = loss_fn(padded_log(hr), *args)
    lifted, _ = loss_fn(padded_log(hr + 0.3), *args)  # and the SR carries a different one
    assert base.item() < 1e-5 and lifted.item() < 1e-5
    t = loss_fn.terms(padded_log(hr + 0.3), *args)
    # the synthetic HR sky averages 0.1 per pixel: 3.6 per LR pixel, plus the pedestals
    assert t["lr_sky"].tolist() == pytest.approx([5.0 + 3.6, 5.0 + 3.6], abs=0.1)
    assert t["pred_sky"].tolist() == pytest.approx([0.3 * 36 + 3.6] * 2, abs=0.2)


def test_saturated_and_edge_pixels_are_excluded():
    hr, seg, lr, model = synthetic_pair()
    lr[0, 0, 33, 41] = 1e4  # at the clip ceiling: the brightest pixel of its image
    loss_fn = cs.ConsistencyLoss(model, sigma_lr=1.0, weight=1.0, edge_px=3)
    t = loss_fn.terms(padded_log(hr), lr_tensor(lr), F.pad(seg, (84,) * 4))
    valid = t["valid"]
    assert not valid[0, 0, 33, 41] and valid[0, 0, 33, 42]
    assert valid[1].sum() == 94 * 94 - 1  # the edge rings and the brightest pixel of image 1
    assert not valid[:, :, :3].any() and not valid[:, :, -3:].any()
    assert not valid[:, :, :, :3].any() and not valid[:, :, :, -3:].any()
    assert t["residual"][~valid].abs().max() == 0
    loss, _ = loss_fn(padded_log(hr), lr_tensor(lr), F.pad(seg, (84,) * 4))
    assert loss.item() < 1e-6  # the 1e4 pixel does not reach the loss
    assert t["source"].sum() > 0 and not (t["source"] & ~(seg.sum() > 0)).any()


def test_loss_is_differentiable_and_acts_where_the_flux_is():
    hr, seg, lr, model = synthetic_pair(noise=3.3)
    loss_fn = cs.ConsistencyLoss(model, sigma_lr=3.3, weight=1.0, edge_px=2)
    sr = padded_log(hr * 0.9).requires_grad_(True)
    loss, _ = loss_fn(sr, lr_tensor(lr), F.pad(seg, (84,) * 4))
    loss.backward()
    g = sr.grad[:, :, 84:-84, 84:-84].abs()
    src = seg > 0
    assert torch.isfinite(sr.grad).all()
    assert g[src].mean() > 20 * g[~src].mean()  # flux-weighted: sources, hardly the sky


def test_from_config_reads_the_units_from_the_pair_headers(tmp_path):
    for units, hr_njy, lr_njy in (("paper", NJY_HR, NJY_LR), ("njy", 1.0, 1.0)):
        split = write_pair(tmp_path / units / "train", units)
        config = configparser.ConfigParser()
        config.read_dict(
            {
                "DEFAULT": {
                    "hst_path_train": str(split / "hr"),
                    "hsc_path_train": str(split / "lr"),
                },
                "CONSISTENCY": {"lambda_consistency": "2.5", "edge_px": "4"},
            }
        )
        assert cs.ConsistencyLoss.enabled(config)
        loss_fn = cs.ConsistencyLoss.from_config(config, "cpu")
        # a section without psf_lr_fwhm_arcsec / flux_scale gets the values fitted on real pairs
        assert loss_fn.model.unit_factor == pytest.approx(hr_njy / lr_njy * cs.FLUX_SCALE, rel=1e-6)
        assert loss_fn.model.flux_scale == cs.FLUX_SCALE
        assert loss_fn.model.psf_lr_fwhm == cs.PSF_LR_FWHM
        assert loss_fn.sigma_lr == pytest.approx(3.3 / lr_njy, rel=1e-6)
        assert loss_fn.lambda_consistency == 2.5 and loss_fn.edge_px == 4
        assert loss_fn.delta == 1.0 and loss_fn.warmup_steps == cs.WARMUP_STEPS
        assert loss_fn.model.sigma_px == cs.ForwardModel(1.0, psf_lr_fwhm=cs.PSF_LR_FWHM).sigma_px
        params = loss_fn.parameters()
        assert set(params) >= {"lambda_consistency", "consistency_unit_factor"}
        assert params["consistency_psf_lr_fwhm_arcsec"] == cs.PSF_LR_FWHM
        assert params["consistency_flux_scale"] == cs.FLUX_SCALE
    config["CONSISTENCY"]["njy_per_px_lr"] = "10"
    config["CONSISTENCY"]["lr_noise_njy"] = "5"
    config["CONSISTENCY"]["flux_scale"] = "0.8"
    config["CONSISTENCY"]["psf_lr_fwhm_arcsec"] = "0.85"
    config["CONSISTENCY"]["warmup_steps"] = "0"
    loss_fn = cs.ConsistencyLoss.from_config(config, "cpu")
    assert loss_fn.model.unit_factor == pytest.approx(0.08) and loss_fn.sigma_lr == 0.5
    assert loss_fn.model.flux_scale == 0.8 and loss_fn.model.psf_lr_fwhm == 0.85
    assert loss_fn.model.sigma_px == cs.ForwardModel(1.0, psf_lr_fwhm=0.85).sigma_px
    assert loss_fn.parameters()["consistency_flux_scale"] == 0.8
    assert loss_fn.weight == 2.5
    config["CONSISTENCY"]["lambda_consistency"] = "0"
    assert not cs.ConsistencyLoss.enabled(config)
    assert not cs.ConsistencyLoss.enabled(configparser.ConfigParser())


def test_defaults_are_the_values_validated_on_real_pairs():
    """The module defaults are the fitted values (scratch/screening_arms/probe,
    validate_forward_final.txt), not the nominal 0.85" / 1.0 at which the model over-predicts
    every source's LR flux by 13-31% and the loss pulls sources below HST; the bare operator
    keeps flux_scale 1 (it conserves flux, test_point_source_conserves_flux...)."""
    assert cs.PSF_LR_FWHM == 1.0 and cs.FLUX_SCALE == 0.87
    model = cs.ForwardModel(2.0)
    assert model.flux_scale == 1.0 and model.unit_factor == 2.0 and model.psf_lr_fwhm == 1.0
    assert cs.ForwardModel(2.0, flux_scale=0.5).unit_factor == 1.0


def test_weight_ramps_up_over_the_warm_up_and_is_the_lambda_after_it():
    hr, seg, lr, model = synthetic_pair()
    loss_fn = cs.ConsistencyLoss(model, sigma_lr=1.0, weight=2.0, warmup_steps=100)
    assert loss_fn.lambda_consistency == 2.0 and loss_fn.step == 0
    assert loss_fn.weight == pytest.approx(0.02)  # step 0: 1 / 100 of lambda, never exactly 0
    loss_fn.step = 49
    assert loss_fn.weight == pytest.approx(1.0)
    loss_fn.step = 5000
    assert loss_fn.weight == 2.0
    assert cs.ConsistencyLoss(model, sigma_lr=1.0, weight=2.0, warmup_steps=0).weight == 2.0
    assert cs.ConsistencyLoss(model, sigma_lr=1.0).warmup_steps == cs.WARMUP_STEPS


def test_to_linear_inverts_the_dataset_scaling_of_the_lr():
    from neo.data.dataset import SR_HST_HSC_Dataset

    lr = np.random.default_rng(0).normal(0.5, 3.3, (100, 100)).astype(np.float32)
    clipped = SR_HST_HSC_Dataset.clip(lr, use_data=False)[0]
    scaled = torch.as_tensor(SR_HST_HSC_Dataset.ds9_scaling(clipped, offset=1))
    np.testing.assert_allclose(to_linear(scaled).numpy(), clipped, atol=1e-5)
