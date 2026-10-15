"""The two code screening arms in the training loop and the evaluation tools.

Flux consistency ([CONSISTENCY], neo/consistency.py) and noise separation ([NOISE], neo/noise.py)
live behind optional config sections: with both absent every loss Pix2Pix computes must be
bit-identical to the code before the arms existed; predict.py --noise and the sweep must handle a
[NOISE] run.
"""

import configparser

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision
from astropy.io import fits
from test_eval_units import NAME, write_pair
from test_sweep import CheapVGG, rows, write_run
from torchvision.transforms import CenterCrop

import neo.models.vgg19
from neo import noise
from neo.consistency import ConsistencyLoss, ForwardModel
from neo.data.dataset import SR_HST_HSC_Dataset
from neo.eval import predict, sweep
from neo.models.generator import Pix2PixGenerator
from neo.models.patchgan import PatchGAN
from neo.pix2pix import Pix2Pix


@pytest.fixture(autouse=True)
def drop_checkpoints(tmp_path):
    """Delete the checkpoints a test wrote (hundreds of MB each)."""
    yield
    for path in tmp_path.rglob("*.pt"):
        path.unlink()


@pytest.fixture
def random_vgg(monkeypatch):
    vgg19 = torchvision.models.vgg19
    monkeypatch.setattr(neo.models.vgg19.models, "vgg19", lambda pretrained=True: vgg19())


class TinyGenerator(nn.Module):
    """128 -> 768 px, with dropout so that the RNG matters, cheap enough for a CPU test."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.drop = nn.Dropout(0.5)
        self.up = nn.Upsample(scale_factor=6, mode="nearest")
        self.out = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, x, identity_map=False):
        return torch.tanh(self.out(self.up(self.drop(torch.relu(self.conv(x))))))


def original_gen_step(p, real_images, conditioned_images, hsc_hr, seg_map_real):
    """Pix2Pix._gen_step as it was before the arms were added."""
    fake_images = p.gen(conditioned_images)
    fake_images = CenterCrop(600)(fake_images)
    real_images = CenterCrop(600)(real_images)
    seg_map_real = CenterCrop(600)(seg_map_real)
    hsc_hr = CenterCrop(600)(hsc_hr)
    disc_logits = p.patch_gan(fake_images, hsc_hr)
    adversarial_loss = p.adversarial_criterion(
        disc_logits.flatten(), torch.ones_like(disc_logits).flatten()
    )
    recon_loss = p.recon_criterion_l1(fake_images, real_images)
    vgg_loss = p.vgg_criterion(fake_images, real_images)
    segmap_loss = p.l1_loss_with_mask(fake_images, real_images, seg_map_real)
    total_loss = p.lr * (
        p.lambda_adv * adversarial_loss
        + p.lambda_recon * recon_loss
        + p.lambda_vgg * vgg_loss
        + p.lambda_segmap * segmap_loss
    )
    return total_loss, adversarial_loss, recon_loss, vgg_loss, segmap_loss


def original_disc_step(p, real_images, conditioned_images, hsc_hr):
    fake_images = p.gen(conditioned_images).detach()
    fake_images = CenterCrop(600)(fake_images)
    real_images = CenterCrop(600)(real_images)
    hsc_hr = CenterCrop(600)(hsc_hr)
    fake_logits = p.patch_gan(fake_images, hsc_hr)
    real_logits = p.patch_gan(real_images, hsc_hr)
    fake_loss = p.adversarial_criterion(
        fake_logits.flatten(), torch.zeros_like(fake_logits).flatten()
    )
    real_loss = p.adversarial_criterion(
        real_logits.flatten(), torch.ones_like(real_logits).flatten()
    )
    return real_loss + fake_loss, fake_logits, real_logits


def batch(seed=0):
    """(hr, lr, hsc_hr, seg) like the dataset's, in log space, batch of 2."""
    g = torch.Generator().manual_seed(seed)
    linear = torch.rand(2, 1, 768, 768, generator=g) * 0.5
    linear[:, :, 300:340, 300:340] += 30.0
    hr = noise.to_log(linear)
    lr = noise.to_log(torch.rand(2, 1, 128, 128, generator=g) * 3 + 0.5)
    hsc_hr = lr.repeat_interleave(6, -1).repeat_interleave(6, -2)
    seg = torch.zeros_like(hr)
    seg[:, :, 290:350, 290:350] = 1
    return hr, lr, hsc_hr, seg


def pix2pix_with(random_vgg, **arms):
    torch.manual_seed(0)
    return Pix2Pix(
        1, 1, "cpu", vgg_loss_weights=[0.0, 1.0, 0.0, 0.0, 0.0], generator=TinyGenerator(), **arms
    )


def consistency_for_test():
    return ConsistencyLoss(ForwardModel(1.0), sigma_lr=1.0, weight=10.0, edge_px=2, warmup_steps=0)


def test_default_losses_are_bit_identical_to_the_code_before_the_arms(random_vgg):
    p = pix2pix_with(random_vgg)
    assert p.consistency is None and p.noise is None
    hr, lr, hsc_hr, seg = batch()
    torch.manual_seed(1)
    new = p._gen_step(hr, lr, hsc_hr, seg)
    torch.manual_seed(1)
    old = original_gen_step(p, hr, lr, hsc_hr, seg)
    assert len(new) == 6 and new[5] == {}
    for a, b in zip(new[:5], old, strict=True):
        assert torch.equal(a, b)
    torch.manual_seed(2)
    new_d = p._disc_step(hr, lr, hsc_hr, seg)
    torch.manual_seed(2)
    old_d = original_disc_step(p, hr, lr, hsc_hr)
    for a, b in zip(new_d, old_d, strict=True):
        assert torch.equal(a, b)
    # the training step reports the same six values
    torch.manual_seed(3)
    losses = p.training_step(hr, lr, hsc_hr, seg, "generator")
    assert len(losses) == 6 and losses[5] == {}


def test_consistency_adds_its_weighted_term_and_leaves_the_other_losses_alone(random_vgg):
    p = pix2pix_with(random_vgg, consistency=consistency_for_test())
    q = pix2pix_with(random_vgg)
    hr, lr, hsc_hr, seg = batch()
    torch.manual_seed(1)
    with_arm = p._gen_step(hr, lr, hsc_hr, seg)
    torch.manual_seed(1)
    without = q._gen_step(hr, lr, hsc_hr, seg)
    for a, b in zip(with_arm[1:5], without[1:5], strict=True):
        assert torch.equal(a, b)
    extras = with_arm[5]
    assert set(extras) == {"flux_loss", "flux_ratio"}
    assert torch.isfinite(extras["flux_loss"]) and extras["flux_loss"] > 0
    expected = without[0] + p.lr * (10.0 * extras["flux_loss"])
    assert with_arm[0].item() == pytest.approx(expected.item(), rel=1e-6)
    with_arm[0].backward()  # the term reaches the generator
    assert all(param.grad is not None for param in p.gen.parameters())


def test_noise_keeps_the_l1_terms_clean_and_judges_the_noisy_fake(random_vgg, tmp_path):
    from test_noise import write_cutouts

    template = noise.measure_template(write_cutouts(tmp_path / "hst", 2), 2)
    hr, lr, hsc_hr, seg = batch()
    for vgg_on in ("noisy", "clean"):
        model = noise.NoiseModel(template, "cpu", vgg_on=vgg_on)
        p = pix2pix_with(random_vgg, noise=model)
        q = pix2pix_with(random_vgg)
        torch.manual_seed(1)
        with_arm = p._gen_step(hr, lr, hsc_hr, seg)
        torch.manual_seed(1)
        without = q._gen_step(hr, lr, hsc_hr, seg)
        # same generator output (same seed before the forward): L1 and segmap L1 are unchanged
        assert torch.equal(with_arm[2], without[2]) and torch.equal(with_arm[4], without[4])
        assert not torch.equal(with_arm[1], without[1])  # the discriminator saw s + n
        assert torch.equal(with_arm[3], without[3]) == (vgg_on == "clean")
        sigma = with_arm[5]["noise_sigma"]
        assert set(with_arm[5]) == {"noise_sigma"} and 0 < sigma < 1
        with_arm[0].backward()
        assert all(param.grad is not None for param in p.gen.parameters())
    # the discriminator step judges the noisy fake too (different logits from the clean one)
    torch.manual_seed(2)
    noisy_d = p._disc_step(hr, lr, hsc_hr, seg)
    torch.manual_seed(2)
    clean_d = q._disc_step(hr, lr, hsc_hr, seg)
    assert torch.equal(noisy_d[2], clean_d[2]) and not torch.equal(noisy_d[1], clean_d[1])
    # both arms together: consistency on the clean output, noise for the discriminator
    both = pix2pix_with(random_vgg, noise=model, consistency=consistency_for_test())
    torch.manual_seed(1)
    losses = both._gen_step(hr, lr, hsc_hr, seg)
    assert set(losses[5]) == {"noise_sigma", "flux_loss", "flux_ratio"}
    assert torch.equal(losses[2], without[2])
    assert torch.isfinite(losses[0])


def noise_config(path, split, template_path, ckpt_dir):
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "DEFAULT": {
                "hst_path_train": str(split / "hr"),
                "hsc_path_train": str(split / "lr"),
                "hst_path_val": str(split / "hr"),
                "hsc_path_val": str(split / "lr"),
            },
            "CHECKPOINT": {"ckpt_dir": str(ckpt_dir)},
            "VGG_LOSS_WEIGHTS": {"vgg_loss_weights": "[0.0, 1.0, 1.0, 0.0, 0.0]"},
            "IDENTIFIER": {"identifier": '"noise_run"'},
            "NOISE": {"template": str(template_path), "sigma": "0.4"},
        }
    )
    with open(path, "w") as f:
        config.write(f)
    return path


def test_predict_writes_the_clean_or_the_noisy_output_and_records_the_choice(tmp_path):
    split = write_pair(tmp_path / "val", "njy")
    template = noise.measure_template([split / "hr" / NAME], 1)
    noise.save_template(tmp_path / "t.npz", template)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    torch.manual_seed(0)
    torch.save({"gen": Pix2PixGenerator(1, 1).state_dict(), "step": 5}, ckpt / "latest.pt")
    config = noise_config(tmp_path / "run.ini", split, tmp_path / "t.npz", ckpt)
    args = ["--config", str(config), "--checkpoint", str(ckpt / "latest.pt")]
    args += ["--split-dir", str(split), "--device", "cpu", "--no-comet"]
    predict.main([*args, "--out", str(tmp_path / "clean")])
    predict.main([*args, "--out", str(tmp_path / "noisy"), "--noise", "hst"])
    clean, h_clean = fits.getdata(tmp_path / "clean" / NAME, header=True)
    noisy, h_noisy = fits.getdata(tmp_path / "noisy" / NAME, header=True)
    assert h_clean["NEONOISE"] == "none" and h_noisy["NEONOISE"] == "hst"
    assert clean.shape == noisy.shape == (600, 600)
    diff = noisy - clean
    assert np.std(diff) == pytest.approx(0.4, rel=0.25)  # ~ the config sigma (clip at 0 aside)
    assert np.all(noisy >= 0)
    # a prediction made with another noise choice is not mixed in silently
    with pytest.raises(SystemExit, match="NEONOISE"):
        predict.main([*args, "--out", str(tmp_path / "clean"), "--noise", "hst"])
    # a file written before the NEONOISE card existed is a clean prediction: a default run
    # reuses it as it is (predict.LEGACY_CARDS), --noise hst still refuses it
    with fits.open(tmp_path / "clean" / NAME, mode="update") as hdul:
        del hdul[0].header["NEONOISE"]
    mtime = (tmp_path / "clean" / NAME).stat().st_mtime_ns
    predict.main([*args, "--out", str(tmp_path / "clean")])
    assert (tmp_path / "clean" / NAME).stat().st_mtime_ns == mtime
    assert "NEONOISE" not in fits.getheader(tmp_path / "clean" / NAME)
    with pytest.raises(SystemExit, match="NEONOISE"):
        predict.main([*args, "--out", str(tmp_path / "clean"), "--noise", "hst"])


def test_sweep_scores_a_noise_run_with_the_noisy_fake_for_the_discriminator(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "VGGLoss", CheapVGG)
    config_path = write_run(tmp_path, [25000])
    plain = configparser.ConfigParser()
    plain.read(config_path)
    template = noise.measure_template([tmp_path / "val" / "hr" / NAME], 1)
    noise.save_template(tmp_path / "t.npz", template)
    plain["NOISE"] = {"template": str(tmp_path / "t.npz")}
    with open(tmp_path / "noise.ini", "w") as f:
        plain.write(f)
    base = ["--device", "cpu", "--no-comet", "--limit", "1"]
    sweep.main(["--config", str(config_path), "--out", str(tmp_path / "plain"), *base])
    sweep.main(["--config", str(tmp_path / "noise.ini"), "--out", str(tmp_path / "noisy"), *base])
    a = rows(tmp_path / "plain" / "sweep.csv")[0]
    b = rows(tmp_path / "noisy" / "sweep.csv")[0]
    for key in ("l1", "segmap_l1", "vgg", "k50"):  # the clean output is scored
        assert float(a[key]) == pytest.approx(float(b[key]), rel=1e-6)
    for key in ("adversarial", "discriminator"):  # judged with noise, as training does
        assert float(a[key]) != pytest.approx(float(b[key]), rel=1e-6)
    # the sweep never measures a template of its own: a run whose cache was not copied with its
    # checkpoints (and names no template) stops instead of scoring with one from local pairs
    del plain["NOISE"]["template"]
    with open(tmp_path / "nocache.ini", "w") as f:
        plain.write(f)
    with pytest.raises(FileNotFoundError, match="noise_template.npz is missing"):
        sweep.main(["--config", str(tmp_path / "nocache.ini"), "--out", str(tmp_path / "x"), *base])
    assert not (tmp_path / "ckpt" / noise.TEMPLATE_FILE).exists()


def test_tiny_generator_is_a_stand_in_for_the_real_one():
    """The real generator, discriminator and dataset shapes the arms rely on."""
    torch.manual_seed(0)
    gen, disc = TinyGenerator(), PatchGAN(2)
    y = gen(torch.randn(1, 1, 128, 128))
    assert y.shape == (1, 1, 768, 768) and y.abs().max() < 1
    assert disc(CenterCrop(600)(y), torch.zeros(1, 1, 600, 600)).shape == (1, 1, 37, 37)
    x = SR_HST_HSC_Dataset.ds9_scaling(np.array([0.0, 1.0]), offset=1)
    assert x[0] == -1 and x[1] == 0
