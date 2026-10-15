"""HST-like noise for the noise-separation arm (neo/noise.py, [NOISE])."""

import configparser

import numpy as np
import pytest
import torch
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from neo import noise
from neo.data.dataset import SR_HST_HSC_Dataset

CORR_SIGMA = 0.8  # px: the synthetic sky is white noise smoothed by this Gaussian


def correlated_sky(shape, sigma, rng, corr=CORR_SIGMA):
    """Gaussian noise with a Gaussian autocorrelation (like drizzled HST sky), std sigma."""
    white = rng.normal(0, 1, shape)
    smooth = gaussian_filter(white, corr, mode="wrap")
    return smooth / smooth.std() * sigma


def write_cutouts(directory, n, sigma=0.3, size=700, seed=0):
    """n HST-like cutouts: correlated sky noise, a few bright Gaussian sources, mean 0."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size]
    paths = []
    for i in range(n):
        sky = correlated_sky((size, size), sigma * (1 + 0.1 * i), rng)
        for _ in range(6):
            cy, cx = rng.uniform(50, size - 50, 2)
            sky += 40 * sigma * np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / 4.0**2)
        path = directory / f"cutout_{i:02d}.fits"
        fits.PrimaryHDU(sky.astype(np.float32)).writeto(path)
        paths.append(path)
    return paths


def lag(x, dy, dx):
    """Autocorrelation at lag (dy, dx) of a (…, H, W) array of zero-mean noise."""
    a = x[..., : x.shape[-2] - dy, : x.shape[-1] - dx]
    b = x[..., dy:, dx:]
    return float((a * b).mean() / (x * x).mean())


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    paths = write_cutouts(tmp_path_factory.mktemp("hst"), 4)
    return noise.measure_template(paths, n_files=3)


def test_log_linear_round_trip_matches_the_dataset_scaling():
    v = torch.tensor([0.0, 1e-4, 0.3, 2.0, 150.0, 999.0])
    expected = SR_HST_HSC_Dataset.ds9_scaling(v.numpy(), offset=1)
    np.testing.assert_allclose(noise.to_log(v).numpy(), expected, rtol=1e-6)
    np.testing.assert_allclose(noise.to_linear(noise.to_log(v)).numpy(), v.numpy(), rtol=1e-5)
    assert noise.to_linear(torch.tensor(-1.0)) == 0  # the dataset's clipped pixels
    assert noise.to_log(torch.zeros(1)).dtype == torch.float32  # dtype preserved (no double)


def test_template_measures_the_sky_spectrum_and_sigma_of_the_first_files(template):
    assert list(template["files"]) == ["cutout_00.fits", "cutout_01.fits", "cutout_02.fits"]
    assert template["k"][0] == 0 and template["k"][-1] <= np.sqrt(2) / 2 + 1e-9
    assert np.all(np.diff(template["k"]) > 0) and np.all(template["power"] >= 0)
    assert template["power"][0] > template["power"][-1]  # red: correlated noise
    acf = template["acf"]
    c = acf.shape[0] // 2
    assert acf[c, c] == pytest.approx(1.0)
    expected_lag1 = np.exp(-1 / (4 * CORR_SIGMA**2))  # Gaussian ACF of width sqrt(2) corr
    assert acf[c, c + 1] == pytest.approx(expected_lag1, abs=0.05)
    assert acf[c + 1, c] == pytest.approx(acf[c, c + 1])  # symmetrised
    assert template["sigma"] == pytest.approx([0.3, 0.33, 0.36], rel=0.05)
    assert template["sigma_median"] == pytest.approx(0.33, rel=0.05)


def test_template_round_trips_through_its_file(template, tmp_path):
    noise.save_template(tmp_path / "t.npz", template)
    loaded = noise.load_template(tmp_path / "t.npz")
    assert set(loaded) == set(template)
    for key in ("k", "power", "acf", "sigma"):
        np.testing.assert_array_equal(loaded[key], template[key])
    assert loaded["sigma_median"] == template["sigma_median"]
    assert isinstance(loaded["sigma_median"], float)


def test_sampled_noise_has_the_requested_sigma_and_the_template_spectrum(template):
    model = noise.NoiseModel(template, "cpu", sigma_default=0.5)
    torch.manual_seed(0)
    sigma = torch.tensor([0.2, 0.4, 0.8])
    n = model.sample((3, 1, 256, 256), sigma)
    assert n.shape == (3, 1, 256, 256) and n.dtype == torch.float32
    assert n.std(dim=(1, 2, 3)).tolist() == pytest.approx(sigma.tolist(), rel=0.03)
    assert n.mean().abs() < 0.01
    c = template["acf"].shape[0] // 2
    x = n.double()
    assert lag(x, 0, 1) == pytest.approx(template["acf"][c, c + 1], abs=0.03)
    assert lag(x, 1, 1) == pytest.approx(template["acf"][c + 1, c + 1], abs=0.03)
    assert lag(x, 0, 4) == pytest.approx(template["acf"][c, c + 4], abs=0.03)
    # a float sigma, and the default sigma (the training median) when none is given
    assert model.sample((1, 1, 128, 128), 0.25).std().item() == pytest.approx(0.25, rel=0.05)
    assert model.sigma_default == 0.5
    assert noise.NoiseModel(template, "cpu").sigma_default == template["sigma_median"]


def test_add_noise_clips_at_zero_and_is_differentiable(template):
    model = noise.NoiseModel(template, "cpu")
    torch.manual_seed(1)
    clean = noise.to_log(torch.full((2, 1, 64, 64), 0.05)).requires_grad_(True)
    out = model.add_noise(clean, 0.3)
    assert out.shape == clean.shape and out.min() >= -1  # linear >= 0, as the dataset's targets
    assert (out == -1).float().mean() > 0.2  # the clip acted: the sky is mostly within a sigma
    assert (out > -1).float().mean() > 0.2
    out.sum().backward()
    unclipped = out > -1
    assert (clean.grad[unclipped] > 0).all() and (clean.grad[~unclipped] == 0).all()


def test_target_sigma_recovers_the_clipped_targets_sky_sigma(template):
    """The HST targets are clipped at 0: half their sky is exactly 0 (a MAD would read ~0)."""
    rng = np.random.default_rng(2)
    sigmas = [0.25, 0.4]
    sky = np.stack([correlated_sky((1, 300, 300), s, rng) for s in sigmas])
    sky[1] += 0.1  # a small positive pedestal (mu = 0.25 sigma)
    seg = np.zeros_like(sky)
    seg[:, :, 100:140, 100:140] = 1  # a "source" region, left out of the sky
    sky[:, :, 100:140, 100:140] += 20
    clipped = torch.as_tensor(np.clip(sky, 0, None), dtype=torch.float32)
    real_log = noise.to_log(clipped)
    model = noise.NoiseModel(template, "cpu")
    got = model.target_sigma(real_log, torch.as_tensor(seg))
    assert got.tolist() == pytest.approx(sigmas, rel=0.05)
    # the simple MAD of the clipped sky would be far too small
    mad = 1.4826 * (clipped[0][seg[0] == 0] - clipped[0][seg[0] == 0].median()).abs().median()
    assert mad < 0.5 * sigmas[0]
    # unclipped data falls back to the MAD
    unclipped = torch.as_tensor(sky, dtype=torch.float32)
    got = noise.clipped_sky_sigma(unclipped, torch.as_tensor(seg) <= 0)
    assert got.tolist() == pytest.approx(sigmas, rel=0.05)


def test_target_sigma_never_hands_the_noise_a_nan(template, capsys):
    """A target without a measurable sky (segmap over the whole crop, or no positive sky pixel)
    gave a nan sigma, hence a nan fake and nan weights, without stopping the run."""
    rng = np.random.default_rng(3)
    sky = np.stack([correlated_sky((1, 200, 200), 0.3, rng) for _ in range(2)])
    real_log = noise.to_log(torch.as_tensor(np.clip(sky, 0, None), dtype=torch.float32))
    seg = torch.zeros_like(real_log)
    seg[0] = 1  # image 0: the segmap covers every pixel
    raw = noise.clipped_sky_sigma(noise.to_linear(real_log), seg <= 0)
    assert torch.isnan(raw[0]) and raw[1].item() == pytest.approx(0.3, rel=0.1)
    model = noise.NoiseModel(template, "cpu", sigma_default=0.7)
    got = model.target_sigma(real_log, seg)
    assert got[0] == got[1] and got[1].item() == pytest.approx(0.3, rel=0.1)  # the batch-mate's
    assert model.n_fallbacks == 1 and "no measurable sky sigma" in capsys.readouterr().out
    assert torch.isfinite(model.add_noise(real_log, got)).all()
    # alone in the batch: the inference default; an all-zero sky is just as unmeasurable
    assert model.target_sigma(real_log[:1], seg[:1]).tolist() == pytest.approx([0.7])
    zero = noise.to_log(torch.zeros(1, 1, 200, 200))
    raw = noise.clipped_sky_sigma(noise.to_linear(zero), torch.ones(1, 1, 200, 200, dtype=bool))
    assert torch.isnan(raw).all()
    assert model.target_sigma(zero, torch.zeros_like(zero)).tolist() == pytest.approx([0.7])
    assert model.n_fallbacks == 3


def test_from_config_measures_caches_and_reads_the_template(tmp_path):
    paths = write_cutouts(tmp_path / "train" / "hr", 3)
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "DEFAULT": {"hst_path_train": str(tmp_path / "train" / "hr")},
            "CHECKPOINT": {"ckpt_dir": str(tmp_path / "ckpt")},
            "NOISE": {"n_files": "2", "vgg_on": "clean"},
        }
    )
    assert noise.NoiseModel.enabled(config)
    model = noise.NoiseModel.from_config(config, "cpu")
    cache = tmp_path / "ckpt" / noise.TEMPLATE_FILE
    assert cache.exists() and list(model.template["files"]) == [p.name for p in paths[:2]]
    assert model.vgg_on == "clean" and model.sigma_default == model.template["sigma_median"]
    # a second run reads the cache (the training files may be gone), and sigma can be overridden
    for p in paths:
        p.unlink()
    config["NOISE"]["sigma"] = "0.7"
    again = noise.NoiseModel.from_config(config, "cpu")
    assert again.sigma_default == 0.7
    np.testing.assert_array_equal(again.template["power"], model.template["power"])
    # an explicit template file, and the switches
    config["NOISE"]["template"] = str(cache)
    config["CHECKPOINT"]["ckpt_dir"] = str(tmp_path / "elsewhere")
    assert noise.NoiseModel.from_config(config, "cpu").sigma_default == 0.7
    # the evaluation tools (measure=False) read the cache or the named template and never
    # measure one: a checkpoint directory copied without its cache stops, nothing is written
    assert noise.NoiseModel.from_config(config, "cpu", measure=False).sigma_default == 0.7
    del config["NOISE"]["template"]
    with pytest.raises(FileNotFoundError, match="noise_template.npz is missing"):
        noise.NoiseModel.from_config(config, "cpu", measure=False)
    assert not (tmp_path / "elsewhere").exists()
    config["NOISE"]["template"] = str(cache)
    config["NOISE"]["add_noise"] = "false"
    assert not noise.NoiseModel.enabled(config)
    assert not noise.NoiseModel.enabled(configparser.ConfigParser())
    config["NOISE"]["vgg_on"] = "sometimes"
    with pytest.raises(ValueError, match="vgg_on"):
        noise.NoiseModel.from_config(config, "cpu")
