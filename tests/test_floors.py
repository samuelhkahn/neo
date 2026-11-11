"""Fixed sky floors in place of the paper's clip at zero (neo/data/floors.py)."""

import configparser
import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from astropy.io import fits

from neo.data import floors
from neo.data.dataset import SR_HST_HSC_Dataset
from neo.eval import mosaic, stack
from neo.eval.postprocess import center_crop, to_physical
from neo.eval.predictors import build_predictor
from neo.models.generator_variants import NeoReLUGenerator

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "neo" / "configs"
HR_FLOOR, LR_FLOOR = -0.01, -0.2


@pytest.fixture(autouse=True)
def drop_checkpoints(tmp_path):
    yield
    for path in tmp_path.rglob("*.pt"):
        path.unlink()


def config_with(hr=None, lr=None, generator=None):
    config = configparser.ConfigParser()
    if hr is not None or lr is not None:
        config["DATASET"] = {
            k: str(v) for k, v in (("hr_floor", hr), ("lr_floor", lr)) if v is not None
        }
    if generator:
        config["MODEL"] = {"generator": generator}
    return config


def sky(shape, sigma, seed):
    rng = np.random.default_rng(seed)
    image = rng.normal(0, sigma, shape)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    c = shape[0] / 2
    image += 50 * sigma * np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * (shape[0] / 30) ** 2))
    return image.astype(np.float32)


def write_pair(root, name="p.fits", seed=0):
    for kind, size, sigma in (("hr", 852, 0.002), ("lr", 142, 0.042)):
        (root / kind).mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(sky((size, size), sigma, seed)).writeto(root / kind / name)
    return root


def dataset(root, **floor_kwargs):
    return SR_HST_HSC_Dataset(
        hst_path=str(root / "hr"),
        hsc_path=str(root / "lr"),
        hr_size=[600, 600],
        lr_size=[100, 100],
        transform_type="ds9_scale",
        data_aug=False,
        experiment=None,
        **floor_kwargs,
    )


def test_config_floors_default_to_the_papers_zero_and_must_not_be_positive():
    assert floors.floors(configparser.ConfigParser()) == {"hr_floor": 0.0, "lr_floor": 0.0}
    assert floors.floors(config_with(HR_FLOOR, LR_FLOOR)) == {"hr_floor": -0.01, "lr_floor": -0.2}
    assert floors.floors(config_with(lr=-0.2)) == {"hr_floor": 0.0, "lr_floor": -0.2}
    for bad in ("0.01", "nan"):
        with pytest.raises(ValueError, match="must be <= 0"):
            floors.floors(config_with(hr=bad))


def test_zero_floor_is_the_papers_clip_bit_for_bit(tmp_path):
    x = sky((600, 600), 0.002, 1)
    paper = SR_HST_HSC_Dataset.ds9_scaling(SR_HST_HSC_Dataset.clip(x, use_data=False)[0], offset=1)
    np.testing.assert_array_equal(SR_HST_HSC_Dataset.floor_scaling(x, 0.0), paper)
    root = write_pair(tmp_path)
    default, zero = dataset(root)[0], dataset(root, hr_floor=0.0, lr_floor=0.0)[0]
    for a, b in zip(default, zero, strict=True):
        assert torch.equal(a, b)


def test_floored_dataset_keeps_negative_sky_and_inverts_exactly(tmp_path):
    root = write_pair(tmp_path)
    hst, hsc, hsc_hr, _ = dataset(root, hr_floor=HR_FLOOR, lr_floor=LR_FLOOR)[0]
    paper_hst = dataset(root)[0][0]
    raw_hr = center_crop(fits.getdata(root / "hr" / "p.fits").astype(np.float32), 600)
    expected = np.clip(raw_hr, HR_FLOOR, np.percentile(raw_hr, 99.999))
    back = to_physical(hst.numpy(), HR_FLOOR)
    np.testing.assert_allclose(back, expected, atol=2e-6, rtol=1e-4)
    assert (back < 0).mean() > 0.4  # the sky's negative half survives the round trip
    assert (to_physical(paper_hst.numpy()) >= 0).all()  # and the paper's clip removes it
    # anything at or below the floor is log space -1, the generator's tanh minimum
    edge = np.array([HR_FLOOR - 1, HR_FLOOR, 0, 1], dtype=np.float32)
    np.testing.assert_allclose(SR_HST_HSC_Dataset.floor_scaling(edge, HR_FLOOR)[:2], -1)
    assert hsc.shape == (128, 128) and hsc_hr.shape == (768, 768)
    # the LR input is clipped and scaled from its own floor
    raw_lr = center_crop(fits.getdata(root / "lr" / "p.fits").astype(np.float32), 100)
    lr_back = floors.unscale(center_crop(hsc.numpy(), 100).astype(np.float64), LR_FLOOR)
    np.testing.assert_allclose(
        lr_back, np.clip(raw_lr, LR_FLOOR, np.percentile(raw_lr, 99.999)), atol=2e-5, rtol=1e-4
    )


def test_mosaic_window_input_is_the_datasets_lr_input(tmp_path):
    root = write_pair(tmp_path)
    for lr_floor in (0.0, LR_FLOOR):
        hsc = dataset(root, lr_floor=lr_floor)[0][1]
        window = center_crop(fits.getdata(root / "lr" / "p.fits").astype(np.float32), 100)
        assert torch.equal(mosaic.window_to_input(window, lr_floor)[0, 0], hsc)


def test_window_model_round_trips_negative_sky_and_refuses_the_zero_refill():
    def upsample(x):
        return torch.nn.functional.interpolate(x, scale_factor=6, mode="nearest")

    lr = np.random.default_rng(3).normal(0, 0.04, (1, 100, 100)).astype(np.float32)
    # an identity "generator" on a shared floor: output = clip(LR, floor) upsampled, in HR units
    out = mosaic.make_window_model(upsample, hr_floor=LR_FLOOR, lr_floor=LR_FLOOR)([lr[..., None]])
    clipped = np.clip(lr, LR_FLOOR, np.percentile(lr, 99.999))  # the window's own clip
    expected = np.repeat(np.repeat(clipped, 6, 1), 6, 2)
    np.testing.assert_allclose(out[0, ..., 0], expected[0], atol=1e-4, rtol=1e-3)
    assert (out < 0).mean() > 0.4
    with pytest.raises(ValueError, match="sky floors"):
        mosaic.make_window_model(upsample, negative_noise="paper", hr_floor=HR_FLOOR)


def test_stacks_of_a_floored_run_average_its_own_linear_flux():
    draws = torch.rand(4, 1, 1, 8, 8) * 1.5 - 1
    stacked = stack.stack(draws, HR_FLOOR)
    lin = floors.unscale(draws.double(), HR_FLOOR)
    torch.testing.assert_close(
        floors.unscale(stacked["mean"].double(), HR_FLOOR), lin.mean(0), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(stacked["std"].double(), lin.std(0), atol=1e-6, rtol=1e-5)
    middle = lin.sort(0).values[1:3].mean(0)  # K = 4: the two middle draws, in linear flux
    torch.testing.assert_close(
        floors.unscale(stacked["median"].double(), HR_FLOOR), middle, atol=1e-6, rtol=1e-5
    )
    # a floor of 0 keeps the paper's maps
    torch.testing.assert_close(stack.stack(draws, 0.0)["mean"], stack.stack(draws)["mean"])


def test_signed_scale_keeps_the_ds9_dynamic_range():
    """Floor -> -1, x = 1 -> 0 as in the paper's scale; positive flux keeps the paper's curve up to
    one linear factor, negative flux mirrors it, and scale / unscale invert each other."""
    for floor in (HR_FLOOR, LR_FLOOR):
        x = np.concatenate([[floor], np.linspace(floor, 5, 2001), np.geomspace(1e-6, 1e3, 200)])
        v = floors.scale(x, floor)
        assert v[0] == pytest.approx(-1) and floors.scale(np.array([1.0]), floor)[
            0
        ] == pytest.approx(0)
        np.testing.assert_allclose(floors.unscale(v, floor), x, rtol=1e-9, atol=1e-12)
        assert np.all(np.diff(floors.scale(np.sort(x), floor)) >= 0)  # monotonic
        pos = x[x > 0]
        paper = SR_HST_HSC_Dataset.ds9_scaling(pos, offset=1)
        slope = np.polyfit(paper, floors.scale(pos, floor), 1)[0]
        assert slope == pytest.approx(1 / (1 - floors.s_floor(floor)))
        np.testing.assert_allclose(
            floors.scale(pos, floor), -1 + (paper + 1 - floors.s_floor(floor)) * slope, atol=1e-9
        )
        # mirror: s(-x) = -s(x), so the sky's two halves get the same contrast
        sf, mid = floors.s_floor(floor), floors.scale(np.array([0.0]), floor)[0]
        small = np.array([0.5 * abs(floor)])
        assert floors.scale(small, floor)[0] - mid == pytest.approx(
            mid - floors.scale(-small, floor)[0]
        )
        assert sf < 0


def load_train_module():
    spec = importlib.util.spec_from_file_location("train", ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoints_record_their_floors_and_evaluation_refuses_other_floors(tmp_path):
    train = load_train_module()
    gen, disc = NeoReLUGenerator(1, 1), nn.Linear(2, 2)
    pix2pix = types.SimpleNamespace(
        gen=gen,
        patch_gan=disc,
        gen_opt=torch.optim.Adam(gen.parameters()),
        disc_opt=torch.optim.Adam(disc.parameters()),
        generator_name="neo_relu",
    )
    sky_floors = floors.floors(config_with(HR_FLOOR, LR_FLOOR))
    train.save_checkpoint(tmp_path / "floor.pt", pix2pix, 7, "run", sky_floors)
    train.save_checkpoint(tmp_path / "paper.pt", pix2pix, 7, "run")  # Arm A: no record
    floored = torch.load(tmp_path / "floor.pt", weights_only=True)
    paper = torch.load(tmp_path / "paper.pt", weights_only=True)
    assert floored["hr_floor"] == HR_FLOOR and floored["lr_floor"] == LR_FLOOR
    assert "hr_floor" not in paper
    floor_config = config_with(HR_FLOOR, LR_FLOOR, "neo_relu")
    paper_config = config_with(generator="neo_relu")
    floors.check(floored, floor_config)
    floors.check(paper, paper_config)
    with pytest.raises(ValueError, match="sky floors"):
        floors.check(paper, floor_config)
    with pytest.raises(ValueError, match="sky floors"):
        build_predictor(paper_config, tmp_path / "floor.pt", "cpu")
    build_predictor(floor_config, tmp_path / "floor.pt", "cpu")


def test_floors_refuse_the_arms_that_assume_a_zero_floor():
    for section in ("CONSISTENCY", "NOISE"):
        config = config_with(HR_FLOOR, LR_FLOOR)
        config[section] = {}
        with pytest.raises(ValueError, match=section):
            floors.unsupported_with(floors.floors(config), config)
        floors.unsupported_with(floors.floors(configparser.ConfigParser()), config)


def read_config(path):
    config = configparser.ConfigParser(interpolation=None)
    config.read(path)
    return {s: dict(config[s]) for s in config.sections()} | {"DEFAULT": dict(config.defaults())}


def test_floor_config_is_arm_a_with_only_the_floors_names_and_length_changed():
    base = read_config(CONFIGS / "lux_neo_relu.ini")
    cfg = read_config(CONFIGS / "lux_neo_relu_floor.ini")
    assert cfg["IDENTIFIER"]["identifier"] == '"lux_cosmosweb_i_neo_relu_floor"'
    assert cfg["CHECKPOINT"]["ckpt_dir"] == "${NEO_DATA}/checkpoints/lux_cosmosweb_i_neo_relu_floor"
    assert cfg["COMET_TAG"]["comet_tag"].strip('"').endswith(" neo_relu_floor")
    assert (cfg["DATASET"]["hr_floor"], cfg["DATASET"]["lr_floor"]) == ("-0.01", "-0.2")
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
        ("DATASET", "hr_floor"),
        ("DATASET", "lr_floor"),
    }


def config_diff(base, cfg):
    return {
        (s, k)
        for s in base.keys() | cfg.keys()
        for k in base.get(s, {}).keys() | cfg.get(s, {}).keys()
        if base.get(s, {}).get(k) != cfg.get(s, {}).get(k)
    }


def test_floor_clean_d_config_is_the_floor_config_with_the_clean_conditioning():
    base = read_config(CONFIGS / "lux_neo_relu_floor.ini")
    cfg = read_config(CONFIGS / "lux_neo_relu_floor_cleand.ini")
    name = "lux_cosmosweb_i_neo_relu_floor_cleand"
    assert cfg["IDENTIFIER"]["identifier"] == f'"{name}"'
    assert cfg["CHECKPOINT"]["ckpt_dir"] == "${NEO_DATA}/checkpoints/" + name
    assert cfg["COMET_TAG"]["comet_tag"].strip('"').endswith(" neo_relu_floor_cleand")
    assert cfg["DATASET"]["hsc_hr_8bit"] == "none"
    assert config_diff(base, cfg) == {
        ("IDENTIFIER", "identifier"),
        ("CHECKPOINT", "ckpt_dir"),
        ("COMET_TAG", "comet_tag"),
        ("DATASET", "hsc_hr_8bit"),
    }
    header = (CONFIGS / "lux_neo_relu_floor.ini").read_text().split("[DEFAULT]")[0]
    assert "CONFIG_B=neo/configs/lux_neo_relu_floor_cleand.ini" in header
    assert "train_pair.sbatch" in header


def test_mosaic_refuses_the_zero_refill_for_a_floored_run_before_touching_the_old_one(tmp_path):
    lr = np.zeros((120, 120), np.float32)
    lr_path = tmp_path / "lr.fits"
    fits.HDUList(
        [fits.PrimaryHDU(lr), fits.ImageHDU(np.zeros(lr.shape, np.uint8), name="BAD")]
    ).writeto(lr_path)
    ini = tmp_path / "floor.ini"
    ini.write_text("[DATASET]\nhr_floor = -0.01\nlr_floor = -0.2\n")
    out = tmp_path / "o"
    out.mkdir()
    for name in ("output.fits", "sr_mean.fits", "mosaic.json"):
        (out / name).write_bytes(b"old")
    args = ["sr", "--config", str(ini), "--checkpoint", str(tmp_path / "x.pt")]
    args += ["--lr", str(lr_path), "--out", str(out), "--no-comet", "--overwrite"]
    with pytest.raises(SystemExit, match="negative-noise none"):
        mosaic.main([*args, "--negative-noise", "paper"])
    # a checkpoint that cannot be loaded is refused before the old run is removed, too
    with pytest.raises(Exception):  # noqa: B017 - missing file: any load error
        mosaic.main([*args, "--negative-noise", "none"])
    assert all((out / n).read_bytes() == b"old" for n in ("output.fits", "sr_mean.fits"))


def test_compare_paper_mode_finds_floored_predictions(tmp_path):
    from neo.eval import compare

    for model, cards in (("paper", {}), ("floor", {"NEOHRFL": -0.01, "NEOLRFL": -0.2})):
        (tmp_path / model).mkdir()
        header = fits.Header()
        header.update(cards)
        fits.PrimaryHDU(np.zeros((4, 4), np.float32), header=header).writeto(
            tmp_path / model / "a.fits"
        )
    preds = {m: tmp_path / m for m in ("paper", "floor")}
    assert compare.floored_models(preds, ["a.fits"]) == {"floor": (-0.01, -0.2)}
