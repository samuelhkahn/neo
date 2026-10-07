"""Stacking several draws of a stochastic model (neo/eval/stack.py, predict.py --samples)."""

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from astropy.io import fits  # noqa: E402
from test_eval_units import NAME, NJY_HR, write_pair  # noqa: E402

from neo.data.dataset import SR_HST_HSC_Dataset  # noqa: E402
from neo.eval import postprocess, predict, stack  # noqa: E402


def test_log_linear_round_trip_matches_the_dataset_scaling():
    v = torch.tensor([0.0, 1e-4, 0.3, 2.0, 150.0], dtype=torch.float64)
    x = stack.to_log(v)
    expected = SR_HST_HSC_Dataset.ds9_scaling(v.numpy(), offset=1)
    np.testing.assert_allclose(x.numpy(), expected, rtol=1e-12)
    np.testing.assert_allclose(
        stack.to_linear(x).numpy(), SR_HST_HSC_Dataset.ds9_unscaling(expected, offset=1)
    )
    np.testing.assert_allclose(stack.to_linear(x).numpy(), v.numpy(), atol=1e-12)


def test_mean_is_taken_in_linear_flux_and_median_is_a_middle_draw():
    flux = torch.tensor([0.0, 1.0, 10.0], dtype=torch.float64).view(3, 1)  # 3 draws, 1 pixel
    s = stack.stack(stack.to_log(flux).float())
    assert stack.to_linear(s["mean"]).item() == pytest.approx(11 / 3, rel=1e-5)
    assert stack.to_linear(s["median"]).item() == pytest.approx(1.0, rel=1e-5)
    assert torch.equal(s["single"], stack.to_log(flux).float()[0])
    assert s["std"].item() == pytest.approx(torch.std(flux).item(), rel=1e-5)
    # a log-space mean would give the geometric-like mean, far below the flux mean
    assert stack.to_linear(stack.to_log(flux).mean(0)).item() < 1.0


def test_even_median_averages_the_two_middle_draws_in_flux():
    flux = torch.tensor([0.0, 1.0, 3.0, 50.0], dtype=torch.float64).view(4, 1)
    s = stack.stack(stack.to_log(flux))
    assert stack.to_linear(s["median"]).item() == pytest.approx(2.0, rel=1e-9)


def test_one_draw_stacks_to_itself():
    draws = torch.rand(1, 2, 1, 5, 5) - 1
    s = stack.stack(draws)
    for kind in stack.IMAGES:
        torch.testing.assert_close(s[kind], draws[0], rtol=0, atol=1e-6)
    assert (s["std"] == 0).all()


def test_stack_figure_has_all_panels():
    rng = np.random.default_rng(0)
    hr = rng.uniform(-1, 0, (768, 768))
    draws = torch.from_numpy(hr + rng.normal(0, 0.05, (4, 768, 768)))
    s = {k: v.numpy() for k, v in stack.stack(draws).items()}
    fig = stack.stack_figure("x", rng.uniform(-1, 0, (128, 128)), hr, s, 4)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles[0] == "LR" and titles[1].startswith("single   L1 ")
    assert "single - median" in titles and "median - HST" in titles and "HST" in titles
    matplotlib.pyplot.close(fig)


def run_stacked(monkeypatch, split, out, samples, generator="diffusion", *extra):
    """predict.main with a sampler that returns the HR target plus noise of std 0.02 (log space)."""
    dataset = SR_HST_HSC_Dataset(
        hst_path=str(split / "hr"),
        hsc_path=str(split / "lr"),
        hr_size=[600, 600],
        lr_size=[100, 100],
        transform_type="ds9_scale",
        data_aug=False,
        experiment=None,
    )
    target = dataset[0][0]

    def build(config, checkpoint, device, mode="eval"):
        def sample(lr, cond):
            noise = torch.randn(len(lr), 1, *target.shape) * 0.02
            return target[None, None] + noise

        return sample

    monkeypatch.setattr(predict, "build_predictor", build)
    ckpt = split.parent / "run" / "step_00000007.pt"
    if not ckpt.exists():
        ckpt.parent.mkdir()
        torch.save({"step": 7}, ckpt)
    config = split.parent / f"{generator}.ini"
    config.write_text(f"[MODEL]\ngenerator = {generator}\n")
    predict.main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(ckpt),
            "--split-dir",
            str(split),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--no-comet",
            "--samples",
            str(samples),
            *extra,
        ]
    )
    return target


def test_predict_writes_each_stack_as_a_prediction_directory(monkeypatch, tmp_path):
    split = write_pair(tmp_path / "val", "paper")
    out = tmp_path / "pred"
    target = run_stacked(monkeypatch, split, out, 5)
    l1 = {}
    for kind in stack.KINDS:
        data, header = fits.getdata(out / kind / NAME, header=True)
        assert data.shape == (600, 600) and header["BUNIT"] == "nJy"
        assert header["NEOSTACK"] == kind and header["NEOSAMPL"] == 5
        assert header["HRNJYPX"] == pytest.approx(NJY_HR)
        if kind != "std":
            l1[kind] = header["L1LOG"]
            assert (out / kind / "predictions.csv").exists()
    assert not (out / NAME).exists()
    # stacking averages the independent noise down: mean and median beat one draw
    assert l1["mean"] < 0.6 * l1["single"] and l1["median"] < 0.8 * l1["single"]
    # std is the spread of the draws in nJy: about d(flux)/d(log) * 0.02 per pixel
    std = fits.getdata(out / "std" / NAME)
    slope = np.log(1001) * (stack.to_linear(target.double()).numpy() + 1e-3)
    expected = postprocess.center_crop(slope, 600) * 0.02 * NJY_HR
    assert np.median(std / expected) == pytest.approx(1.0, abs=0.25)

    run_stacked(monkeypatch, split, out, 5)  # same checkpoint and K: reused
    with pytest.raises(SystemExit, match="exists but came from"):
        run_stacked(monkeypatch, split, out, 3)


def test_samples_refused_for_a_deterministic_generator(monkeypatch, tmp_path):
    split = write_pair(tmp_path / "val", "paper")
    with pytest.raises(SystemExit, match="deterministic"):
        run_stacked(monkeypatch, split, tmp_path / "pred", 4, "neo")
    run_stacked(
        monkeypatch, split, tmp_path / "pred", 2, "neo", "--gen-mode", "train", "--batch-size", "1"
    )
    assert fits.getheader(tmp_path / "pred" / "median" / NAME)["NEOGMODE"] == "train"
