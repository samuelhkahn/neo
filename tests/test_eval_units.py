"""Evaluation units and --paper-mode: pairs in the paper's units (NJYPERPX) and older nJy pairs."""

import csv

import numpy as np
import pytest
from astropy.io import fits
from conftest import make_tan_wcs
from test_eval import block_sum, gaussian

from neo.data.dataset import SR_HST_HSC_Dataset
from neo.eval import catalogs, compare, postprocess, predict

HR_PIX = 0.2 / 6
HRSCALE = 152.7566058238  # nJy per e-/s at AB zeropoint 25.94
NJY_HR = HRSCALE * (HR_PIX / 0.03) ** 2  # e-/s per 0.03" px -> nJy per HR px (pairs.py)
NJY_LR = 10 ** (-0.4 * (27.0 - 31.4)) * (0.2 / 0.168) ** 2  # HSC count per 0.168" px -> nJy
NAME = "deep_coadd_test_val_00000.fits"


def sky_njy(seed=5):
    """HR (852 px) and LR (142 px) cutouts of the same sky in nJy per pixel."""
    rng = np.random.default_rng(seed)
    shape = (852, 852)
    hr = (
        gaussian(shape, 426, 426, 9.0, 20.0)
        + gaussian(shape, 300, 560, 6.0, 12.0, 0.5, 30)
        + gaussian(shape, 560, 300, 5.0, 10.0, 0.7, 70)
    )
    hr = hr + rng.normal(0, 0.25, shape)
    lr = block_sum(hr, 6) + rng.normal(0, 3.4, (142, 142))
    return hr, lr


def write_pair(split, units, name=NAME):
    """One pair as neo.preprocess.pairs writes it: --units paper, or nJy without NJYPERPX."""
    hr, lr = sky_njy()
    cards = {"LRFILE": "deep_coadd_test.fits", "LRX0": 7, "LRY0": 11, "SRFACTOR": 6}
    hr_cards = {**cards, "HRSCALE": HRSCALE, "HRZPAB": 25.94, "BUNIT": "nJy"}
    lr_cards = {**cards, "BUNIT": "nJy"}
    if units == "paper":
        hr, lr = hr / NJY_HR, lr / NJY_LR
        hr_cards.update(UNITS="paper", BUNIT="e-/s per 0.03as px", NJYPERPX=NJY_HR)
        lr_cards.update(UNITS="paper", BUNIT="HSC count (ZP 27) per 0.168as px", NJYPERPX=NJY_LR)
    for kind, data, scale, extra in (("hr", hr, HR_PIX, hr_cards), ("lr", lr, 0.2, lr_cards)):
        (split / kind).mkdir(parents=True, exist_ok=True)
        header = make_tan_wcs(scale, data.shape).to_header()
        header.update(extra)
        fits.PrimaryHDU(data.astype(np.float32), header=header).writeto(split / kind / name)
    return split


@pytest.fixture
def paper_split(tmp_path):
    return write_pair(tmp_path / "paper" / "val", "paper")


@pytest.fixture
def njy_split(tmp_path):
    return write_pair(tmp_path / "njy" / "val", "njy")


def test_njy_per_px_defaults_to_njy_for_pairs_without_the_card():
    assert postprocess.njy_per_px(fits.Header()) == 1.0
    assert postprocess.njy_per_px(fits.Header({"NJYPERPX": 188.5})) == 188.5


def test_cutouts_in_paper_units_load_as_the_same_njy_images(paper_split, njy_split):
    for paper_mode in (False, True):
        for kind, size in (("hr", 600), ("lr", 100)):
            a, _ = compare.load_cutout(paper_split / kind / NAME, size, paper_mode)
            b, _ = compare.load_cutout(njy_split / kind / NAME, size, paper_mode)
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6 * np.abs(b).max())


def test_paper_clip_is_the_training_datasets_clip(paper_split):
    stored = postprocess.center_crop(fits.getdata(paper_split / "hr" / NAME), 600)
    hst, _ = compare.load_cutout(paper_split / "hr" / NAME, 600, paper_mode=True)
    assert hst.min() == 0 and np.mean(hst == 0) > 0.3  # negative sky noise is gone
    assert hst.max() == pytest.approx(np.percentile(stored, 99.999) * NJY_HR, rel=1e-6)
    expected = SR_HST_HSC_Dataset.clip(stored.astype(np.float32), use_data=False)[0] * NJY_HR
    np.testing.assert_allclose(hst, expected, rtol=1e-6)


def run_process(monkeypatch, split, paper_mode, sr):
    """compare.process on one pair; returns what it passes to catalog_set."""
    seen = {}

    def fake_catalog_set(hst, srs, lr, **kwargs):
        seen.update(hst=hst, srs=srs, lr=lr, **kwargs)
        return None

    pred_dir = split.parent / "pred"
    pred_dir.mkdir(exist_ok=True)
    fits.PrimaryHDU(sr.astype(np.float32)).writeto(pred_dir / NAME, overwrite=True)
    monkeypatch.setattr(compare, "catalog_set", fake_catalog_set)
    opts = {
        "factor": 6,
        "npixels": None if paper_mode else catalogs.NPIXELS,
        "threshold": None,
        "nsigma": None,
        "balance_noise": False,
        "paper_mode": paper_mode,
    }
    name, result, error, settings = compare.process((NAME, split, {"m": pred_dir}, opts))
    assert error is None and settings == {k: seen[k] for k in settings}
    return seen


def test_paper_mode_settings_clipping_and_no_background_subtraction(monkeypatch, paper_split):
    sr = np.abs(np.random.default_rng(0).normal(1.0, 0.3, (600, 600)))
    got = run_process(monkeypatch, paper_split, True, sr)
    assert got["npixels"] == 71
    assert got["fwhm"] == pytest.approx(1.652, abs=1e-3)  # the paper's 3x3 kernel width on the sky
    assert got["lr_fwhm"] == pytest.approx(1.652, abs=1e-3)  # as wide on the sky as 3 px at 0.168"
    assert got["threshold"] == pytest.approx(catalogs.PAPER_THRESHOLD_CPS * NJY_HR, rel=1e-9)
    for kind, size, image in (("hr", 600, got["hst"]), ("lr", 100, got["lr"])):
        stored = postprocess.center_crop(fits.getdata(paper_split / kind / NAME), size)
        clipped = postprocess.paper_clip(stored) * (NJY_HR if kind == "hr" else NJY_LR)
        np.testing.assert_array_equal(image, clipped)  # clipped, nothing subtracted
    np.testing.assert_allclose(got["srs"]["m"], sr.astype(np.float32))  # SR as predicted


def test_default_mode_subtracts_background_in_njy(monkeypatch, paper_split):
    sr = np.abs(np.random.default_rng(0).normal(1.0, 0.3, (600, 600)))
    got = run_process(monkeypatch, paper_split, False, sr)
    assert (got["npixels"], got["fwhm"], got["lr_fwhm"]) == (100, 3.0, None)
    assert got["threshold"] == pytest.approx(1.3036, rel=1e-4)
    stored = postprocess.center_crop(fits.getdata(paper_split / "hr" / NAME), 600)
    expected = postprocess.subtract_background(stored.astype(np.float64) * NJY_HR)
    np.testing.assert_allclose(got["hst"], expected)
    assert got["hst"].min() < 0  # unclipped
    np.testing.assert_allclose(got["srs"]["m"], postprocess.subtract_background(sr), atol=1e-6)


def test_old_njy_pairs_are_measured_as_before(monkeypatch, njy_split):
    sr = np.abs(np.random.default_rng(0).normal(1.0, 0.3, (600, 600)))
    got = run_process(monkeypatch, njy_split, False, sr)
    hr = fits.getdata(njy_split / "hr" / NAME)
    lr = fits.getdata(njy_split / "lr" / NAME)
    np.testing.assert_array_equal(
        got["hst"], postprocess.subtract_background(postprocess.center_crop(hr, 600))
    )
    np.testing.assert_array_equal(
        got["lr"], np.asarray(postprocess.center_crop(lr, 100), dtype=np.float64)
    )
    assert got["threshold"] == pytest.approx(catalogs.default_threshold(HRSCALE, HR_PIX))


def test_paper_mode_refuses_options_it_sets():
    base = ["--split-dir", "x", "--pred", "m=y", "--out", "z", "--paper-mode"]
    for extra in (
        ["--npixels", "100"],
        ["--threshold", "1.5"],
        ["--nsigma", "3"],
        ["--min-ellipticity", "0.1"],
        ["--balance-noise"],
        ["--threshold", "0"],  # an explicit zero is still an override
        ["--nsigma", "0"],
        ["--npixels", "0"],
    ):
        with pytest.raises(SystemExit):
            compare.resolve_mode(compare.parse_args(base + extra))
    args = compare.parse_args(base)
    compare.resolve_mode(args)
    assert args.min_ellipticity == 0 and args.npixels is None
    args = compare.parse_args(base[:-1])
    compare.resolve_mode(args)
    assert (args.npixels, args.min_ellipticity) == (100, 0.1)


def test_paper_mode_flags_predictions_not_made_in_train_mode():
    args = compare.parse_args(["--split-dir", "x", "--pred", "m=y", "--out", "z", "--paper-mode"])
    checkpoints = {"a": "r/s.pt/1/train", "b": "r/s.pt/1/eval", "c": "r/s.pt/1/None"}
    lines = compare.prediction_lines(args, checkpoints)
    assert lines[0] == "Predictions: a = r/s.pt/1/train; b = r/s.pt/1/eval; c = r/s.pt/1/None."
    assert len(lines) == 2 and "for b (eval), c (None):" in lines[1]
    args.paper_mode = False
    assert compare.prediction_lines(args, checkpoints) == lines[:1]


def run_predict(monkeypatch, split, out, *extra):
    """predict.main with a generator that returns the HR target exactly (in log space)."""
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
        return lambda lr, cond: target[None, None].repeat(len(lr), 1, 1, 1)

    monkeypatch.setattr(predict, "build_predictor", build)
    ckpt = split.parent / "run" / "step_00000007.pt"
    if not ckpt.exists():
        import torch

        ckpt.parent.mkdir()
        torch.save({"step": 7}, ckpt)
        (split.parent / "train.ini").write_text("[MODEL]\ngenerator = neo\n")
    predict.main(
        [
            "--config",
            str(split.parent / "train.ini"),
            "--checkpoint",
            str(ckpt),
            "--split-dir",
            str(split),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--no-comet",
            *extra,
        ]
    )
    return fits.getdata(out / NAME, header=True)


@pytest.mark.parametrize("units", ["paper", "njy"])
def test_predictions_are_written_in_njy(monkeypatch, tmp_path, units):
    split = write_pair(tmp_path / units / "val", units)
    out = tmp_path / units / "pred"
    sr, header = run_predict(monkeypatch, split, out)
    njy = NJY_HR if units == "paper" else 1.0
    assert header["BUNIT"] == "nJy" and header["NEOGMODE"] == "eval"
    assert header["HRNJYPX"] == pytest.approx(njy)
    stored = postprocess.center_crop(fits.getdata(split / "hr" / NAME), 600)
    # the perfect generator gives back the training target: the clipped HR, now in nJy (up to
    # float32 rounding in log space, ~1e-10 stored units near zero)
    expected = postprocess.paper_clip(stored) * njy
    np.testing.assert_allclose(sr, expected, rtol=1e-5, atol=1e-9 * njy)
    run_predict(monkeypatch, split, out)  # same checkpoint, mode, units and sky: reused
    with pytest.raises(SystemExit):  # another generator mode must not reuse them
        run_predict(monkeypatch, split, out, "--gen-mode", "train", "--batch-size", "1")


def test_compare_rejects_predictions_left_in_paper_units(paper_split, njy_split):
    for split, bunit, ok in ((paper_split, "e-/s per 0.03as px", False), (njy_split, "", True)):
        pred = split.parent / "old_pred"
        pred.mkdir()
        header = fits.Header({"BUNIT": bunit, "PAIRID": "deep_coadd_test.fits:7:11"})
        fits.PrimaryHDU(np.zeros((600, 600), np.float32), header=header).writeto(pred / NAME)
        if ok:
            compare.verify_predictions(split, {"m": pred}, [NAME])
        else:
            with pytest.raises(SystemExit, match="not nJy"):
                compare.verify_predictions(split, {"m": pred}, [NAME])


def read_sources(out):
    with open(out / "sources.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_compare_end_to_end_in_paper_units(monkeypatch, paper_split):
    preds = paper_split.parent / "pred"
    run_predict(monkeypatch, paper_split, preds)
    args = ["--split-dir", str(paper_split), "--pred", f"neo={preds}", "--subset", "all"]
    out = paper_split.parent / "paper_mode"
    compare.main([*args, "--out", str(out), "--no-comet", "--paper-mode"])
    report = (out / "table4.md").read_text()
    assert "paper mode" in report and "npixels 71" in report and "FWHM 1.652 px" in report
    assert "q (paper code)" in report and (out / "paper_q.csv").exists()
    # predicted in eval mode: named, and flagged as not the paper's generator mode
    assert "Predictions: neo = run/step_00000007.pt/7/eval." in report
    assert "Not the paper's generator mode for neo (eval)" in report
    rows = read_sources(out)
    assert len(rows) >= 3
    assert all(float(r["hst_kron_flux"]) > 0 for r in rows)
    # SR (predicted, nJy) and HST (clipped, nJy) are the same image here: zero bias
    for p in ("R_e", "FWHM", "flux", "q"):
        assert np.allclose([float(r[f"neo:{p}"]) for r in rows], 0, atol=1e-4), p
    # LR in nJy: its fluxes agree with HST's (a unit slip would be off by ~80x)
    lr_flux = np.median([float(r["lr:flux"]) for r in rows])
    assert abs(lr_flux) < 0.3

    out = paper_split.parent / "default_mode"
    compare.main([*args, "--out", str(out), "--no-comet"])
    report = (out / "table4.md").read_text()
    assert "default mode" in report and "threshold 1.304 nJy" in report
    assert "Predictions: neo" in report and "Not the paper's generator mode" not in report
    assert not (out / "paper_q.csv").exists()
    rows = read_sources(out)
    assert abs(np.median([float(r["neo:flux"]) for r in rows])) < 0.1
