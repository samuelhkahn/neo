"""The fixed-set checkpoint sweep (neo/eval/sweep.py) on tiny local checkpoints."""

import configparser
import csv
import json

import numpy as np
import pytest
import torch
from astropy.io import fits
from test_eval_units import NAME, write_pair

from neo.eval import fine_structure, sweep
from neo.models.generator import Pix2PixGenerator
from neo.models.patchgan import PatchGAN


class CheapVGG:
    """Stands in for the VGG perceptual loss so the test needs no pretrained weights."""

    def __init__(self, device, weights):
        pass

    def __call__(self, x, y):
        return ((x - y) ** 2).mean()


def write_run(tmp_path, steps):
    split = write_pair(tmp_path / "val", "paper")
    write_pair(tmp_path / "val", "paper", name="deep_coadd_test_val_00001.fits")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for step in steps:
        torch.manual_seed(step)
        torch.save(
            {
                "step": step,
                "gen": Pix2PixGenerator(1, 1).state_dict(),
                "patch_gan": PatchGAN(2).state_dict(),
            },
            ckpt / f"step_{step:08d}.pt",
        )
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "DEFAULT": {"hst_path_val": str(split / "hr"), "hsc_path_val": str(split / "lr")},
            "CHECKPOINT": {"ckpt_dir": str(ckpt)},
            "VGG_LOSS_WEIGHTS": {"vgg_loss_weights": "[0.0, 1.0, 1.0, 0.0, 0.0]"},
            "IDENTIFIER": {"identifier": '"test_run"'},
            "DATASET": {"hsc_hr_8bit": "wrap"},
        }
    )
    path = tmp_path / "run.ini"
    with open(path, "w") as f:
        config.write(f)
    return path


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_sweep_scores_checkpoints_and_averages(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "VGGLoss", CheapVGG)
    config = write_run(tmp_path, [25000, 50000, 75000])
    out = tmp_path / "out"
    args = ["--config", str(config), "--out", str(out), "--device", "cpu", "--no-comet"]
    sweep.main([*args, "--average", "50000:75000"])
    summary = rows(out / "sweep.csv")
    assert [r["checkpoint"] for r in summary] == [
        "step_00025000",
        "step_00050000",
        "step_00075000",
        "avg_50000-75000",
    ]
    assert all(int(r["n_images"]) == 2 for r in summary)
    assert len(rows(out / "per_image.csv")) == 8
    assert (out / "sweep.png").exists() and (out / "fine_structure.json").exists()

    # a rerun scores only what is new, and refuses other settings in the same directory
    torch.save(
        torch.load(out.parent / "ckpt" / "step_00075000.pt"),
        out.parent / "ckpt" / "step_00100000.pt",
    )
    sweep.main(args)
    summary = rows(out / "sweep.csv")
    assert summary[-1]["checkpoint"] == "step_00100000"
    # step 100000 is a copy of 75000: same images, same dropout masks, same numbers
    by_name = {r["checkpoint"]: r for r in summary}
    for key in ("l1", "segmap_l1", "vgg", "adversarial", "discriminator"):
        assert float(by_name["step_00100000"][key]) == pytest.approx(
            float(by_name["step_00075000"][key]), rel=1e-6
        )
    with pytest.raises(SystemExit, match="other settings"):
        sweep.main([*args, "--seed", "1"])
    # the fine-structure columns: random weights blow up every HST source (an SR far brighter
    # than HST), which is counted, so the pooled summaries are NaN with NaN intervals
    for r in summary:
        counted = sum(int(r[k]) for k in ("fine_n_sources", "fine_n_saturated", "fine_n_blown"))
        assert counted == 2 * 3 and r["fine_n_star"] == "0"
        for key in sweep.FINE_CI:
            lo, value, hi = (float(r[f"{key}{s}"]) for s in ("_lo", "", "_hi"))
            assert (lo <= value <= hi) or np.isnan(value), key

    # a sweep.csv written by an older version of the metric (other columns) is refused before
    # anything is evaluated, whether or not it holds every checkpoint already
    def never(*args, **kwargs):
        raise AssertionError("evaluate() ran before the old sweep.csv was refused")

    monkeypatch.setattr(sweep, "evaluate", never)
    old_rows = [f"step_{s:08d},{s},2,0.1" for s in (25000, 50000, 75000, 100000)]
    for name, listed in (("older", old_rows[:1]), ("older_done", old_rows)):
        older = tmp_path / name
        older.mkdir()
        (older / "sweep.csv").write_text("checkpoint,step,n_images,l1\n" + "\n".join(listed) + "\n")
        with pytest.raises(SystemExit, match="other columns"):
            sweep.main(
                ["--config", str(config), "--out", str(older), "--device", "cpu", "--no-comet"]
            )
        assert not (older / "settings.json").exists()
    # the figure alone still draws such a file (no blown_fraction column) on request
    fig = sweep.sweep_figure(
        [
            {
                "checkpoint": r["checkpoint"],
                "step": r["step"],
                "l1": 0.1,
                "l1_se": 0.01,
                **{
                    k: 0.5
                    for k in (
                        "segmap_l1",
                        "vgg",
                        "adversarial",
                        "discriminator",
                        "k50",
                        "T_band",
                        "saturated_fraction",
                    )
                },
            }
            for r in summary
        ],
        "old",
    )
    assert fig is not None


def test_summarize_writes_exactly_the_columns_main_checks_for():
    fine = fine_structure.Accumulator(fine_structure.Spectra()).result()
    assert list(sweep.summarize("x", 0, [], fine)) == list(sweep.COLUMNS)
    assert sweep.variant_labels(1) == [""] and sweep.variant_labels(8) == [
        "",
        ":median8",
        ":mean8",
    ]


def test_ensemble_adds_stack_rows_and_leaves_the_single_pass_unchanged(monkeypatch, tmp_path):
    """K = 3, not 2: with two draws the median stack is the mean stack by construction
    (neo.eval.stack averages the two middle draws), so only an odd K tells the two rows apart."""
    monkeypatch.setattr(sweep, "VGGLoss", CheapVGG)
    config = write_run(tmp_path, [25000, 50000])
    args = ["--config", str(config), "--device", "cpu", "--no-comet", "--limit", "1"]
    single, ensemble = tmp_path / "single", tmp_path / "ensemble"
    sweep.main([*args, "--out", str(single)])
    sweep.main([*args, "--out", str(ensemble), "--ensemble", "3"])
    plain = rows(single / "sweep.csv")
    stacked = rows(ensemble / "sweep.csv")
    assert [r["checkpoint"] for r in stacked] == [
        "step_00025000",
        "step_00025000:median3",
        "step_00025000:mean3",
        "step_00050000",
        "step_00050000:median3",
        "step_00050000:mean3",
    ]
    # the single pass (k = 0, seed + i) is byte-identical to the K = 1 run
    assert [r for r in stacked if ":" not in r["checkpoint"]] == plain
    by_name = {r["checkpoint"]: r for r in stacked}
    for kind in ("median3", "mean3"):
        row = by_name[f"step_00025000:{kind}"]
        assert row["step"] == "25000" and row["n_images"] == "1"
        for key in sweep.LOSSES:
            assert np.isfinite(float(row[key])), key
        assert "T_band_lo" in row and "blown_fraction" in row
        # a stack of three dropout draws is not either draw
        assert float(row["l1"]) != float(by_name["step_00025000"]["l1"])
    # and the median stack is not the mean stack
    assert float(by_name["step_00025000:median3"]["l1"]) != float(
        by_name["step_00025000:mean3"]["l1"]
    )
    per_image = rows(ensemble / "per_image.csv")
    assert len(per_image) == 6 and {r["checkpoint"] for r in per_image} == set(by_name)
    fine = json.loads((ensemble / "fine_structure.json").read_text())
    assert set(fine) == set(by_name) and "T_band_lo" in fine["step_00025000:mean3"]
    assert json.loads((ensemble / "settings.json").read_text())["ensemble"] == 3
    assert "ensemble" not in json.loads((single / "settings.json").read_text())
    assert (ensemble / "sweep.png").exists()
    # the two directories hold sweeps with other settings
    with pytest.raises(SystemExit, match="other settings"):
        sweep.main([*args, "--out", str(ensemble)])
    with pytest.raises(SystemExit, match="other settings"):
        sweep.main([*args, "--out", str(single), "--ensemble", "3"])
    with pytest.raises(SystemExit, match="at least 1"):
        sweep.main([*args, "--out", str(tmp_path / "zero"), "--ensemble", "0"])
    # in eval mode every pass is the same image: refused, as predict.py refuses --samples
    with pytest.raises(SystemExit, match="needs --mode train"):
        sweep.main([*args, "--out", str(tmp_path / "eval"), "--ensemble", "2", "--mode", "eval"])
    assert not (tmp_path / "eval").exists()

    # a run killed after a checkpoint's single-pass row but before its stack rows: the rerun
    # scores that checkpoint again and writes only the rows that are missing, nothing twice
    sweep_csv, per_csv = ensemble / "sweep.csv", ensemble / "per_image.csv"
    for path in (sweep_csv, per_csv):
        kept = [line for line in path.read_text().splitlines() if "step_00050000:m" not in line]
        path.write_text("\n".join(kept) + "\n")
    fine = json.loads((ensemble / "fine_structure.json").read_text())
    (ensemble / "fine_structure.json").write_text(
        json.dumps({k: v for k, v in fine.items() if not k.startswith("step_00050000:")})
    )
    assert [r["checkpoint"] for r in rows(sweep_csv)] == [r["checkpoint"] for r in stacked[:4]]
    sweep.main([*args, "--out", str(ensemble), "--ensemble", "3"])
    again = rows(sweep_csv)
    assert [r["checkpoint"] for r in again] == [r["checkpoint"] for r in stacked[:4]] + [
        "step_00050000:median3",
        "step_00050000:mean3",
    ]
    assert {r["checkpoint"]: r for r in again} == by_name  # the same seeds: the same numbers
    assert (
        len(rows(per_csv)) == 6
        and len(json.loads((ensemble / "fine_structure.json").read_text())) == 6
    )
    # with every row there the rerun scores nothing
    before = sweep_csv.read_text()
    sweep.main([*args, "--out", str(ensemble), "--ensemble", "3"])
    assert sweep_csv.read_text() == before


def test_images_without_a_wcs_get_a_notice_and_no_star_mask(monkeypatch, tmp_path, capsys):
    config_path = write_run(tmp_path, [25000])
    config = configparser.ConfigParser()
    config.read(config_path)
    hr = tmp_path / "val" / "hr" / NAME
    data, header = fits.getdata(hr, header=True)
    for key in list(header):
        if key.startswith(("CTYPE", "CRVAL", "CRPIX", "PC", "CDELT", "CUNIT", "WCSAXES")):
            del header[key]
    fits.PrimaryHDU(data, header=header).writeto(hr, overwrite=True)
    images = sweep.load_images(
        config, star_options=sweep.star_options(sweep.parse_args(["--config", "x", "--out", "y"]))
    )
    assert len(images) == 2 and all(im["star_mask"] is None and im["n_stars"] == 0 for im in images)
    assert f"{NAME}: no Gaia star mask (no celestial WCS)" in capsys.readouterr().out
    # --keep-stars still looks the stars up (it only decides whether the metric uses them);
    # without star options (no catalog) there is no lookup at all
    keep = sweep.parse_args(["--config", "x", "--out", "y", "--keep-stars"])
    assert sweep.star_options(keep)["radius_arcsec"] == 0.3
    images = sweep.load_images(config, star_options=None)
    assert all(im["star_mask"] is None for im in images) and capsys.readouterr().out == ""


def test_average_of_identical_checkpoints_is_the_checkpoint(tmp_path):
    state = {"gen": Pix2PixGenerator(1, 1).state_dict(), "patch_gan": PatchGAN(2).state_dict()}
    for name in ("a.pt", "b.pt"):
        torch.save(state, tmp_path / name)
    avg = sweep.average_states([tmp_path / "a.pt", tmp_path / "b.pt"])
    for part in ("gen", "patch_gan"):
        for key, value in state[part].items():
            assert torch.allclose(avg[part][key].to(value.dtype), value, atol=1e-6), key
