"""Source realism (neo/eval/realism.py, compare.py): ghosts, completeness and sky excess."""

import csv

import numpy as np
import pytest
from astropy.io import fits
from conftest import make_tan_wcs
from scipy import ndimage
from test_eval import gaussian, scene_images
from test_eval_units import HR_PIX, HRSCALE, NAME, run_predict, write_pair

from neo.eval import catalogs, compare, postprocess, realism
from neo.eval.subsets import pair_id

SHAPE = (600, 600)
PX_ARCMIN2 = HR_PIX**2 / 3600
THRESHOLD = catalogs.default_threshold(HRSCALE, HR_PIX)  # the paper's, 1.30 nJy per HR px
# (row, column, sigma, amplitude in nJy, axis ratio, angle): three galaxies well apart, off the
# diagonal so that a centroid with x and y swapped would land on empty sky
GALAXIES = [(300, 300, 9.0, 20.0, 1.0, 0), (174, 434, 6.0, 12.0, 0.5, 30), (480, 90, 5.0, 10.0)]
GHOST = (80, 520, 4.0, 20.0)  # far from every galaxy
GHOST_FLUX = 2 * np.pi * 4.0**2 * 20.0
# Two faint galaxies at the detection limit: the first just under the full cut (it crosses it 10%
# brighter), the second just over it (it drops under it 10% fainter)
FAINT = [(130, 150, 8.0, 1.60), (420, 470, 8.0, 1.78)]
SKY_NAME = "deep_coadd_sky_val_00000.fits"


def galaxies(skip=()):
    return sum(gaussian(SHAPE, *g) for i, g in enumerate(GALAXIES) if i not in skip)


def noise(seed=0):
    return np.random.default_rng(seed).normal(0, 0.25, SHAPE)


def measure(hst, srs, margin=realism.MARGIN_ARCSEC):
    """Realism of one cutout as compare.py's default mode measures it: detection on the images
    SEP background subtracted, sky excess on the images as given."""
    prepared = {k: postprocess.subtract_background(v) for k, v in srs.items()}
    hst_prepared = postprocess.subtract_background(hst)
    detected = catalogs.detect_hst(hst_prepared, THRESHOLD, catalogs.NPIXELS)
    return realism.measure_pair(
        NAME,
        hst_prepared,
        prepared,
        detected,
        THRESHOLD,
        catalogs.NPIXELS,
        catalogs.KERNEL_FWHM,
        HR_PIX,
        margin,
        hst_raw=hst,
        srs_raw=srs,
    )


def stats(found, model, groups=None):
    (summary,) = realism.summarize(found, [model], groups)
    return {k: v["value"] for k, v in summary["stats"].items()}


def test_identical_image_is_pure_complete_and_adds_no_sky_flux():
    hst = galaxies() + noise()
    found = measure(hst, {"same": hst.copy()})
    (row,) = found["pairs"]
    assert row["n_hst"] == row["n_sr"] == row["n_recovered"] == row["n_matched"] == 3
    assert row["n_ghost"] == 0 and row["ghost_flux_njy"] == 0 and row["sky_excess_njy"] == 0
    assert row["n_relaxed_only"] == 0
    assert row["margin_px"] == pytest.approx(6.0)
    settings = [row[k] for k in ("threshold_njy", "npixels", "fwhm")]
    assert settings == [THRESHOLD, catalogs.NPIXELS, catalogs.KERNEL_FWHM]
    assert (row["relaxed_threshold_njy"], row["relaxed_npixels"]) == (THRESHOLD / 2, 25)
    assert row["area_arcmin2"] == pytest.approx((600 * HR_PIX / 60) ** 2)
    assert 0 < row["sky_area_arcmin2"] < row["area_arcmin2"]
    s = stats(found, "same")
    assert s["purity"] == 1 and s["completeness"] == 1 and s["ghosts_per_cutout"] == 0
    assert s["sky_excess_njy_per_arcmin2"] == 0
    assert all(r["recovered:same"] == 1 for r in found["hst"])
    # under each SR detection HST holds the same flux, far above its noise
    for r in found["sr"]:
        assert r["hst_flux_njy"] == pytest.approx(r["segment_flux_njy"], rel=0.02)
        assert r["hst_snr"] > 20 and not r["relaxed_only"]
    # kron magnitudes in AB from nJy: the brightest galaxy holds 2 pi sigma^2 amp nJy
    bright = min(r["kron_mag"] for r in found["hst"])
    assert bright == pytest.approx(31.4 - 2.5 * np.log10(2 * np.pi * 81 * 20), abs=0.1)


def test_an_injected_ghost_is_one_ghost_with_its_flux():
    hst = galaxies() + noise()
    found = measure(hst, {"ghosty": hst + gaussian(SHAPE, *GHOST)})
    (row,) = found["pairs"]
    assert (row["n_sr"], row["n_ghost"], row["n_recovered"]) == (4, 1, 3)
    assert row["ghost_flux_njy"] == pytest.approx(GHOST_FLUX, rel=0.1)
    assert row["sky_excess_njy"] == pytest.approx(GHOST_FLUX, rel=1e-3)  # all of it, in the sky
    (ghost,) = [r for r in found["sr"] if r["ghost"]]
    assert (ghost["ycentroid"], ghost["xcentroid"]) == pytest.approx(GHOST[:2], abs=0.5)
    assert ghost["mag"] == pytest.approx(31.4 - 2.5 * np.log10(GHOST_FLUX), abs=0.1)
    assert ghost["hst_labels"] == "" and not ghost["relaxed_only"]
    assert abs(ghost["hst_snr"]) < 4  # HST shows only noise there
    assert all(r["hst_labels"] for r in found["sr"] if not r["ghost"])
    s = stats(found, "ghosty")
    assert s["purity"] == pytest.approx(3 / 4) and s["completeness"] == 1
    assert s["ghosts_per_cutout"] == 1
    assert s["ghosts_per_arcmin2"] == pytest.approx(1 / row["sky_area_arcmin2"])


def test_a_missing_source_lowers_completeness_but_is_no_ghost():
    hst = galaxies() + noise()
    found = measure(hst, {"lossy": galaxies(skip=(2,)) + noise()})
    (row,) = found["pairs"]
    assert (row["n_hst"], row["n_sr"], row["n_ghost"], row["n_recovered"]) == (3, 2, 0, 2)
    s = stats(found, "lossy")
    assert s["completeness"] == pytest.approx(2 / 3) and s["purity"] == 1
    missed = [r for r in found["hst"] if not r["recovered:lossy"]]
    assert len(missed) == 1
    assert (missed[0]["ycentroid"], missed[0]["xcentroid"]) == pytest.approx((480, 90), abs=1)
    assert abs(row["sky_excess_njy"]) < 0.05 * GHOST_FLUX  # the hole is inside HST's footprint


def test_flux_bias_near_the_cut_is_neither_a_ghost_nor_a_miss():
    """A copy of HST 10% too bright invents nothing and one 10% too faint loses nothing, though at
    the full cut alone the first faint galaxy would be a ghost and the second missed."""
    hst = galaxies() + sum(gaussian(SHAPE, *f) for f in FAINT) + noise()
    found = measure(hst, {"bright": 1.1 * hst, "faint": 0.9 * hst})
    rows = {r["model"]: r for r in found["pairs"]}
    assert rows["bright"]["n_hst"] == 4  # the three galaxies and the second faint one
    bright = rows["bright"]
    assert (bright["n_sr"], bright["n_ghost"], bright["n_relaxed_only"]) == (5, 0, 1)
    (loose,) = [r for r in found["sr"] if r["relaxed_only"]]
    assert (loose["ycentroid"], loose["xcentroid"]) == pytest.approx(FAINT[0][:2], abs=1.5)
    assert loose["hst_labels"] == "" and loose["hst_snr"] > 20  # HST shows it, under the cut
    assert (rows["faint"]["n_sr"], rows["faint"]["n_recovered"]) == (3, 4)
    assert stats(found, "faint")["completeness"] == 1 and stats(found, "bright")["purity"] == 1


def test_sky_excess_sees_flux_spread_wider_than_the_background_mesh():
    """Default mode subtracts each image's own SEP background (64 px mesh) before detection, which
    takes out a uniform offset and most of a broad glow; the sky excess differences the images as
    given, so it sees all of either."""
    hst = galaxies() + noise()
    glow = gaussian(SHAPE, 450, 450, 40.0, 0.5)  # 0.5 nJy at its peak, > 5 sigma from galaxies
    found = measure(hst, {"pedestal": hst + 0.01, "glow": hst + glow})
    rows = {r["model"]: r for r in found["pairs"]}
    assert all((r["n_ghost"], r["n_recovered"]) == (0, 3) for r in rows.values())
    n_sky = rows["pedestal"]["sky_area_arcmin2"] / PX_ARCMIN2
    assert rows["pedestal"]["sky_excess_njy"] == pytest.approx(0.01 * n_sky, rel=1e-6)
    assert rows["glow"]["sky_excess_njy"] == pytest.approx(glow.sum(), rel=1e-3)


def test_with_no_hst_source_every_sr_detection_is_a_ghost():
    sky = noise()
    found = measure(sky, {"ghosty": sky + gaussian(SHAPE, *GHOST), "empty": sky.copy()})
    assert found["hst"] == []
    rows = {r["model"]: r for r in found["pairs"]}
    assert (rows["ghosty"]["n_hst"], rows["ghosty"]["n_sr"], rows["ghosty"]["n_ghost"]) == (0, 1, 1)
    assert rows["ghosty"]["sky_area_arcmin2"] == pytest.approx(rows["ghosty"]["area_arcmin2"])
    assert (rows["empty"]["n_sr"], rows["empty"]["n_ghost"]) == (0, 0)
    s = stats(found, "ghosty")
    assert s["purity"] == 0 and s["ghosts_per_cutout"] == 1 and np.isnan(s["completeness"])
    s = stats(found, "empty")  # nothing detected anywhere: no purity, no completeness
    assert np.isnan(s["purity"]) and np.isnan(s["completeness"]) and s["ghosts_per_cutout"] == 0


def test_with_no_sr_source_nothing_is_recovered():
    found = measure(galaxies() + noise(), {"blank": noise(1)})
    (row,) = found["pairs"]
    assert (row["n_sr"], row["n_ghost"], row["n_recovered"]) == (0, 0, 0)
    assert found["sr"] == []
    s = stats(found, "blank")
    assert s["completeness"] == 0 and np.isnan(s["purity"])


def test_the_margin_is_a_disk_of_pixels_matching_binary_dilation():
    d = realism.disk(0.2 / (0.2 / 6))  # 6 px, though 0.2 / (0.2 / 6) is not exactly 6
    assert d.shape == (13, 13) and d[6, 0] and d[0, 6] and not d[0, 0] and d.sum() == 113
    assert realism.disk(0.0).tolist() == [[True]]
    labels = np.zeros((40, 40), int)
    labels[20, 20] = 7
    near = realism.labels_near(labels, [26, 27, 24.4, 0], [20, 20, 24.4, 0], d)
    assert [list(v) for v in near] == [[7], [], [7], []]  # 6 px in, 7 px out, (24, 24) is 5.7 px
    footprint = ndimage.binary_dilation(labels > 0, structure=d)
    ys, xs = np.mgrid[0:40, 0:40]
    near = realism.labels_near(labels, xs.ravel(), ys.ravel(), d)
    np.testing.assert_array_equal([len(v) > 0 for v in near], footprint.ravel())


def test_ab_magnitudes_and_bins():
    mags = realism.ab_mag([1.0, 3631e9, 0.0, -2.0])  # 3631 Jy is AB mag 0
    np.testing.assert_allclose(mags, [31.4, 0, np.nan, np.nan], atol=1e-3)
    assert realism.bin_labels() == ["<21", "21-22", "22-23", "23-24", "24-25", ">=25"]
    assert realism.mag_bin([20.0, 21.0, 24.99, 25.0, 30.0, np.nan]).tolist() == [0, 1, 4, 5, 5, -1]


def synthetic(n_ghost, groups):
    """A found dict of len(n_ghost) cutouts, one model, n_ghost[i] ghosts among 4 detections."""
    pairs = [
        {
            "name": f"c{i}",
            "model": "m",
            **{k: 0.0 for k in realism.PAIR_FIELDS[2:]},
            "n_sr": 4,
            "n_ghost": g,
            "n_matched": 4 - g,
            "area_arcmin2": 0.1,
            "sky_area_arcmin2": 0.09,
            "sky_excess_njy_per_arcmin2": float(g),
        }
        for i, g in enumerate(n_ghost)
    ]
    return {"pairs": pairs, "sr": [], "hst": []}, {f"c{i}": grp for i, grp in enumerate(groups)}


def test_bootstrap_resamples_whole_groups():
    found, groups = synthetic([0, 1, 2, 3, 0, 4], ["a", "b", "c", "d", "e", "f"])
    (alone,) = realism.summarize(found, ["m"], groups)
    st = alone["stats"]["ghosts_per_cutout"]
    assert st["value"] == pytest.approx(10 / 6) and st["ci_lo"] < st["value"] < st["ci_hi"]
    assert alone["n_groups"] == 6
    # one group holding every cutout: every resample is the whole set, so no spread at all
    (together,) = realism.summarize(found, ["m"], {f"c{i}": "all" for i in range(6)})
    for name in ("purity", "ghosts_per_cutout", "ghosts_per_arcmin2", "sky_excess_njy_per_arcmin2"):
        st = together["stats"][name]
        assert st["ci_lo"] == pytest.approx(st["value"]) == pytest.approx(st["ci_hi"]), name
    assert together["n_groups"] == 1
    assert together["stats"]["purity"]["value"] == pytest.approx(14 / 24)
    assert together["stats"]["ghosts_per_arcmin2"]["value"] == pytest.approx(10 / 0.54)  # of sky
    assert together["stats"]["sky_excess_njy_per_arcmin2"]["value"] == pytest.approx(1.5)
    assert np.isnan(together["stats"]["completeness"]["value"])  # no HST sources


def test_bootstrap_median_is_the_median_of_the_resampled_cutouts():
    rng = np.random.default_rng(0)
    values = rng.normal(size=9)
    values[3] = np.nan  # a cutout without sky is left out
    weights = realism.resample_weights([str(i) for i in range(9)], n_boot=50, seed=1)
    lo, hi = realism.median(values, weights)["ci_lo"], realism.median(values, weights)["ci_hi"]
    ok = np.isfinite(values)
    brute = [np.median(np.repeat(values[ok], w[ok].astype(int))) for w in weights]
    assert (lo, hi) == pytest.approx(tuple(np.percentile(brute, [2.5, 97.5])))
    assert realism.median(values, weights)["value"] == pytest.approx(np.median(values[ok]))


def same_catalogs(a, b):
    for x, y in zip([a[0], *a[1].values(), a[2]], [b[0], *b[1].values(), b[2]], strict=True):
        for name in x.colnames:
            np.testing.assert_array_equal(x[name], y[name])


def test_a_given_hst_detection_catalogs_exactly_as_detecting_again():
    hst, srs, lr = scene_images()
    hst = postprocess.subtract_background(hst)
    fresh = catalogs.catalog_set(hst, srs, lr, threshold=0.5)
    given = catalogs.catalog_set(
        hst, srs, lr, threshold=0.5, detected=catalogs.detect_hst(hst, 0.5)
    )
    same_catalogs(fresh, given)


def write_prediction(directory, split, name, image):
    """A prediction as predict.py writes it: 600 px in nJy with the sky and checkpoint cards (those
    of run_predict's checkpoint, so one directory never mixes checkpoints)."""
    hr_header = fits.getheader(split / "hr" / name)
    header = make_tan_wcs(HR_PIX, SHAPE).to_header()
    header.update(BUNIT="nJy", PAIRID=pair_id(hr_header), NEORUN="run", NEOCKPT="step_00000007.pt")
    header.update(NEOSTEP=7, NEOGMODE="eval")
    directory.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(image.astype(np.float32), header=header).writeto(directory / name)


def spy(calls, key, real):
    """`real`, recording each call's arguments in calls[key]."""

    def wrapped(*args, **kwargs):
        calls[key].append((args, dict(kwargs)))
        return real(*args, **kwargs)

    return wrapped


@pytest.mark.parametrize("paper_mode", [True, False])
def test_process_pair_detects_once_with_the_pairs_settings(monkeypatch, tmp_path, paper_mode):
    """compare.process_pair in this process (compare.main runs it in worker processes, which a
    monkeypatch cannot reach): HST is detected once with the pair's settings (the paper's npixels
    and FWHM in paper mode) for both Table 4 and realism, every SR image with the same settings,
    the margin arrives, and the sky excess differences the images before background subtraction."""
    split = write_pair(tmp_path / "val", "njy")
    crop = postprocess.center_crop(fits.getdata(split / "hr" / NAME), 600)
    same = postprocess.paper_clip(crop) if paper_mode else crop  # HST exactly as compare loads it
    preds = {"same": tmp_path / "same", "pedestal": tmp_path / "pedestal"}
    write_prediction(preds["same"], split, NAME, same)
    write_prediction(preds["pedestal"], split, NAME, same + 0.01)
    calls = {"hst": [], "realism": [], "catalog_set": []}
    monkeypatch.setattr(compare, "detect_hst", spy(calls, "hst", catalogs.detect_hst))
    monkeypatch.setattr(realism, "detect_hst", spy(calls, "realism", catalogs.detect_hst))
    monkeypatch.setattr(compare, "catalog_set", spy(calls, "catalog_set", catalogs.catalog_set))
    opts = {
        "factor": 6,
        "npixels": None if paper_mode else catalogs.NPIXELS,
        "threshold": None,
        "nsigma": None,
        "balance_noise": False,
        "paper_mode": paper_mode,
        "realism": True,
        "margin": 0.1,
    }
    name, result, error, settings, found = compare.process_pair((NAME, split, preds, opts))
    assert error is None and result is not None and "error" not in found
    detect = (settings["threshold"], settings["npixels"], settings["fwhm"])
    expected = (71, pytest.approx(1.652, abs=1e-3)) if paper_mode else (100, 3.0)
    assert detect[1:] == expected
    assert [args[1:] for args, _ in calls["hst"]] == [detect]  # once, for Table 4 and realism
    loose = (*realism.relaxed(*detect[:2]), detect[2])
    used = [args[1:] for args, _ in calls["realism"]]
    assert used[0] == loose and used.count(detect) == len(preds) and set(used) <= {detect, loose}
    # Table 4 from that one detection is what catalog_set gives detecting HST itself
    ((args, kwargs),) = calls["catalog_set"]
    assert kwargs.pop("detected") is not None
    same_catalogs(result, catalogs.catalog_set(*args, **kwargs))

    rows = {r["model"]: r for r in found["pairs"]}
    for row in rows.values():
        assert row["margin_px"] == pytest.approx(0.1 / HR_PIX)
        assert (row["threshold_njy"], row["npixels"], row["fwhm"]) == detect
    assert rows["same"]["n_ghost"] == 0 and rows["same"]["n_recovered"] == rows["same"]["n_hst"]
    assert rows["same"]["n_hst"] >= 3 and rows["same"]["sky_excess_njy"] == 0
    n_sky = rows["pedestal"]["sky_area_arcmin2"] / PX_ARCMIN2
    assert rows["pedestal"]["sky_excess_njy"] == pytest.approx(0.01 * n_sky, rel=1e-4)


def write_sky_pair(split, name):
    """A pair showing only sky noise (nJy): no HST source, so Table 4 drops it."""
    rng = np.random.default_rng(9)
    cards = {"LRFILE": "deep_coadd_sky.fits", "LRX0": 3, "LRY0": 5, "SRFACTOR": 6}
    for kind, shape, scale, sigma in (
        ("hr", (852, 852), HR_PIX, 0.25),
        ("lr", (142, 142), 0.2, 3.4),
    ):
        header = make_tan_wcs(scale, shape).to_header()
        header.update(cards, HRSCALE=HRSCALE, BUNIT="nJy")
        data = rng.normal(0, sigma, shape).astype(np.float32)
        fits.PrimaryHDU(data, header=header).writeto(split / kind / name)


def ghost_split(monkeypatch, tmp_path):
    """The test pair and a pure-sky pair, predicted by "neo" (the clipped HST itself, in nJy) and
    "ghosty" (the same plus GHOST); returns (split, compare.py arguments without --out)."""
    split = write_pair(tmp_path / "val", "paper")
    preds = tmp_path / "pred"
    run_predict(monkeypatch, split, preds)
    write_sky_pair(split, SKY_NAME)
    perfect = fits.getdata(preds / NAME)
    ghosty = tmp_path / "ghosty"
    write_prediction(ghosty, split, NAME, perfect + gaussian(SHAPE, *GHOST))
    sky = postprocess.center_crop(fits.getdata(split / "hr" / SKY_NAME), 600)
    write_prediction(ghosty, split, SKY_NAME, sky + gaussian(SHAPE, *GHOST))
    write_prediction(preds, split, SKY_NAME, sky)
    args = ["--split-dir", str(split), "--pred", f"neo={preds}", "--pred", f"ghosty={ghosty}"]
    return split, [*args, "--subset", "all", "--no-comet"]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("paper_mode", [True, False])
def test_compare_counts_ghosts_in_every_pair_and_keeps_table4(monkeypatch, tmp_path, paper_mode):
    split, args = ghost_split(monkeypatch, tmp_path)
    args += ["--paper-mode"] if paper_mode else []
    out = tmp_path / "out"
    compare.main([*args, "--out", str(out)])
    rows = {(r["name"], r["model"]): r for r in read_csv(out / "realism.csv")}
    assert len(rows) == 4  # both pairs, also the one Table 4 drops
    assert {r["name"] for r in read_csv(out / "sources.csv")} == {NAME}
    n_hst = int(rows[NAME, "neo"]["n_hst"])
    assert n_hst >= 3 and int(rows[SKY_NAME, "neo"]["n_hst"]) == 0
    assert int(rows[NAME, "neo"]["n_ghost"]) == 0 and int(rows[NAME, "neo"]["n_recovered"]) == n_hst
    for name in (NAME, SKY_NAME):
        assert int(rows[name, "ghosty"]["n_ghost"]) == 1
        assert float(rows[name, "ghosty"]["ghost_flux_njy"]) == pytest.approx(GHOST_FLUX, rel=0.1)
    summary = {
        (r["model"], r["statistic"]): float(r["value"])
        for r in read_csv(out / "realism_summary.csv")
    }
    assert summary["neo", "purity"] == 1 and summary["neo", "completeness"] == 1
    assert summary["ghosty", "ghosts_per_cutout"] == 1
    assert summary["ghosty", "purity"] == pytest.approx(n_hst / (n_hst + 2))
    hst_rows = read_csv(out / "realism_hst.csv")
    assert len(hst_rows) == n_hst and all(r["recovered:ghosty"] == "1" for r in hst_rows)
    assert len([r for r in read_csv(out / "realism_sr.csv") if r["ghost"] == "1"]) == 2
    report = (out / "realism.md").read_text()
    assert "| neo | " in report and "| ghosty | " in report and "2 cutouts" in report
    assert ("paper mode" in report) == paper_mode
    assert '0.2" (a disk of radius 6 HR px)' in report and "failed" not in report

    # the realism check leaves every Table-4 output as it was
    plain = tmp_path / "plain"
    compare.main([*args, "--out", str(plain), "--no-realism"])
    assert not any(plain.glob("realism*"))
    for fname in ("sources.csv", "table4.csv", "table4.md", "gains.csv", "pairwise.csv"):
        assert (plain / fname).read_text() == (out / fname).read_text(), fname


def test_compare_passes_the_margin_and_groups_to_realism(monkeypatch, tmp_path):
    split, args = ghost_split(monkeypatch, tmp_path)
    (split / "groups.csv").write_text(f"name,group\n{NAME},g\n{SKY_NAME},g\n")
    out = tmp_path / "out"
    compare.main([*args, "--out", str(out), "--realism-margin", "0.1"])
    for row in read_csv(out / "realism.csv"):
        assert float(row["margin_px"]) == pytest.approx(0.1 / HR_PIX)
    report = (out / "realism.md").read_text()
    assert "(a disk of radius 3 HR px)" in report
    assert "resampling whole groups of overlapping cutouts (1 groups" in report
    # ghosty's purity differs between the two cutouts; one group holding both cannot spread it
    summary = {(r["model"], r["statistic"]): r for r in read_csv(out / "realism_summary.csv")}
    purity = {k: float(summary["ghosty", "purity"][k]) for k in ("value", "ci_lo", "ci_hi")}
    assert 0 < purity["value"] < 1
    assert purity["ci_lo"] == pytest.approx(purity["value"]) == pytest.approx(purity["ci_hi"])


class InProcess:
    """Stands in for compare.main's ProcessPoolExecutor, so a monkeypatch reaches the pair work."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, jobs, chunksize=1):
        return map(fn, jobs)


def test_pairs_whose_realism_fails_are_counted_not_hidden(monkeypatch, tmp_path, capsys):
    split, args = ghost_split(monkeypatch, tmp_path)
    monkeypatch.setattr(compare, "ProcessPoolExecutor", InProcess)
    measure_pair = realism.measure_pair

    def fails_on_sky(name, *rest, **kwargs):
        if name == SKY_NAME:
            raise RuntimeError("boom")
        return measure_pair(name, *rest, **kwargs)

    monkeypatch.setattr(realism, "measure_pair", fails_on_sky)
    out = tmp_path / "some"
    compare.main([*args, "--out", str(out)])
    assert {r["name"] for r in read_csv(out / "realism.csv")} == {NAME}
    report = (out / "realism.md").read_text()
    assert "1 cutouts (" in report and "1 more failed (see the log) and are left out" in report
    assert f"{SKY_NAME}: realism: RuntimeError: boom" in capsys.readouterr().out

    def fails(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(realism, "measure_pair", fails)
    out = tmp_path / "none"
    compare.main([*args, "--out", str(out)])
    assert not any(out.glob("realism*")) and (out / "table4.md").exists()
    assert "source realism failed on all 2 pairs" in capsys.readouterr().out
