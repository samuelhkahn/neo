"""Gaia star masks (neo/eval/stars.py) and their use in the fine-structure metric, the realism
check and compare.py."""

import json

import numpy as np
import pytest
from astropy.io import fits
from conftest import make_tan_wcs
from test_eval import gaussian
from test_eval_realism import GALAXIES, GHOST, SHAPE, galaxies, measure, noise, read_csv
from test_eval_units import HR_PIX, run_predict, write_pair
from test_sweep import CheapVGG, rows, write_run

from neo.eval import compare, fine_structure, realism, stars, sweep

DT = stars.LSST_EPOCH - stars.HST_EPOCH  # 21.2 years


def write_catalog(path, rows, box=stars.BOX):
    """A catalog CSV as `fetch` writes it; rows are (ra, dec, pmra, pmdec, parallax, gmag)."""
    lines = ["# test catalog", f"# box: {','.join(str(v) for v in box)}", ",".join(stars.COLUMNS)]
    for ra, dec, pmra, pmdec, parallax, gmag in rows:
        lines.append(f"{ra:.7f},{dec:.7f},{pmra},{pmdec},{parallax},{gmag},2016.0")
    path.write_text("\n".join(lines) + "\n")
    return path


def header_852(crval=(150.0, 2.0)):
    header = make_tan_wcs(HR_PIX, (852, 852), crval=crval).to_header()
    header["NAXIS1"] = header["NAXIS2"] = 852
    return header


def test_the_shipped_catalog_reads_and_covers_cosmos():
    cat = stars.load_catalog()
    assert len(cat) > 5000 and cat.box == stars.BOX
    assert cat.covers(150.1, 2.2) and not cat.covers(140.0, 2.2)
    assert np.all(np.isfinite(cat.column("ra"))) and np.all(cat.column("ref_epoch") == 2016.0)
    motion = cat.motion_arcsec(stars.HST_EPOCH, stars.LSST_EPOCH)
    assert 0.15 < np.median(motion) < 0.25 and 0.1 < np.mean(motion > 0.5) < 0.16


def test_a_known_proper_motion_lands_where_expected(tmp_path):
    # pmra* = +100 mas/yr moves the star east (lower x in a north-up image), pmdec = -50 south
    catalog = write_catalog(tmp_path / "cat.csv", [(150.0, 2.0, 100.0, -50.0, 1.0, 15.0)])
    header = header_852()
    at_hst = stars.star_positions(header, stars.HST_EPOCH, catalog=catalog)
    at_lsst = stars.star_positions(header, stars.LSST_EPOCH, catalog=catalog)
    at_ref = stars.star_positions(header, 2016.0, catalog=catalog)
    assert at_hst.shape == at_lsst.shape == (1, 2)
    assert at_ref[0] == pytest.approx((425.5, 425.5), abs=1e-3)  # CRPIX 426.5 is 1-based
    dx = (at_lsst - at_hst)[0, 0] * HR_PIX * 1000 / DT  # mas/yr along x
    dy = (at_lsst - at_hst)[0, 1] * HR_PIX * 1000 / DT
    assert (dx, dy) == pytest.approx((-100.0, -50.0), rel=1e-3)
    # the HST-epoch position lies 11.7 years before 2016 along the same track
    expected = at_ref + (at_lsst - at_ref) * (stars.HST_EPOCH - 2016.0) / (
        stars.LSST_EPOCH - 2016.0
    )
    np.testing.assert_allclose(at_hst, expected, atol=1e-6)
    # in the central 600 px crop the same star sits 126 px lower in both axes
    cropped = stars.star_positions(header, stars.HST_EPOCH, (600, 600), catalog=catalog)
    np.testing.assert_allclose(cropped, at_hst - 126, atol=1e-9)
    tracks = stars.star_tracks(header, (600, 600), catalog=catalog)
    assert len(tracks) == 1 and tracks.notice is None
    assert tracks.motion[0] == pytest.approx(np.hypot(100, 50) * DT / 1000, rel=1e-6)
    assert tracks.pixel_arcsec == pytest.approx(HR_PIX, rel=1e-9)


def test_the_mask_covers_both_epochs_with_the_larger_radius(tmp_path):
    fast = (150.0, 2.0, 100.0, -50.0, 1.0, 15.0)  # moves 2.37": radius 1.5 x that = 3.55"
    slow = (150.0 + 300 * HR_PIX / 3600, 2.0, 1.0, 0.0, 2.0, 18.0)  # 300 px east, 0.02": 0.3"
    catalog = write_catalog(tmp_path / "cat.csv", [fast, slow])
    header = header_852()
    mask = stars.star_mask(header, (852, 852), catalog=catalog)
    tracks = stars.star_tracks(header, (852, 852), catalog=catalog)
    assert mask.shape == (852, 852) and len(tracks) == 2
    radii = tracks.radii_px()
    r = 1.5 * np.hypot(100, 50) * DT / 1000 / HR_PIX
    assert radii[0] == pytest.approx(r, rel=1e-6) and radii[1] == pytest.approx(9.0, rel=1e-6)
    for x, y in zip([*tracks.x_hst, *tracks.x_lsst], [*tracks.y_hst, *tracks.y_lsst], strict=True):
        assert mask[int(round(y)), int(round(x))]
    # the fast star's two disks (centres 71 px apart) overlap: between 1.3 and 2 disk areas,
    # all east of column 270; the slow star's one disk of radius 9 px sits around column 126
    assert np.pi * r**2 * 1.3 < mask[:, 270:].sum() < 2 * np.pi * r**2
    assert mask[:, :270].sum() == pytest.approx(np.pi * 9**2, rel=0.05)
    assert mask[425, 126] and not mask[0, 0] and not mask[851, 851]
    # a larger minimum radius and a smaller motion factor change the radii as asked
    wide = stars.star_mask(header, (852, 852), radius_arcsec=5.0, catalog=catalog)
    assert wide.sum() > mask.sum() and wide[425, 126 + 100]
    narrow = stars.star_mask(header, (852, 852), motion_factor=1.0, catalog=catalog)
    assert narrow.sum() < mask.sum()


def test_cutouts_without_wcs_or_outside_the_box_get_no_mask(tmp_path):
    catalog = write_catalog(tmp_path / "cat.csv", [(150.0, 2.0, 0.0, 0.0, 1.0, 15.0)])
    plain = fits.Header({"NAXIS1": 600, "NAXIS2": 600})
    tracks = stars.star_tracks(plain, (600, 600), catalog=catalog)
    assert len(tracks) == 0 and "no celestial WCS" in tracks.notice
    assert not stars.star_mask(plain, (600, 600), catalog=catalog).any()
    assert stars.star_positions(plain, 2020.0, catalog=catalog).shape == (0, 2)
    far = header_852(crval=(10.0, -30.0))
    tracks = stars.star_tracks(far, (600, 600), catalog=catalog)
    assert len(tracks) == 0 and "outside the catalog box" in tracks.notice
    assert not stars.star_mask(far, (600, 600), catalog=catalog).any()
    # inside the box but no star near: no notice, no mask
    near = header_852(crval=(150.5, 2.5))
    tracks = stars.star_tracks(near, (600, 600), catalog=catalog)
    assert len(tracks) == 0 and tracks.notice is None and tracks.mask().sum() == 0


def test_a_star_just_outside_the_crop_still_masks_its_disk(tmp_path):
    # 303 px east of the centre: outside the 600 px crop (edge at 300) but its 9 px disk reaches in
    catalog = write_catalog(
        tmp_path / "cat.csv", [(150.0 + 303 * HR_PIX / 3600, 2.0, 0.0, 0.0, 1.0, 15.0)]
    )
    mask = stars.star_mask(header_852(), (600, 600), catalog=catalog)
    assert mask.any() and mask[300, 0] and not mask[300, 20]
    # 320 px east: the disk ends 11 px outside; not in the crop at all
    catalog = write_catalog(
        tmp_path / "cat.csv", [(150.0 + 320 * HR_PIX / 3600, 2.0, 0.0, 0.0, 1.0, 15.0)]
    )
    assert not stars.star_mask(header_852(), (600, 600), catalog=catalog).any()


def test_fine_structure_skips_sources_inside_the_exclusion_mask():
    rng = np.random.default_rng(0)
    truth = sum(gaussian((600, 600), *g[:4]) for g in GALAXIES)  # three bright sources
    hst = truth * 20 + rng.normal(0, 0.3, (600, 600))
    spectra = fine_structure.Spectra()
    plain = fine_structure.HstSources(hst, spectra)
    assert len(plain.positions) == 3 and plain.n_excluded == 0
    mask = np.zeros((600, 600), dtype=bool)
    mask[290:310, 290:310] = True  # the first galaxy (row 300, column 300)
    masked = fine_structure.HstSources(hst, spectra, exclude=mask)
    assert len(masked.positions) == 2 and masked.n_excluded == 1
    assert all(not (295 < x < 305 and 295 < y < 305) for x, y in masked.positions)
    acc = fine_structure.Accumulator(spectra)
    acc.add(masked, truth * 20)
    assert acc.result()["n_excluded"] == 1 and acc.result()["n_sources"] == 2


def test_realism_sets_aside_detections_and_sources_inside_the_star_mask():
    hst = galaxies() + noise()
    ghosty = hst + gaussian(SHAPE, *GHOST)
    mask = np.zeros(SHAPE, dtype=bool)
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    mask |= (xx - GHOST[1]) ** 2 + (yy - GHOST[0]) ** 2 <= 9**2  # the injected ghost
    mask |= (xx - GALAXIES[2][1]) ** 2 + (yy - GALAXIES[2][0]) ** 2 <= 9**2  # the third galaxy
    plain = measure(hst, {"ghosty": ghosty})["pairs"][0]
    assert (plain["n_hst"], plain["n_ghost"], plain["n_star_sr"], plain["n_gaia_stars"]) == (
        3,
        1,
        0,
        None,  # no catalog lookup was made
    )
    found = measure(hst, {"ghosty": ghosty}, star_mask=mask, n_stars=2)
    (row,) = found["pairs"]
    assert (row["n_hst"], row["n_star_hst"], row["n_gaia_stars"]) == (2, 1, 2)
    assert (row["n_sr"], row["n_ghost"], row["n_star_sr"], row["n_matched"]) == (2, 0, 2, 2)
    assert row["n_recovered"] == 2 and row["ghost_flux_njy"] == 0 and row["stars_set_aside"] == 1
    starred = [r for r in found["sr"] if r["near_star"]]
    assert len(starred) == 2 and all(not r["ghost"] and not r["relaxed_only"] for r in starred)
    assert [r["near_star"] for r in found["hst"]].count(1) == 1
    # the masked disks are neither source nor sky: the sky shrinks by the ghost's disk (the
    # galaxy's disk lies inside its own dilated segment, which was never sky)
    lost = plain["sky_area_arcmin2"] - row["sky_area_arcmin2"]
    assert lost == pytest.approx(np.pi * 9**2 * HR_PIX**2 / 3600, rel=0.1)
    # of the ghost's flux only its wings beyond the 9 px disk (8%) still reach the sky excess
    assert 0 < row["sky_excess_njy"] < 0.1 * plain["sky_excess_njy"]
    # --keep-stars (set_aside=False): the same flags and counts of flagged sources, but every
    # number of the plain run (the ghost, the sky, its excess) and the completeness bins too
    kept = measure(hst, {"ghosty": ghosty}, star_mask=mask, n_stars=2, set_aside=False)
    (krow,) = kept["pairs"]
    assert (krow["n_star_sr"], krow["n_star_hst"], krow["n_gaia_stars"]) == (2, 1, 2)
    assert krow["stars_set_aside"] == 0
    for key in ("n_hst", "n_sr", "n_ghost", "n_matched", "n_recovered", "ghost_flux_njy"):
        assert krow[key] == plain[key], key
    assert krow["sky_area_arcmin2"] == plain["sky_area_arcmin2"]
    assert krow["sky_excess_njy"] == plain["sky_excess_njy"]
    assert [r["near_star"] for r in kept["sr"]].count(1) == 2
    assert [r["ghost"] for r in kept["sr"] if r["near_star"]].count(1) == 1
    assert [r["near_star"] for r in kept["hst"]].count(1) == 1
    (aside,) = realism.summarize(found, ["ghosty"])
    (counted,) = realism.summarize(kept, ["ghosty"])
    bins = [f"completeness_{b}" for b in realism.bin_labels()]
    assert sum(aside["stats"][b]["n"] for b in bins) == 2
    assert sum(counted["stats"][b]["n"] for b in bins) == 3
    assert aside["stars_set_aside"] and not counted["stars_set_aside"]
    assert aside["stars_checked"] and counted["stars_checked"]
    lines = "\n".join(realism.report_lines([counted], 0.2, [6.0]))
    assert "2 in 1 of the 1 cutouts" in lines and "Counted like any other source" in lines
    lines = "\n".join(realism.report_lines([aside], 0.2, [6.0]))
    assert "Set aside from every count above: 1 HST sources" in lines
    # a cutout nobody checked against the catalog says so instead of claiming it holds no star
    (unchecked,) = realism.summarize(measure(hst, {"ghosty": ghosty}), ["ghosty"])
    assert not unchecked["stars_checked"] and unchecked["n_gaia_stars"] == 0
    text = "\n".join(realism.report_lines([unchecked], 0.2, [6.0]))
    assert "Gaia DR3 stars (neo/eval/stars.py): not checked" in text
    assert "none in these cutouts" not in text


class InProcess:
    """compare.main's ProcessPoolExecutor, in this process, so the monkeypatched catalog path
    reaches the pair work."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, jobs, chunksize=1):
        return map(fn, jobs)


def test_compare_excludes_sources_near_gaia_stars_unless_asked(monkeypatch, tmp_path):
    """A catalog star on the central galaxy of the test pair: that source is flagged, left out
    of Table 4 by default and counted with --keep-stars; the realism files set it aside."""
    split = write_pair(tmp_path / "val", "paper")
    preds = tmp_path / "pred"
    run_predict(monkeypatch, split, preds)
    # the pair's WCS is centred on (150, 2) and its brightest galaxy sits at the centre
    catalog = write_catalog(tmp_path / "cat.csv", [(150.0, 2.0, 0.0, 0.0, 1.0, 15.0)])
    monkeypatch.setattr(stars, "CATALOG", catalog)
    monkeypatch.setattr(compare, "ProcessPoolExecutor", InProcess)
    args = ["--split-dir", str(split), "--pred", f"neo={preds}", "--subset", "all", "--no-comet"]
    out = tmp_path / "out"
    compare.main([*args, "--out", str(out)])
    sources = read_csv(out / "sources.csv")
    flagged = [r for r in sources if r["near_gaia_star"] == "1"]
    assert len(flagged) == 1 and len(sources) >= 3
    assert (float(flagged[0]["hst_flux"])) == max(float(r["hst_flux"]) for r in sources)
    table = {(r["image"], r["parameter"]): int(r["n"]) for r in read_csv(out / "table4.csv")}
    assert table["neo", "flux"] == len(sources) - 1
    report = (out / "table4.md").read_text()
    assert f"{len(sources)} HST-detected sources, 1 inside Gaia DR3 star masks excluded" in report
    realism = read_csv(out / "realism.csv")[0]
    assert (realism["n_star_hst"], realism["n_star_sr"], realism["n_gaia_stars"]) == ("1", "1", "1")
    assert int(realism["n_hst"]) == len(sources) - 1 and realism["n_ghost"] == "0"
    md = (out / "realism.md").read_text()
    assert "Gaia DR3 stars (neo/eval/stars.py): 1 in 1 of the 1 cutouts" in md
    assert (
        "Set aside from every count above: 1 HST sources, and SR detections per model: neo 1" in md
    )
    hst_rows = read_csv(out / "realism_hst.csv")
    assert [r["near_star"] for r in hst_rows].count("1") == 1

    masks = "a disk of max(0.3\", 1.5 x the star's motion between epochs 2004.3 and 2025.5)"
    assert masks in report
    assert "flagged and set aside from Table 4 and the realism counts" in report
    summary = {(r["model"], r["statistic"]): r for r in read_csv(out / "realism_summary.csv")}
    assert int(summary["neo", "completeness"]["n"]) == len(sources) - 1
    assert realism["stars_set_aside"] == "1"

    # --keep-stars: the source is still flagged everywhere, and counted everywhere
    kept = tmp_path / "kept"
    compare.main([*args, "--out", str(kept), "--keep-stars", "--star-radius", "0.5"])
    sources_kept = read_csv(kept / "sources.csv")
    assert [r["near_gaia_star"] for r in sources_kept].count("1") == 1
    table = {(r["image"], r["parameter"]): int(r["n"]) for r in read_csv(kept / "table4.csv")}
    assert table["neo", "flux"] == len(sources)
    report = (kept / "table4.md").read_text()
    assert "1 inside Gaia DR3 star masks counted (--keep-stars; near_gaia_star in sources.csv)" in (
        report
    )
    assert 'a disk of max(0.5", 1.5 x' in report  # the radius the run used, not the default
    assert "counted like any other (--keep-stars)" in report
    assert "Gaia star masking off" not in report
    realism_kept = read_csv(kept / "realism.csv")[0]
    assert (realism_kept["n_star_hst"], realism_kept["n_star_sr"]) == ("1", "1")
    assert (realism_kept["n_gaia_stars"], realism_kept["stars_set_aside"]) == ("1", "0")
    assert int(realism_kept["n_hst"]) == len(sources) and realism_kept["n_ghost"] == "0"
    md = (kept / "realism.md").read_text()
    assert "Gaia DR3 stars (neo/eval/stars.py): 1 in 1 of the 1 cutouts" in md
    assert "Counted like any other source (--keep-stars): 1 HST sources" in md
    assert "counted like any other (--keep-stars)" in md  # the intro line carries the options
    assert [r["near_star"] for r in read_csv(kept / "realism_hst.csv")].count("1") == 1
    summary = {(r["model"], r["statistic"]): r for r in read_csv(kept / "realism_summary.csv")}
    assert int(summary["neo", "completeness"]["n"]) == len(sources)
    assert float(summary["neo", "completeness"]["value"]) == 1


def test_compare_without_a_star_nearby_flags_nothing(monkeypatch, tmp_path):
    split = write_pair(tmp_path / "val", "paper")
    preds = tmp_path / "pred"
    run_predict(monkeypatch, split, preds)
    catalog = write_catalog(tmp_path / "cat.csv", [(150.5, 2.5, 0.0, 0.0, 1.0, 15.0)])
    monkeypatch.setattr(stars, "CATALOG", catalog)
    monkeypatch.setattr(compare, "ProcessPoolExecutor", InProcess)
    args = ["--split-dir", str(split), "--pred", f"neo={preds}", "--subset", "all", "--no-comet"]
    compare.main([*args, "--out", str(tmp_path / "out")])
    sources = read_csv(tmp_path / "out" / "sources.csv")
    assert all(r["near_gaia_star"] == "0" for r in sources)
    assert "0 inside Gaia DR3 star masks excluded" in (tmp_path / "out" / "table4.md").read_text()
    assert (
        "Gaia DR3 stars (neo/eval/stars.py): none" in (tmp_path / "out" / "realism.md").read_text()
    )


def test_a_missing_catalog_stops_compare_and_sweep_before_any_work(monkeypatch, tmp_path, capsys):
    """Without the shipped catalog (a checkout that lacks it) the run stops at once with the
    remedy and writes nothing; --keep-stars still runs, with no star masks, and says so."""
    split = write_pair(tmp_path / "val", "paper")
    preds = tmp_path / "pred"
    run_predict(monkeypatch, split, preds)
    missing = tmp_path / "missing" / "gaia.csv"
    monkeypatch.setattr(stars, "CATALOG", missing)
    monkeypatch.setattr(compare, "ProcessPoolExecutor", InProcess)
    with pytest.raises(FileNotFoundError, match="Gaia catalog .*gaia.csv not found: run"):
        stars.load_catalog()
    args = ["--split-dir", str(split), "--pred", f"neo={preds}", "--subset", "all", "--no-comet"]
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="Gaia catalog .* not found: run .*or pass --keep-stars"):
        compare.main([*args, "--out", str(out)])
    assert not out.exists()
    compare.main([*args, "--out", str(out), "--keep-stars"])
    assert "Gaia stars not checked (--keep-stars)" in capsys.readouterr().out
    sources = read_csv(out / "sources.csv")
    assert all(r["near_gaia_star"] == "0" for r in sources)
    report = (out / "table4.md").read_text()
    assert "Gaia DR3 stars not checked (no catalog; --keep-stars)" in report
    assert "no star masks were built and every source is counted (--keep-stars)" in report
    assert "inside Gaia DR3 star masks" not in report
    md = (out / "realism.md").read_text()
    assert "Gaia DR3 stars (neo/eval/stars.py): not checked" in md and "none in these" not in md
    assert read_csv(out / "realism.csv")[0]["n_gaia_stars"] == ""

    # the sweep: the same stop before the dataset, catalog, generator and output directory
    monkeypatch.setattr(sweep, "VGGLoss", CheapVGG)
    config = write_run(tmp_path / "run", [25000])
    base = ["--config", str(config), "--device", "cpu", "--no-comet", "--limit", "1"]
    sweep_out = tmp_path / "sweep"
    with pytest.raises(SystemExit, match="Gaia catalog .* not found: run .*or pass --keep-stars"):
        sweep.main([*base, "--out", str(sweep_out)])
    assert not sweep_out.exists()
    sweep.main([*base, "--out", str(sweep_out), "--keep-stars"])
    printed = capsys.readouterr().out
    assert "Gaia stars not checked (--keep-stars)" in printed and "Gaia stars:" not in printed
    (row,) = rows(sweep_out / "sweep.csv")
    assert row["fine_n_star"] == "0"
    assert json.loads((sweep_out / "settings.json").read_text())["keep_stars"] is True


def test_sweep_counts_the_stars_with_or_without_keep_stars(monkeypatch, tmp_path, capsys):
    """A catalog star on the central galaxy of the sweep's val pairs: left out of the
    fine-structure metric by default (fine_n_star), kept with --keep-stars, counted per image
    either way, with a non-default radius recorded in settings.json."""
    monkeypatch.setattr(sweep, "VGGLoss", CheapVGG)
    config = write_run(tmp_path, [25000])
    catalog = write_catalog(tmp_path / "cat.csv", [(150.0, 2.0, 0.0, 0.0, 1.0, 15.0)])
    monkeypatch.setattr(stars, "CATALOG", catalog)
    base = ["--config", str(config), "--device", "cpu", "--no-comet"]
    sweep.main([*base, "--out", str(tmp_path / "aside")])
    printed = capsys.readouterr().out
    assert "Gaia stars: 2 in 2 images; 2 HST sources inside their masks left out" in printed
    (row,) = rows(tmp_path / "aside" / "sweep.csv")
    assert row["fine_n_star"] == "2"
    sweep.main([*base, "--out", str(tmp_path / "kept"), "--keep-stars", "--star-radius", "0.4"])
    printed = capsys.readouterr().out
    assert "Gaia stars: 2 in 2 images; none left out of the fine-structure metric" in printed
    (kept,) = rows(tmp_path / "kept" / "sweep.csv")
    assert kept["fine_n_star"] == "0"
    counted = sum(int(kept[k]) for k in ("fine_n_sources", "fine_n_saturated", "fine_n_blown"))
    aside = sum(int(row[k]) for k in ("fine_n_sources", "fine_n_saturated", "fine_n_blown"))
    assert counted == aside + 2
    settings = json.loads((tmp_path / "kept" / "settings.json").read_text())
    assert settings["keep_stars"] is True and settings["stars"]["radius_arcsec"] == 0.4
    assert "stars" not in json.loads((tmp_path / "aside" / "settings.json").read_text())
