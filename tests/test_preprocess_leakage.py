import csv
import itertools

import numpy as np
from astropy.io import fits
from conftest import make_tan_wcs

from neo.eval.compare import in_subset, load_groups
from neo.preprocess import leakage


def square(cx, cy, half, angle=0.0):
    c, s = np.cos(angle), np.sin(angle)
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    return corners @ np.array([[c, s], [-s, c]]) + (cx, cy)


def test_overlaps_needs_shared_area():
    a = square(0, 0, 1)
    assert leakage.overlaps(a, square(1.5, 0, 1))
    assert not leakage.overlaps(a, square(2, 0, 1))  # touching edges share no area
    assert not leakage.overlaps(a, square(2.01, 0.5, 1))
    assert leakage.overlaps(a, square(2.3, 0, 1, np.pi / 4))  # diamond tip pokes in
    assert not leakage.overlaps(a, square(2.5, 0, 1, np.pi / 4))
    assert leakage.overlaps(a, square(0.2, 0.1, 0.1))  # contained


def test_distance_is_exact_including_corners():
    a = square(0, 0, 1)
    assert leakage.distance(a, square(1.5, 0, 1)) == 0
    assert np.isclose(leakage.distance(a, square(3, 0.5, 1)), 1.0)  # edge to edge
    assert np.isclose(leakage.distance(a, square(3, 3, 1)), np.sqrt(2))  # corner to corner
    assert np.isclose(leakage.distance(a, square(2 + np.sqrt(2), 0, 1, np.pi / 4)), 1.0)
    # a margin catches what is closer than it, corners included, and nothing farther
    assert leakage.conflict(a, square(3, 3, 1), 1.5) and not leakage.conflict(
        a, square(3, 3, 1), 1.4
    )
    assert leakage.conflict(a, square(2, 0, 1), 0.1) and not leakage.conflict(a, square(2, 0, 1), 0)


def brute_force_cover(n_left, n_right, edges):
    vertices = range(n_left + n_right)
    for k in range(n_left + n_right + 1):
        for chosen in itertools.combinations(vertices, k):
            s = set(chosen)
            if all(i in s or n_left + j in s for i, j in edges):
                return k


def test_min_vertex_cover_is_a_minimum_cover():
    rng = np.random.default_rng(0)
    for _ in range(60):
        n_left, n_right = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        edges = sorted(
            {
                (int(rng.integers(n_left)), int(rng.integers(n_right)))
                for _ in range(rng.integers(0, 12))
            }
        )
        left, right = leakage.min_vertex_cover(n_left, n_right, edges)
        assert all(i in left or j in right for i, j in edges)
        assert len(left) + len(right) == brute_force_cover(n_left, n_right, edges)


def test_first_of_overlapping_and_groups():
    kept = leakage.first_of_overlapping(4, [(0, 1), (1, 2)])
    assert kept.tolist() == [True, False, True, True]
    groups = leakage.groups_of(["a", "b", "c", "d", "e"], [(0, 1), (3, 2), (1, 3)])
    assert {groups[n] for n in "abcd"} == {"a"} and groups["e"] == "e"


def test_tangent_plane_keeps_arcsec_distances():
    wcs = make_tan_wcs(0.2, (100, 100))
    corners = np.array([wcs.pixel_to_world_values([0, 50], [0, 0])]).transpose(0, 2, 1)
    plane = leakage.tangent_plane(np.repeat(corners, 2, axis=1))  # points p0, p0, p1, p1
    assert np.isclose(np.linalg.norm(plane[0, 2] - plane[0, 0]), 10.0, rtol=1e-6)


def write_pair(root, split, name, wcs, y0, x0, size=20):
    header = wcs[y0 : y0 + size, x0 : x0 + size].to_header()
    for kind, n in (("lr", size), ("hr", 6 * size)):
        (root / split / kind).mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(np.zeros((n, n), np.float32), header=header).writeto(
            root / split / kind / name
        )


def test_main_quarantines_leaks_and_duplicates(tmp_path):
    wcs = make_tan_wcs(0.2, (400, 400))
    write_pair(tmp_path, "train", "t0.fits", wcs, 0, 0)  # 1" from v0: a leak at 5" margin
    write_pair(tmp_path, "train", "t1.fits", wcs, 0, 100)  # far from everything
    write_pair(tmp_path, "val", "v0.fits", wcs, 25, 0)
    write_pair(tmp_path, "val", "v1.fits", wcs, 200, 200)
    write_pair(tmp_path, "val", "v2.fits", wcs, 205, 205)  # same sky as v1

    assert leakage.main(["--pairs", str(tmp_path), "--margin", "0.5"]) == 1  # duplicate only
    assert leakage.main(["--pairs", str(tmp_path)]) == 1
    assert leakage.main(["--pairs", str(tmp_path), "--apply"]) == 0
    assert leakage.main(["--pairs", str(tmp_path)]) == 0  # nothing left to find

    remaining = {p.name for p in (tmp_path / "train" / "lr").glob("*.fits")}
    remaining |= {p.name for p in (tmp_path / "val" / "lr").glob("*.fits")}
    moved = {p.name for p in (tmp_path / "quarantine").rglob("lr/*.fits")}
    assert "v2.fits" in moved and "t1.fits" in remaining and "v1.fits" in remaining
    assert len({"t0.fits", "v0.fits"} & moved) == 1  # one side of the single conflict
    for name in moved:
        split = "train" if name.startswith("t") else "val"
        assert (tmp_path / "quarantine" / split / "hr" / name).exists()
    with open(tmp_path / "leakage.csv") as f:
        reasons = {row["name"]: row["reason"] for row in csv.DictReader(f)}
    assert reasons["v2.fits"] == "duplicate val sky"
    groups = load_groups(tmp_path / "val")
    assert set(groups) == {p.name for p in (tmp_path / "val" / "lr").glob("*.fits")}


def test_compare_keeps_each_sky_group_in_one_subset():
    names = [f"cut_{i:03d}.fits" for i in range(200)]
    groups = {n: names[(i // 5) * 5] for i, n in enumerate(names)}
    for subset in ("select", "report"):
        for start in range(0, 200, 5):
            members = {in_subset(n, subset, groups) for n in names[start : start + 5]}
            assert len(members) == 1
    selected = sum(in_subset(n, "select", groups) for n in names)
    assert 0 < selected < 200
    assert all(in_subset(n, "select", groups) != in_subset(n, "report", groups) for n in names)
