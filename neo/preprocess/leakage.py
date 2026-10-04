"""Verify, and with --apply enforce, that train and val pairs share no sky.

Run after neo.preprocess.pairs on a pairs directory (train/ and val/, each with lr/ and hr/):
  1. val duplicates: LSST patches and tracts overlap, so two val cutouts can show the same sky.
     Keeping only the first (by name) of any overlapping val cutouts makes every val source count
     once.
  2. train/val leaks: a train cutout that overlaps, or comes within --margin arcsec of, a val
     cutout. The smallest set of cutouts that removes every such conflict is taken out: a minimum
     vertex cover of the bipartite conflict graph (Konig's theorem, via a maximum matching). The
     sky-stripe split in pairs.py leaves none by construction; the legacy row split does not.
  3. val groups: val cutouts still overlapping each other (none after step 1) are linked, and
     <pairs>/val/groups.csv lets neo.eval.compare keep each group in one subset, so the
     checkpoint-selection and report subsets share no sky either.
  4. incomplete pairs: a name present in only one of lr/ and hr/ (the loaders list hr/).
With --apply, removed pairs are moved to <pairs>/quarantine/{train,val}/{lr,hr}/ (nothing is
deleted) and <pairs>/leakage.csv lists why. --apply also writes
  <pairs>/val_select/{lr,hr}/  links to the 20% select subset of val (neo.eval.subsets): the only
                               val pairs training-time validation and checkpoint choice may see;
  <pairs>/manifest.json        the build id that neo.preprocess.manifest guard checks before
                               training resumes a checkpoint.
A dry run exits non-zero when it finds any problem, or when those outputs are missing or stale.

Footprints are the cutouts' outer pixel edges through their WCS, compared in one gnomonic
projection centred on the field: two cutouts conflict when they share area or their exact
polygon distance is below --margin. The projection stretches distances by up to sec^2 of the
largest angle from its centre, so the margin is scaled by that factor (never under-counting).
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

from neo.eval.subsets import in_subset, load_groups, pair_id
from neo.preprocess.manifest import build_id, read_build, write_manifest

KINDS = ("lr", "hr")


def footprint(header) -> np.ndarray:
    """Sky corners (deg) of a cutout: the outer pixel edges, shape (4, 2)."""
    wcs = WCS(header).celestial
    ny, nx = header["NAXIS2"], header["NAXIS1"]
    x = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
    y = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
    return np.column_stack(wcs.pixel_to_world_values(x, y))


def tangent_plane(corners_deg: np.ndarray, return_stretch: bool = False):
    """Project (N, 4, 2) sky corners to arcsec offsets in one TAN plane centred on the field.

    With return_stretch, also return the largest factor by which the projection lengthens a
    distance anywhere among these points (sec^2 of the largest angle from the centre).
    """
    ra, dec = np.radians(corners_deg[..., 0]), np.radians(corners_deg[..., 1])
    xyz = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)
    centre = xyz.reshape(-1, 3).mean(0)
    centre /= np.linalg.norm(centre)
    ra0, dec0 = np.arctan2(centre[1], centre[0]), np.arcsin(centre[2])
    east = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    north = np.array([-np.sin(dec0) * np.cos(ra0), -np.sin(dec0) * np.sin(ra0), np.cos(dec0)])
    depth = xyz @ centre
    if (depth <= 0.5).any():
        raise ValueError("cutouts span more than 60 degrees; one tangent plane is not enough")
    arcsec = np.degrees(1) * 3600
    plane = np.stack([(xyz @ east) / depth, (xyz @ north) / depth], -1) * arcsec
    return (plane, float(1 / depth.min() ** 2)) if return_stretch else plane


def overlaps(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> bool:
    """Separating-axis test for convex quads (4, 2): True if their interiors intersect.

    Quads that only touch along an edge (within eps arcsec) do not overlap.
    """
    for quad in (a, b):
        edges = np.roll(quad, -1, axis=0) - quad
        for normal in np.column_stack([-edges[:, 1], edges[:, 0]]):
            normal = normal / np.linalg.norm(normal)
            pa, pb = a @ normal, b @ normal
            if pa.max() <= pb.min() + eps or pb.max() <= pa.min() + eps:
                return False
    return True


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between convex quads (4, 2); 0 when their interiors intersect."""
    if overlaps(a, b):
        return 0.0
    best = np.inf
    for points, quad in ((a, b), (b, a)):
        start, end = quad, np.roll(quad, -1, axis=0)
        edge = end - start
        for p in points:
            t = np.clip(((p - start) * edge).sum(1) / (edge * edge).sum(1), 0, 1)
            best = min(best, np.linalg.norm(p - (start + t[:, None] * edge), axis=1).min())
    return float(best)


def conflict(a: np.ndarray, b: np.ndarray, margin: float) -> bool:
    """True when two footprints share area, or (margin > 0) are closer than margin."""
    if margin > 0:
        return distance(a, b) < margin
    return overlaps(a, b)


def conflicts(quads_a, quads_b, margin=0.0, same=False):
    """Index pairs (i, j) of conflicting quads_a[i], quads_b[j] (i < j and a is b if same)."""
    if len(quads_a) == 0 or len(quads_b) == 0:
        return []
    reach = max(
        np.linalg.norm(q - q.mean(1, keepdims=True), axis=-1).max() for q in (quads_a, quads_b)
    )
    radius = (2 * reach + margin) * 1.001
    tree_a = cKDTree(quads_a.mean(1))
    if same:
        candidates = tree_a.query_pairs(radius)
    else:
        hits = tree_a.query_ball_tree(cKDTree(quads_b.mean(1)), radius)
        candidates = [(i, j) for i, js in enumerate(hits) for j in js]
    return sorted((i, j) for i, j in candidates if conflict(quads_a[i], quads_b[j], margin))


def first_of_overlapping(n, edges):
    """Indices kept when walking 0..n-1 and dropping anything overlapping an earlier keeper."""
    neighbours = [[] for _ in range(n)]
    for i, j in edges:
        neighbours[i].append(j)
        neighbours[j].append(i)
    kept = np.zeros(n, bool)
    for i in range(n):
        kept[i] = not any(kept[j] for j in neighbours[i])
    return kept


def min_vertex_cover(n_left, n_right, edges):
    """Minimum vertex cover of a bipartite graph: (left indices, right indices)."""
    if not edges:
        return set(), set()
    rows, cols = np.array(edges).T
    graph = csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(n_left, n_right))
    match_left = maximum_bipartite_matching(graph, perm_type="column")
    match_right = np.full(n_right, -1)
    for i, j in enumerate(match_left):
        if j >= 0:
            match_right[j] = i
    adjacency = [graph.indices[graph.indptr[i] : graph.indptr[i + 1]] for i in range(n_left)]
    # Konig: Z = vertices reachable from unmatched left vertices along alternating paths;
    # the cover is (left not in Z) + (right in Z).
    seen_left = match_left < 0
    seen_right = np.zeros(n_right, bool)
    stack = list(np.flatnonzero(seen_left))
    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            if v != match_left[u] and not seen_right[v]:
                seen_right[v] = True
                w = match_right[v]
                if w >= 0 and not seen_left[w]:
                    seen_left[w] = True
                    stack.append(w)
    left = set(np.flatnonzero(~seen_left).tolist())
    right = set(np.flatnonzero(seen_right).tolist())
    assert all(i in left or j in right for i, j in edges), "not a vertex cover"
    assert len(left) + len(right) == int((match_left >= 0).sum()), "cover is not minimum"
    return left, right


def groups_of(names, edges):
    """Connected components (union-find) of `edges`, each named by its first member."""
    parent = list(range(len(names)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges:
        parent[root(i)] = root(j)
    members = {}
    for i, name in enumerate(names):
        members.setdefault(root(i), []).append(name)
    return {name: min(group) for group in members.values() for name in group}


def read_split(pairs: Path, split: str):
    """Complete pairs (names, LR footprints, pair ids) and names missing their lr or hr half."""
    listed = {kind: {p.name for p in (pairs / split / kind).glob("*.fits")} for kind in KINDS}
    names = sorted(listed["lr"] & listed["hr"])
    orphans = sorted(listed["lr"] ^ listed["hr"])
    headers = [fits.getheader(pairs / split / "lr" / n) for n in names]
    corners = np.array([footprint(h) for h in headers]).reshape(-1, 4, 2)
    return names, corners, [pair_id(h) for h in headers], orphans


def linked_names(directory: Path):
    return {p.name for p in directory.glob("*.fits")} if directory.is_dir() else set()


def write_val_select(pairs: Path, names):
    """Rebuild <pairs>/val_select/{lr,hr}/ as relative links to the given val pairs."""
    root = pairs / "val_select"
    shutil.rmtree(root, ignore_errors=True)
    for kind in KINDS:
        (root / kind).mkdir(parents=True)
        for name in names:
            os.symlink(os.path.join("..", "..", "val", kind, name), root / kind / name)


def audit(train_quads, val_quads, margin):
    """Decide what to remove. Returns (val_duplicates, train_out, val_out, conflict_edges)."""
    val_dup = ~first_of_overlapping(len(val_quads), conflicts(val_quads, val_quads, same=True))
    val_kept = np.flatnonzero(~val_dup)
    edges = conflicts(train_quads, val_quads[val_kept], margin)
    train_out, val_out_local = min_vertex_cover(len(train_quads), len(val_kept), edges)
    val_out = {int(val_kept[j]) for j in val_out_local}
    edges = [(i, int(val_kept[j])) for i, j in edges]
    return set(np.flatnonzero(val_dup).tolist()), train_out, val_out, edges


def quarantine(pairs: Path, split: str, name: str):
    for kind in KINDS:
        src = pairs / split / kind / name
        if src.exists():
            dest = pairs / "quarantine" / split / kind
            dest.mkdir(parents=True, exist_ok=True)
            os.replace(src, dest / name)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", required=True, help="pairs directory with train/ and val/")
    parser.add_argument(
        "--margin", type=float, default=5.0, help="arcsec of sky required between train and val"
    )
    parser.add_argument("--apply", action="store_true", help="quarantine, write val/groups.csv")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pairs = Path(args.pairs)
    train_names, train_corners, train_ids, train_orphans = read_split(pairs, "train")
    val_names, val_corners, val_ids, val_orphans = read_split(pairs, "val")
    if not train_names or not val_names:
        raise SystemExit(f"need both train and val pairs under {pairs}")
    plane, stretch = tangent_plane(
        np.concatenate([train_corners, val_corners]), return_stretch=True
    )
    train_quads, val_quads = plane[: len(train_names)], plane[len(train_names) :]
    margin = args.margin * stretch

    val_dup, train_out, val_out, edges = audit(train_quads, val_quads, margin)
    n_train, n_val = len(train_names), len(val_names)
    kept_train = [i for i in range(n_train) if i not in train_out]
    kept_val = [j for j in range(n_val) if j not in val_dup and j not in val_out]
    kept_names = [val_names[j] for j in kept_val]
    groups = groups_of(kept_names, conflicts(val_quads[kept_val], val_quads[kept_val], same=True))
    select = [n for n in kept_names if in_subset(n, "select", groups)]
    entries = [("train", train_names[i], train_ids[i]) for i in kept_train]
    entries += [("val", val_names[j], val_ids[j]) for j in kept_val]
    orphans = [("train", n) for n in train_orphans] + [("val", n) for n in val_orphans]
    print(
        f"{n_train} train, {n_val} val pairs; margin {args.margin} arcsec "
        f"(x{stretch:.6f} for the projection)\n"
        f"incomplete pairs (lr or hr missing): {len(orphans)}\n"
        f"val duplicates (same sky as an earlier val cutout): {len(val_dup)}\n"
        f"train/val conflicts: {len(edges)} between {len({i for i, _ in edges})} train and "
        f"{len({j for _, j in edges})} val cutouts; minimum removal: {len(train_out)} train + "
        f"{len(val_out)} val\n"
        f"after removal: {len(kept_train)} train, {len(kept_val)} val "
        f"({len(select)} select, {len(kept_val) - len(select)} report)"
    )
    problems = len(orphans) + len(val_dup) + len(train_out) + len(val_out)
    if not args.apply:
        stale = []
        if read_build(pairs) != build_id(entries):
            stale.append("manifest.json")
        if load_groups(pairs / "val") != groups:
            stale.append("val/groups.csv")
        if any(linked_names(pairs / "val_select" / kind) != set(select) for kind in KINDS):
            stale.append("val_select/")
        if stale:
            print(f"missing or out of date: {', '.join(stale)} (use --apply)")
        if problems:
            print("dry run: nothing moved (use --apply)")
        elif not stale:
            print("dry run: nothing moved; no leakage")
        return 1 if problems or stale else 0

    with open(pairs / "leakage.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["split", "name", "reason", "conflicts_with"])
        partners = {}
        for i, j in edges:
            partners.setdefault(("train", i), []).append(val_names[j])
            partners.setdefault(("val", j), []).append(train_names[i])
        for split, name in orphans:
            writer.writerow([split, name, "lr or hr missing", ""])
        for j in sorted(val_dup):
            writer.writerow(["val", val_names[j], "duplicate val sky", ""])
        for split, removed, names in (
            ("train", train_out, train_names),
            ("val", val_out, val_names),
        ):
            for i in sorted(removed):
                others = " ".join(sorted(partners[(split, i)]))
                writer.writerow([split, names[i], "train/val overlap", others])
    for split, name in orphans:
        quarantine(pairs, split, name)
    for j in val_dup | val_out:
        quarantine(pairs, "val", val_names[j])
    for i in train_out:
        quarantine(pairs, "train", train_names[i])

    with open(pairs / "val" / "groups.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "group"])
        writer.writerows(sorted(groups.items()))
    write_val_select(pairs, select)
    build = write_manifest(pairs, entries, margin_arcsec=args.margin, select=len(select))
    print(
        f"moved {problems} pairs to {pairs / 'quarantine'}; wrote leakage.csv, "
        f"val/groups.csv ({len(set(groups.values()))} groups), val_select/ ({len(select)} pairs) "
        f"and manifest.json (build {build}) under {pairs}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
