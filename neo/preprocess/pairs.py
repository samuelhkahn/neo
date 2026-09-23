"""Extract paired low/high resolution cutouts for super-resolution training.

Each LR image gets an HR grid nested exactly `factor` times finer than its own pixels, and every
HR mosaic tile overlapping it is reprojected (flux-conserving) onto that grid, so an LR window and
its HR counterpart cover the same sky. A band of rows in every LR image is reserved for validation,
and no window straddles the boundary, so train and val cutouts never share a pixel.
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from reproject import reproject_adaptive
from scipy.ndimage import binary_erosion

from neo.preprocess.grid import lr_window_mask, upsampled_wcs
from neo.surveys.hst.mosaic import MosaicSet
from neo.surveys.rubin.coadd import flagged_pixels, load_coadd

DEFAULT_REJECT = ("NO_DATA", "SATURATED")
SPLITS = ("train", "val")


def sky_bbox(wcs: WCS, shape) -> tuple[float, float, float, float]:
    ny, nx = shape
    corners = wcs.calc_footprint(axes=(nx, ny))
    return corners[:, 0].min(), corners[:, 0].max(), corners[:, 1].min(), corners[:, 1].max()


def overlap_box(lr_wcs: WCS, lr_shape, hr_wcs: WCS, hr_shape, pad: int = 2):
    """LR pixel box (y0, y1, x0, x1) enclosing the HR image's footprint, or None if disjoint."""
    ny_hr, nx_hr = hr_shape
    corners = hr_wcs.calc_footprint(axes=(nx_hr, ny_hr))
    x, y = lr_wcs.all_world2pix(corners[:, 0], corners[:, 1], 0)
    ny, nx = lr_shape
    y0 = int(np.clip(np.floor(y.min()) - pad, 0, ny))
    y1 = int(np.clip(np.ceil(y.max()) + pad + 1, 0, ny))
    x0 = int(np.clip(np.floor(x.min()) - pad, 0, nx))
    x1 = int(np.clip(np.ceil(x.max()) + pad + 1, 0, nx))
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1


def union_box(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    y0 = min(b[0] for b in boxes)
    y1 = max(b[1] for b in boxes)
    x0 = min(b[2] for b in boxes)
    x1 = max(b[3] for b in boxes)
    return y0, y1, x0, x1


def hr_on_lr_grid(hr_data, hr_wcs, lr_wcs, lr_shape, factor, block_size=2048, parallel=False):
    """Flux-conserving reprojection of `hr_data` onto the grid nested `factor`x inside `lr_wcs`."""
    target = upsampled_wcs(lr_wcs, factor)
    shape_out = (lr_shape[0] * factor, lr_shape[1] * factor)
    hr, footprint = reproject_adaptive(
        (hr_data, hr_wcs),
        target,
        shape_out=shape_out,
        conserve_flux=True,
        block_size=(block_size, block_size),
        parallel=parallel,
    )
    valid = (footprint > 0) & np.isfinite(hr) & (hr != 0)
    return np.nan_to_num(hr).astype(np.float32), valid, target


def merge_on_lr_grid(regions, lr_wcs, lr_shape, factor, block_size=2048, parallel=False):
    """Reproject several HR tiles onto one nested grid; the first tile with data wins per pixel."""
    hr = np.zeros((lr_shape[0] * factor, lr_shape[1] * factor), np.float32)
    valid = np.zeros_like(hr, dtype=bool)
    hr_wcs = None
    for data, wcs in regions:
        tile, tile_valid, hr_wcs = hr_on_lr_grid(
            data, wcs, lr_wcs, lr_shape, factor, block_size, parallel
        )
        take = tile_valid & ~valid
        hr[take] = tile[take]
        valid |= take
    return hr, valid, hr_wcs


def full_window_corners(ok: np.ndarray, size: int) -> np.ndarray:
    """Map over top-left corners: True where the size x size window is all-True in `ok`."""
    ny, nx = ok.shape
    if ny < size or nx < size:
        return np.zeros((0, 0), bool)
    s = np.pad(ok.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    total = s[size:, size:] - s[:-size, size:] - s[size:, :-size] + s[:-size, :-size]
    return total == size * size


def sample_windows(ok: np.ndarray, size: int, n: int, rng) -> tuple[list[tuple[int, int]], int]:
    """Up to n distinct top-left corners of fully valid windows, and how many were available."""
    corners = np.argwhere(full_window_corners(ok, size))
    if len(corners) == 0 or n <= 0:
        return [], len(corners)
    pick = rng.choice(len(corners), size=min(n, len(corners)), replace=False)
    return [(int(y), int(x)) for y, x in corners[pick]], len(corners)


def split_masks(ok: np.ndarray, val_frac: float) -> dict[str, np.ndarray]:
    """Train uses the top rows, val the bottom ones; windows cannot cross the boundary.

    The boundary is placed at the (1 - val_frac) quantile of rows that contain valid pixels,
    so val gets its share even when coverage does not reach the bottom of the image.
    """
    rows = np.flatnonzero(ok.any(axis=1))
    cut = int(round(len(rows) * (1 - val_frac)))
    split = rows[cut] if cut < len(rows) else ok.shape[0]
    train, val = ok.copy(), ok.copy()
    train[split:, :] = False
    val[:split, :] = False
    return {"train": train, "val": val}


def cutout_hdu(data, wcs, y0, x0, cards) -> fits.PrimaryHDU:
    ny, nx = data.shape
    hdu = fits.PrimaryHDU(data=data, header=wcs[y0 : y0 + ny, x0 : x0 + nx].to_header())
    hdu.header.update(cards)
    return hdu


def process_patch(
    lr_path, mosaic, out, size, factor, per_patch, rng, reject, block_size, val_frac, parallel=False
):
    counts = {"train": 0, "val": 0, "valid": 0}
    coadd = load_coadd(lr_path)
    regions = mosaic.regions(*sky_bbox(coadd.wcs, coadd.image.shape))
    box = union_box(
        overlap_box(coadd.wcs, coadd.image.shape, wcs, data.shape) for data, wcs in regions
    )
    if box is None:
        return counts
    y0, y1, x0, x1 = box
    sub_wcs = coadd.wcs[y0:y1, x0:x1]
    hr, hr_valid, hr_wcs = merge_on_lr_grid(
        regions, sub_wcs, (y1 - y0, x1 - x0), factor, block_size, parallel
    )
    # Erode one LR pixel so windows stay clear of the resampled footprint edge.
    ok = binary_erosion(lr_window_mask(hr_valid, factor))
    ok &= ~flagged_pixels(coadd, reject)[y0:y1, x0:x1]
    counts["valid"] = int(full_window_corners(ok, size).sum())

    n_train = int(round(per_patch * (1 - val_frac)))
    wanted = {"train": n_train, "val": per_patch - n_train}
    for split, mask in split_masks(ok, val_frac).items():
        corners, _ = sample_windows(mask, size, wanted[split], rng)
        for k, (cy, cx) in enumerate(corners):
            ly, lx = y0 + cy, x0 + cx
            lr_cut = coadd.image[ly : ly + size, lx : lx + size]
            hr_cut = hr[factor * cy : factor * (cy + size), factor * cx : factor * (cx + size)]
            hr_cut = (hr_cut * mosaic.njy_per_count).astype(np.float32)
            cards = {"LRFILE": lr_path.name, "LRX0": lx, "LRY0": ly, "SRFACTOR": factor}
            hr_cards = {
                **cards,
                "BUNIT": "nJy",
                "HRZPAB": mosaic.zeropoint,
                "HRSCALE": mosaic.njy_per_count,
            }
            name = f"{lr_path.stem}_{split}_{k:05d}.fits"
            lr_hdu = cutout_hdu(lr_cut, coadd.wcs, ly, lx, {**cards, "BUNIT": coadd.bunit})
            hr_hdu = cutout_hdu(hr_cut, hr_wcs, factor * cy, factor * cx, hr_cards)
            lr_hdu.writeto(out / split / "lr" / name, overwrite=True)
            hr_hdu.writeto(out / split / "hr" / name, overwrite=True)
        counts[split] = len(corners)
    return counts


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lr-dir", required=True, help="directory of LR FITS images")
    parser.add_argument("--lr-glob", default="*.fits", help="pattern for LR files in --lr-dir")
    parser.add_argument(
        "--hr-mosaic",
        required=True,
        nargs="+",
        help="HR mosaic FITS file(s), e.g. COSMOS-Web tiles",
    )
    parser.add_argument(
        "--hr-zeropoint",
        type=float,
        default=None,
        help="AB zeropoint for HR mosaics lacking PHOTFLAM (COSMOS-Web F814W: 25.94)",
    )
    parser.add_argument("--out", required=True, help="output dir; gets {train,val}/{lr,hr}/")
    parser.add_argument("--lr-size", type=int, default=142, help="LR cutout size in pixels")
    parser.add_argument("--factor", type=int, default=6, help="HR/LR pixel scale ratio")
    parser.add_argument("--per-patch", type=int, default=200, help="cutouts per LR image")
    parser.add_argument("--val-frac", type=float, default=0.2, help="fraction of rows held out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reject",
        nargs="*",
        default=list(DEFAULT_REJECT),
        help="LR mask planes that disqualify a window",
    )
    parser.add_argument("--limit", type=int, help="process only the first N LR images")
    parser.add_argument("--block-size", type=int, default=2048, help="reproject block size")
    parser.add_argument("--parallel", type=int, default=0, help="reproject worker processes")
    parser.add_argument("--dry-run", action="store_true", help="report overlap, write nothing")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    warnings.simplefilter("ignore", AstropyWarning)
    lr_paths = sorted(Path(args.lr_dir).glob(args.lr_glob))[: args.limit]
    mosaic = MosaicSet(args.hr_mosaic, zeropoint=args.hr_zeropoint)
    out = Path(args.out)
    pair_bytes = 4 * (args.lr_size**2 + (args.lr_size * args.factor) ** 2)

    if args.dry_run:
        n_overlap = 0
        for path in lr_paths:
            header = fits.getheader(path, "IMAGE")
            shape = (header["NAXIS2"], header["NAXIS1"])
            tiles = mosaic.overlapping(*sky_bbox(WCS(header), shape))
            if not tiles:
                continue
            n_overlap += 1
            print(f"  {path.name}: {len(tiles)} tile(s): {', '.join(t.path.name for t in tiles)}")
        n_max = n_overlap * args.per_patch
        print(
            f"{n_overlap}/{len(lr_paths)} LR images overlap the mosaic: "
            f"up to {n_max} pairs, ~{n_max * pair_bytes / 1e9:.1f} GB"
        )
        return

    for split in SPLITS:
        for kind in ("lr", "hr"):
            (out / split / kind).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    totals = {"train": 0, "val": 0}
    for path in lr_paths:
        t0 = time.time()
        counts = process_patch(
            path,
            mosaic,
            out,
            args.lr_size,
            args.factor,
            args.per_patch,
            rng,
            args.reject,
            args.block_size,
            args.val_frac,
            args.parallel or False,
        )
        for split in SPLITS:
            totals[split] += counts[split]
        print(
            f"{path.name}: {counts['valid']} valid windows, wrote {counts['train']} train + "
            f"{counts['val']} val ({time.time() - t0:.0f} s)"
        )
    size_gb = sum(totals.values()) * pair_bytes / 1e9
    print(f"{totals['train']} train + {totals['val']} val pairs -> {out} (~{size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
