"""Extract paired low/high resolution cutouts for super-resolution training.

Each LR image gets an HR grid nested exactly `factor` times finer than its own pixels, and every
HR mosaic tile overlapping it is reprojected (flux-conserving) onto that grid, so an LR window and
its HR counterpart cover the same sky.

Train/val split (--split):
  sky   (default) global declination stripes: val is the first --val-frac of every --val-period
        arcmin in Dec, train is everything else farther than --guard arcsec from val. Overlapping
        LSST patches and tracts agree on every pixel's split, so no sky is shared between train
        and val (neo.preprocess.leakage verifies this on the written cutouts).
  rows  (legacy) the bottom rows of each LSST image are val. Patches overlap their neighbours by
        300 px, so this shares sky between one patch's val and the next patch's train.
Units (--units):
  paper  the NEO paper's conventions, so its fixed log stretch sees the paper's values for any sky:
         HR is the HST mosaic resampled with reproject_interp (surface brightness preserved), as
         e-/s per 0.03" pixel (the paper's CANDELS mosaic pixel); LR is converted from nJy to
         HSC counts (AB zeropoint 27) per 0.168" pixel of equal surface brightness. NJYPERPX in
         each header converts a stored value to nJy per pixel of that cutout.
  njy    both in nJy per pixel; HR flux-conserving (reproject_adaptive).
--hr-sky-subtract removes each LSST image's HST sky pedestal (3-sigma-clipped median of the
resampled HR) before cutting: the paper's CANDELS mosaic sat at ~0.01 sigma, a COSMOS-Web DR1 tile
at ~0.3 sigma, which shifts what survives the dataset's clip at 0.
--flux-filter LOW HIGH keeps a pair only if LOW < ln(HR pixel sum) < HIGH, as the paper's
neo/data/filter_samples.py did (2 7, e-/s). The paper summed its whole 852 px cutout on a
0.028" grid (23.9"); here the sum covers the same sky (the central 0.168/0.2 of the cutout),
scaled to the paper's sampling density, so the cut rejects the same fields (--units paper only).

Window counts: --train-density D samples about D windows per train pixel (coverage ~1 - e^-D);
--val-tile packs non-overlapping val windows (every val source appears once). Without them,
--per-patch random windows are split by --val-frac.

Each LR image draws its windows from its own seed (--seed and the file name) and leaves a marker in
<out>/.done/ when finished, so --resume continues an interrupted run exactly: finished images are
skipped, a half-written one is cleared and redone, and the settings must match the first run's.
"""

import argparse
import json
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_adaptive, reproject_interp
from scipy.ndimage import binary_erosion, distance_transform_edt

from neo.preprocess.grid import lr_window_mask, upsampled_wcs
from neo.surveys.hst.mosaic import MosaicSet, njy_per_count
from neo.surveys.rubin.coadd import flagged_pixels, load_coadd

DEFAULT_REJECT = ("NO_DATA", "SATURATED")
# The NEO paper's LR: HSC coadds in counts at AB zeropoint 27, 0.168" pixels
HSC_ZEROPOINT = 27.0
HSC_PIXEL_ARCSEC = 0.168
# ... and HR: HST e-/s per 0.03" pixel (CANDELS F814W, resampled preserving surface brightness)
PAPER_HST_PIXEL_ARCSEC = 0.03
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


def hr_on_lr_grid(
    hr_data, hr_wcs, lr_wcs, lr_shape, factor, block_size=2048, parallel=False, method="adaptive"
):
    """Reproject `hr_data` onto the grid nested `factor`x inside `lr_wcs`.

    method "adaptive": flux-conserving (values become flux per output pixel);
    method "interp": bilinear reproject_interp, as the NEO paper did (values stay surface
    brightness, i.e. per native input pixel).
    """
    target = upsampled_wcs(lr_wcs, factor)
    shape_out = (lr_shape[0] * factor, lr_shape[1] * factor)
    common = dict(shape_out=shape_out, block_size=(block_size, block_size), parallel=parallel)
    if method == "interp":
        hr, footprint = reproject_interp((hr_data, hr_wcs), target, order="bilinear", **common)
    elif method == "adaptive":
        hr, footprint = reproject_adaptive((hr_data, hr_wcs), target, conserve_flux=True, **common)
    else:
        raise ValueError(f"unknown resampling method {method!r}")
    valid = (footprint > 0) & np.isfinite(hr) & (hr != 0)
    return np.nan_to_num(hr).astype(np.float32), valid, target


def merge_on_lr_grid(
    regions, lr_wcs, lr_shape, factor, block_size=2048, parallel=False, method="adaptive"
):
    """Reproject several HR tiles onto one nested grid; the first tile with data wins per pixel."""
    hr = np.zeros((lr_shape[0] * factor, lr_shape[1] * factor), np.float32)
    valid = np.zeros_like(hr, dtype=bool)
    hr_wcs = None
    for data, wcs in regions:
        tile, tile_valid, hr_wcs = hr_on_lr_grid(
            data, wcs, lr_wcs, lr_shape, factor, block_size, parallel, method
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


def sky_split_masks(
    ok: np.ndarray, wcs: WCS, val_frac: float, period_arcmin: float, guard_arcsec: float
) -> dict[str, np.ndarray]:
    """Split by global Dec stripes: val where Dec (arcmin) mod period < val_frac * period; train
    where the pixel is at least guard_arcsec (edge to edge) from any val pixel of any image.

    The stripes are computed a guard's width beyond the image, so val sky just outside it still
    pushes train windows away.
    """
    ny, nx = ok.shape
    celestial = wcs.celestial
    scale = float(np.mean(proj_plane_pixel_scales(celestial))) * 3600
    guard = guard_arcsec / scale
    g = int(np.ceil(guard)) + 2
    val = np.zeros((ny + 2 * g, nx + 2 * g), bool)
    xs = np.arange(-g, nx + g)
    for r0 in range(0, ny + 2 * g, 256):
        ys = np.arange(r0, min(r0 + 256, ny + 2 * g)) - g
        xx, yy = np.meshgrid(xs, ys)
        _, dec = celestial.pixel_to_world_values(xx, yy)
        val[r0 : r0 + len(ys)] = np.mod(dec * 60 / period_arcmin, 1.0) < val_frac
    # Centre-to-centre distance d leaves d - 1 px between pixel edges in this grid; one more pixel
    # covers images on other grids (neighbouring patches, tracts), whose val pixels can poke up to
    # half a pixel past the stripe edge. So train and val stay >= guard apart on the sky.
    near_val = distance_transform_edt(~val) < guard + 2
    inner = (slice(g, g + ny), slice(g, g + nx))
    return {"train": ok & ~near_val[inner], "val": ok & val[inner]}


def tile_windows(ok: np.ndarray, size: int) -> list[tuple[int, int]]:
    """Top-left corners of non-overlapping, fully valid windows, packed greedily in raster order."""
    free = full_window_corners(ok, size)
    if free.size == 0:
        return []
    flat = free.ravel()
    nx = free.shape[1]
    corners, start = [], 0
    while True:
        idx = start + int(np.argmax(flat[start:]))
        if not flat[idx]:
            return corners
        y, x = divmod(idx, nx)
        corners.append((y, x))
        free[max(0, y - size + 1) : y + size, max(0, x - size + 1) : x + size] = False
        start = idx


def cutout_hdu(data, wcs, y0, x0, cards) -> fits.PrimaryHDU:
    ny, nx = data.shape
    hdu = fits.PrimaryHDU(data=data, header=wcs[y0 : y0 + ny, x0 : x0 + nx].to_header())
    hdu.header.update(cards)
    return hdu


def process_patch(
    lr_path,
    mosaic,
    out,
    size,
    factor,
    per_patch,
    rng,
    reject,
    block_size,
    val_frac,
    parallel=False,
    split_mode="rows",
    val_period=10.0,
    guard=6.0,
    train_density=None,
    val_tile=False,
    units="njy",
    flux_filter=None,
    hr_sky_subtract=False,
):
    counts = {"train": 0, "val": 0, "valid": 0, "filtered": 0, "hr_sky": 0.0}
    coadd = load_coadd(lr_path)
    regions = mosaic.regions(*sky_bbox(coadd.wcs, coadd.image.shape))
    box = union_box(
        overlap_box(coadd.wcs, coadd.image.shape, wcs, data.shape) for data, wcs in regions
    )
    if box is None:
        return counts
    y0, y1, x0, x1 = box
    sub_wcs = coadd.wcs[y0:y1, x0:x1]
    if flux_filter is not None and units != "paper":
        raise ValueError("--flux-filter thresholds are in the paper's units: use --units paper")
    method = "interp" if units == "paper" else "adaptive"
    hr, hr_valid, hr_wcs = merge_on_lr_grid(
        regions, sub_wcs, (y1 - y0, x1 - x0), factor, block_size, parallel, method
    )
    if hr_sky_subtract and hr_valid.any():
        sample = hr[::7, ::7][hr_valid[::7, ::7]]  # ~2% of pixels is plenty for a median
        _, sky, sigma = sigma_clipped_stats(sample, sigma=3, maxiters=10)
        hr[hr_valid] -= sky
        counts["hr_sky"] = float(sky)
        counts["hr_sky_sigma"] = float(sky / sigma) if sigma > 0 else 0.0
    lr_pixel = float(np.mean(proj_plane_pixel_scales(coadd.wcs.celestial))) * 3600
    hr_pixel = lr_pixel / factor
    native = float(np.mean(proj_plane_pixel_scales(regions[0][1].celestial))) * 3600
    if units == "paper":
        # stored = nJy * lr_scale = HSC counts (ZP 27) per 0.168" pixel at equal surface brightness
        lr_scale = (HSC_PIXEL_ARCSEC / lr_pixel) ** 2 / njy_per_count(HSC_ZEROPOINT)
        # resampled values are e-/s per native mosaic pixel; restate per 0.03" pixel
        hr_scale = (PAPER_HST_PIXEL_ARCSEC / native) ** 2
        hr_njy_per_px = mosaic.njy_per_count * (hr_pixel / PAPER_HST_PIXEL_ARCSEC) ** 2
        lr_bunit, hr_bunit = "HSC count (ZP 27) per 0.168as px", "e-/s per 0.03as px"
    else:
        lr_scale, hr_scale, hr_njy_per_px = 1.0, mosaic.njy_per_count, 1.0
        lr_bunit, hr_bunit = coadd.bunit, "nJy"
    # Erode one LR pixel so windows stay clear of the resampled footprint edge.
    ok = binary_erosion(lr_window_mask(hr_valid, factor))
    ok &= ~flagged_pixels(coadd, reject)[y0:y1, x0:x1]
    counts["valid"] = int(full_window_corners(ok, size).sum())

    n_train = int(round(per_patch * (1 - val_frac)))
    wanted = {"train": n_train, "val": per_patch - n_train}
    if split_mode == "sky":
        masks = sky_split_masks(ok, sub_wcs, val_frac, val_period, guard)
    else:
        masks = split_masks(ok, val_frac)
    for split, mask in masks.items():
        if split == "val" and val_tile:
            corners = tile_windows(mask, size)
        elif split == "train" and train_density:
            n = int(np.ceil(train_density * mask.sum() / size**2))
            corners, _ = sample_windows(mask, size, n, rng)
        else:
            corners, _ = sample_windows(mask, size, wanted[split], rng)
        written = 0
        for k, (cy, cx) in enumerate(corners):
            ly, lx = y0 + cy, x0 + cx
            hr_cut = hr[factor * cy : factor * (cy + size), factor * cx : factor * (cx + size)]
            hr_cut = (hr_cut * hr_scale).astype(np.float32)
            if flux_filter is not None:
                n = int(round(hr_cut.shape[0] * HSC_PIXEL_ARCSEC / lr_pixel))
                o = (hr_cut.shape[0] - n) // 2
                paper_sum = (
                    np.sum(hr_cut[o : o + n, o : o + n], dtype=np.float64)
                    * (hr_pixel * factor / HSC_PIXEL_ARCSEC) ** 2
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_sum = np.log(paper_sum)
                if not flux_filter[0] < log_sum < flux_filter[1]:  # NaN (sum <= 0) fails too
                    counts["filtered"] += 1
                    continue
            lr_cut = (coadd.image[ly : ly + size, lx : lx + size] * lr_scale).astype(np.float32)
            cards = {
                "LRFILE": lr_path.name,
                "LRX0": lx,
                "LRY0": ly,
                "SRFACTOR": factor,
                "UNITS": units,
            }
            hr_cards = {
                **cards,
                "BUNIT": hr_bunit,
                "NJYPERPX": hr_njy_per_px,
                "HRZPAB": mosaic.zeropoint,
                "HRSCALE": mosaic.njy_per_count,
                "HRPIXNAT": native,
                "HRSKYSUB": counts["hr_sky"],
            }
            lr_cards = {**cards, "BUNIT": lr_bunit, "NJYPERPX": 1.0 / lr_scale}
            name = f"{lr_path.stem}_{split}_{k:05d}.fits"
            lr_hdu = cutout_hdu(lr_cut, coadd.wcs, ly, lx, lr_cards)
            hr_hdu = cutout_hdu(hr_cut, hr_wcs, factor * cy, factor * cx, hr_cards)
            lr_hdu.writeto(out / split / "lr" / name, overwrite=True)
            hr_hdu.writeto(out / split / "hr" / name, overwrite=True)
            written += 1
        counts[split] = written
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
    parser.add_argument("--val-frac", type=float, default=0.2, help="fraction of sky held out")
    parser.add_argument(
        "--split", choices=["sky", "rows"], default="sky", help="train/val split (module docstring)"
    )
    parser.add_argument("--val-period", type=float, default=10.0, help="Dec stripe period, arcmin")
    parser.add_argument(
        "--guard", type=float, default=6.0, help="arcsec of sky kept clear between train and val"
    )
    parser.add_argument(
        "--train-density", type=float, help="train windows per train pixel (overrides --per-patch)"
    )
    parser.add_argument(
        "--val-tile", action="store_true", help="pack non-overlapping val windows (not --per-patch)"
    )
    parser.add_argument(
        "--units", choices=["njy", "paper"], default="njy", help="pixel units (module docstring)"
    )
    parser.add_argument(
        "--hr-sky-subtract",
        action="store_true",
        help="subtract each LSST image's HST sky pedestal (module docstring)",
    )
    parser.add_argument(
        "--flux-filter",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="keep pairs with LOW < ln(sum of stored HR pixels) < HIGH (paper: 2 7, --units paper)",
    )
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
    parser.add_argument(
        "--resume", action="store_true", help="continue an interrupted run into the same --out"
    )
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

    done_dir = out / ".done"
    settings = {
        k: getattr(args, k)
        for k in (
            "lr_size",
            "factor",
            "per_patch",
            "val_frac",
            "seed",
            "reject",
            "split",
            "val_period",
            "guard",
            "train_density",
            "val_tile",
            "hr_zeropoint",
            "units",
            "flux_filter",
            "hr_sky_subtract",
        )
    }
    settings["hr_mosaic"] = sorted(Path(m).name for m in args.hr_mosaic)
    dirs = [out / s / k for s in SPLITS for k in ("lr", "hr")]
    stale = [d for d in dirs if any(d.glob("*.fits"))]
    if (done_dir / "settings.json").exists():
        previous = json.loads((done_dir / "settings.json").read_text())
        if not args.resume:
            raise SystemExit(f"{out} holds an earlier run; pass --resume or delete it first")
        if previous != settings:
            changed = sorted(k for k in settings if previous.get(k) != settings[k])
            raise SystemExit(f"cannot resume {out}: settings differ from its first run ({changed})")
    elif stale:
        raise SystemExit(
            f"{out} already holds pairs ({stale[0]}) with no resume record; move or delete it "
            "first, so cutouts from an earlier run (possibly another split) cannot mix in"
        )
    for split in SPLITS:
        for kind in ("lr", "hr"):
            (out / split / kind).mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(exist_ok=True)
    (done_dir / "settings.json").write_text(json.dumps(settings, indent=1))
    totals = {"train": 0, "val": 0}
    for path in lr_paths:
        marker = done_dir / f"{path.stem}.json"
        if marker.exists():
            counts = json.loads(marker.read_text())
            for split in SPLITS:
                totals[split] += counts[split]
            print(f"{path.name}: done earlier ({counts['train']} train + {counts['val']} val)")
            continue
        for split in SPLITS:  # clear what an interrupted attempt at this image left behind
            for kind in ("lr", "hr"):
                for leftover in (out / split / kind).glob(f"{path.stem}_{split}_*.fits"):
                    leftover.unlink()
        rng = np.random.default_rng([args.seed, zlib.crc32(path.name.encode())])
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
            split_mode=args.split,
            val_period=args.val_period,
            guard=args.guard,
            train_density=args.train_density,
            val_tile=args.val_tile,
            units=args.units,
            flux_filter=args.flux_filter,
            hr_sky_subtract=args.hr_sky_subtract,
        )
        for split in SPLITS:
            totals[split] += counts[split]
        marker.write_text(json.dumps(counts))
        print(
            f"{path.name}: {counts['valid']} valid windows, wrote {counts['train']} train + "
            f"{counts['val']} val, {counts['filtered']} filtered, HR sky "
            f"{counts['hr_sky']:.2e} ({counts.get('hr_sky_sigma', 0):+.2f} sigma) removed "
            f"({time.time() - t0:.0f} s)"
        )
    size_gb = sum(totals.values()) * pair_bytes / 1e9
    print(f"{totals['train']} train + {totals['val']} val pairs -> {out} (~{size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
