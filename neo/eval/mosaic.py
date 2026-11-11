"""Super-resolve a whole sky region as one mosaic with morpheus-core, as the NEO paper did.

Out-of-sample test: the model never saw this sky (e.g. ECDFS/GOODS-S for a COSMOS-trained model).
Three steps, each a subcommand:

  lr        Stitch one tract's LSST deep_coadd patches into a single LR image around (RA, Dec).
            Patches of a tract share one pixel grid (each header's alternate WCS 'A' gives the
            patch origin xy0 in the tract), so every LR pixel comes from the patch whose 3000 px
            inner region holds it, copied without resampling. Values are converted exactly as the
            training pairs were (neo.preprocess.pairs --units paper): HSC counts at AB zeropoint 27
            per 0.168" pixel of equal surface brightness. The side lengths are rounded up to a
            multiple of the stride, so morpheus-core's windows tile the image exactly (its variance
            finalisation assumes this; see check_tiling).
  sr        The paper's mosaic recipe (its 'Color Image' notebook): morpheus_core.predict with
            100 x 100 LR windows, stride 20, dilation 6 and MEAN_VAR aggregation, so every HR pixel
            away from the edges is the mean of 25 overlapping window predictions. Each window goes
            through the training preprocessing (clip at 0 and its own 99.999th percentile, ds9 log
            scale, 14 px reflection pad to 128), the generator (in train mode at batch 1 by default,
            as the paper generated its SR images), then back to the HR pair units (central 600 px,
            inverse log scale). --tta rot4 runs every window four times, rotated by 0/90/180/270
            degrees, rotates each output back (exact: a 6 x 6 HR block stays on its LR pixel) and
            hands morpheus the per-pixel median (or mean) of the four in linear flux.
            --negative-noise paper (the default) then restores the sky noise that training's clip
            at zero removed, as the paper did in its Crop-InvScale step (Sec. 3; its notebook's
            add_negative_noise): in each window, pixels exactly 0 (the generator's tanh saturated
            at -1) become minus a random draw from the window's positive pixels outside sources
            (photutils segmentation above mean + 3 std, 5 px). This happens per window, before
            morpheus averages the overlapping windows. A model trained with [DATASET] sky floors
            (neo/data/floors.py) is run with its own clip and signed log scale, predicts the
            negative sky itself and needs --negative-noise none.
  finalize  (run at the end of sr; rerun on its own if needed) turns morpheus's output.fits/n.fits
            into nJy images with the HR WCS: sr_mean (the mosaic), sr_std (spread of the window
            predictions), sr_n (windows per pixel) and sr_paper (the paper's display image: where
            n > 1 a draw from N(mean, var), as its notebook did) and sr_flag (2 where the LR
            pixel is bad, 1 where a window covering it held a bad LR pixel, else 0). The bad LR
            pixels (the BAD extension of the lr step) went into the model as zeros, so sr_mean,
            sr_std and sr_paper are NaN on them and suspect where sr_flag is 1.

Units: LR stored values carry NJYPERPX like the pair cutouts; HR outputs are written in nJy per HR
pixel (0.2"/6), from the training HR units (e-/s per 0.03" pixel at the F814W AB zeropoint 25.94).
"""

import argparse
import configparser
import glob
import json
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from neo.preprocess.grid import upsampled_wcs

FACTOR = 6
WINDOW = 100  # LR window, as the paper and the training cutouts (centre crop 100)
STRIDE = 20  # the paper's mosaic stride
PAD = 14  # reflection pad 100 -> 128 (neo.data.dataset square_pad_lr)
PATCH_BORDER = 150  # DP2 deep_coadd patches: 3300 px outer, 3000 px inner, 150 px border
TTA_CHOICES = ("none", "rot4")
REDUCE_CHOICES = ("median", "mean")
NEGATIVE_NOISE_CHOICES = ("paper", "none")
PRODUCTS = ("sr_mean", "sr_std", "sr_n", "sr_paper", "sr_flag")
# everything a run writes into its output directory
MOSAIC_FILES = ("output.fits", "n.fits", "mosaic.json", "preview.png") + tuple(
    f"{name}.fits" for name in PRODUCTS
)
FLAG_NEAR_BAD = 1  # some window covering the pixel held a bad LR pixel (zeros the model saw)
FLAG_BAD = 2  # the LR pixel itself is bad (no patch, a rejected mask plane, or non-finite)


# ── LR mosaic from one tract's patches ─────────────────────────────────────────────────────────


def patch_origin(header) -> tuple[int, int]:
    """(x0, y0) of a patch's pixel (0, 0) in its tract, from the alternate WCS 'A' (LTV style)."""
    x0 = header["CRVAL1A"] - (header.get("CRPIX1A", 1.0) - 1.0)
    y0 = header["CRVAL2A"] - (header.get("CRPIX2A", 1.0) - 1.0)
    if abs(x0 - round(x0)) > 1e-6 or abs(y0 - round(y0)) > 1e-6:
        raise ValueError(f"patch origin is not an integer pixel: {x0}, {y0}")
    return int(round(x0)), int(round(y0))


def tract_wcs(header) -> WCS:
    """WCS of the tract pixel grid (tract pixel (0, 0) at array index (0, 0))."""
    x0, y0 = patch_origin(header)
    wcs = WCS(header).celestial.deepcopy()
    wcs.wcs.crpix = wcs.wcs.crpix + np.array([x0, y0])
    return wcs


def round_up(n: int, multiple: int) -> int:
    return int(np.ceil(n / multiple) * multiple)


def check_tiling(shape) -> None:
    """morpheus-core marks a pixel's variance final using `H % stride`; with H and W multiples of
    the stride (and at least one window) the windows tile the image exactly and every pixel is
    finalised once."""
    for n in shape:
        if n < WINDOW or n % STRIDE or (n - WINDOW) % STRIDE:
            raise ValueError(
                f"LR shape {tuple(shape)} must be >= {WINDOW} and a multiple of {STRIDE} per side"
            )


def build_lr(
    patch_paths, ra, dec, size_arcmin, reject=("NO_DATA",), units_scale=None, border=PATCH_BORDER
):
    """Stitch the patches (one tract) into an LR image of side >= size_arcmin centred on (ra, dec).

    Returns (stored image, bad-pixel mask, WCS, header cards). Stored units are the pairs'
    --units paper LR units; pixels no patch covers, or flagged with a `reject` plane, are 0 in the
    image and True in the mask.
    """
    from neo.preprocess.pairs import HSC_PIXEL_ARCSEC, HSC_ZEROPOINT
    from neo.surveys.hst.mosaic import njy_per_count
    from neo.surveys.rubin.coadd import flagged_pixels, load_coadd

    headers = []
    for path in patch_paths:
        with fits.open(path) as hdul:
            headers.append(hdul["IMAGE"].header.copy())
    if len({h["CRVAL1"] for h in headers}) != 1 or len({h["CRVAL2"] for h in headers}) != 1:
        raise ValueError("patches come from more than one tract (different CRVAL); pass one tract")
    wcs = tract_wcs(headers[0])
    for header, path in zip(headers, patch_paths, strict=True):
        if not np.allclose(tract_wcs(header).wcs.crpix, wcs.wcs.crpix, atol=1e-6):
            raise ValueError(f"{path}: patch WCS does not sit on the tract grid")

    lr_pixel = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600
    side = round_up(int(np.ceil(size_arcmin * 60 / lr_pixel)), STRIDE)
    side = max(side, WINDOW)
    cx, cy = wcs.world_to_pixel_values(ra, dec)
    x0, y0 = int(round(float(cx))) - side // 2, int(round(float(cy))) - side // 2
    image = np.zeros((side, side), dtype=np.float32)
    bad = np.ones((side, side), dtype=bool)
    bunits = set()
    for header, path in zip(headers, patch_paths, strict=True):
        px, py = patch_origin(header)
        ny, nx = header["NAXIS2"], header["NAXIS1"]
        # inner region in tract pixels, intersected with the box
        ix0, ix1 = max(px + border, x0), min(px + nx - border, x0 + side)
        iy0, iy1 = max(py + border, y0), min(py + ny - border, y0 + side)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        coadd = load_coadd(path)
        bunits.add(coadd.bunit)
        sub = (slice(iy0 - py, iy1 - py), slice(ix0 - px, ix1 - px))
        dest = (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))
        image[dest] = coadd.image[sub]
        bad[dest] = flagged_pixels(coadd, reject)[sub] if reject else False
    if bunits - {"nJy"}:
        raise ValueError(f"expected nJy coadds, got BUNIT {bunits}")
    bad |= ~np.isfinite(image)
    image[bad] = 0.0
    if units_scale is None:
        units_scale = (HSC_PIXEL_ARCSEC / lr_pixel) ** 2 / njy_per_count(HSC_ZEROPOINT)
    stored = (image * units_scale).astype(np.float32)
    sub_wcs = wcs[y0 : y0 + side, x0 : x0 + side]
    cards = {
        "BUNIT": "HSC count (ZP 27) per 0.168as px",
        "NJYPERPX": 1.0 / units_scale,
        "TRACTX0": x0,
        "TRACTY0": y0,
        "CENRA": ra,
        "CENDEC": dec,
        "NPATCH": len(patch_paths),
        "BADFRAC": float(bad.mean()),
    }
    return stored, bad, sub_wcs, cards


def write_lr(path, stored, bad, wcs, cards) -> None:
    header = wcs.to_header()
    for key, value in cards.items():
        header[key] = value
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(stored, header=header),
            fits.ImageHDU(bad.astype(np.uint8), header=wcs.to_header(), name="BAD"),
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(path, overwrite=True)


# ── per-window model ───────────────────────────────────────────────────────────────────────────


def window_to_input(window: np.ndarray, floor: float = 0.0):
    """One 100 x 100 LR window (stored units) -> the generator's (1, 1, 128, 128) input, through
    the training dataset's own clip / ds9 scale / reflection-pad steps (neo.data.dataset), with
    the run's [DATASET] lr_floor (neo/data/floors.py; 0 = the paper's clip at zero)."""
    import torchvision.transforms as transforms

    from neo.data.dataset import SquarePad, SR_HST_HSC_Dataset

    window = np.asarray(window, dtype=np.float32)
    scaled = SR_HST_HSC_Dataset.floor_scaling(window, floor).astype(np.float32)
    pad = transforms.Compose([transforms.ToPILImage(mode="F"), SquarePad(PAD, "reflect")])
    return transforms.ToTensor()(pad(scaled)).unsqueeze(0)  # (1, 1, 128, 128)


def add_negative_noise(window: np.ndarray, rng) -> int:
    """The paper's add_negative_noise, in place on one HR window in linear units: every pixel
    exactly 0 becomes -|draw|, the draws taken from the window's positive pixels outside sources
    (photutils detect_sources above mean + 3 std with 5 connected px; as the paper, a 5e-5
    threshold when that finds nothing). Returns the number of pixels filled."""
    from photutils.segmentation import detect_sources

    zeros = window == 0
    n_zero = int(zeros.sum())
    if not n_zero:
        return 0
    sources = np.zeros(window.shape, dtype=bool)
    for threshold in (window.mean() + 3 * window.std(), 5e-5):
        segm = detect_sources(window, threshold, 5)
        if segm is not None:
            sources = segm.data > 0
            break
    pool = window[~sources & (window > 0)]
    if pool.size == 0:  # nothing positive outside sources: leave the zeros
        return 0
    window[zeros] = -np.abs(rng.choice(pool, size=n_zero))
    return n_zero


def make_window_model(
    generate,
    tta: str = "none",
    reduce: str = "median",
    batch: int = 1,
    negative_noise: str = "none",
    seed: int = 0,
    hr_floor: float = 0.0,
    lr_floor: float = 0.0,
):
    """The callable morpheus-core calls: (B, 100, 100, 1) LR windows -> (B, 600, 600, 1) HR
    windows in the HR pair units.

    `generate` maps a (b, 1, 128, 128) tensor to the (b, 1, 768, 768) log-space output. Windows
    (and, with rot4, each rotation) are run `batch` at a time; batch 1 keeps train-mode batch
    norm on each window's own statistics, as the paper did. negative_noise="paper" applies
    add_negative_noise to each window's final (reduced) prediction; the returned callable's
    `stats` dict counts the windows and the pixels it filled. hr_floor / lr_floor: the run's
    [DATASET] sky floors (neo/data/floors.py); the paper's zero refill needs floors of 0.
    """
    import torch

    from neo.eval.postprocess import to_physical

    if tta not in TTA_CHOICES:
        raise ValueError(f"tta must be one of {TTA_CHOICES}")
    if reduce not in REDUCE_CHOICES:
        raise ValueError(f"reduce must be one of {REDUCE_CHOICES}")
    if negative_noise not in NEGATIVE_NOISE_CHOICES:
        raise ValueError(f"negative_noise must be one of {NEGATIVE_NOISE_CHOICES}")
    if negative_noise == "paper" and (hr_floor or lr_floor):
        raise ValueError(
            "negative_noise 'paper' refills the zeros of the paper's clip at 0; a model trained "
            "with sky floors predicts negative sky itself: use negative_noise 'none'"
        )
    turns = (0, 1, 2, 3) if tta == "rot4" else (0,)
    rng = np.random.default_rng(seed)
    stats = {"windows": 0, "windows_with_zeros": 0, "zero_pixels": 0, "max_zero_frac": 0.0}

    def run(inputs):
        outs = []
        for start in range(0, len(inputs), batch):
            outs.append(generate(torch.cat(inputs[start : start + batch])).float().cpu())
        return torch.cat(outs)

    def model(batch_inputs):
        # morpheus passes one (B, 100, 100, 1) batch per input image; there is one input
        windows = np.asarray(batch_inputs[0])[..., 0]
        base = [window_to_input(w, lr_floor) for w in windows]
        # one (len(turns), ...) stack per window, rotated inputs ordered window-major
        inputs = [torch.rot90(x, k, dims=(-2, -1)) for x in base for k in turns]
        outputs = run(inputs)
        hr = []
        for i in range(len(windows)):
            per_turn = [
                torch.rot90(outputs[i * len(turns) + j], -k, dims=(-2, -1)).numpy()[0]
                for j, k in enumerate(turns)
            ]
            # (turns, 600, 600), linear HR units
            physical = to_physical(np.stack(per_turn), hr_floor)
            if len(turns) == 1:
                window = physical[0]
            elif reduce == "median":
                window = np.median(physical, axis=0)
            else:
                window = np.mean(physical, axis=0)
            stats["windows"] += 1
            if negative_noise == "paper":
                filled = add_negative_noise(window, rng)
                if filled:
                    stats["windows_with_zeros"] += 1
                    stats["zero_pixels"] += filled
                    stats["max_zero_frac"] = max(stats["max_zero_frac"], filled / window.size)
            hr.append(window)
        return np.stack(hr).astype(np.float32)[..., np.newaxis]

    model.stats = stats
    return model


def hr_njy_per_px(lr_wcs: WCS) -> float:
    """nJy per HR pixel per model output unit (e-/s per 0.03" pixel at F814W AB 25.94)."""
    from neo.preprocess.pairs import PAPER_HST_PIXEL_ARCSEC
    from neo.surveys.hst.mosaic import F814W_AB_ZEROPOINT, njy_per_count

    hr_pixel = float(np.mean(proj_plane_pixel_scales(lr_wcs))) * 3600 / FACTOR
    return njy_per_count(F814W_AB_ZEROPOINT) * (hr_pixel / PAPER_HST_PIXEL_ARCSEC) ** 2


def lr_flag(bad: np.ndarray) -> np.ndarray:
    """Per-LR-pixel flag: FLAG_BAD on bad pixels, FLAG_NEAR_BAD where at least one of the
    morpheus windows covering the pixel (WINDOW px, every STRIDE px, as check_tiling requires)
    contains a bad pixel, else 0."""
    bad = np.asarray(bad, dtype=bool)
    check_tiling(bad.shape)
    height, width = bad.shape
    # int32 holds both sums for any LR image under 2**31 px (a 7200^2 box: ~0.2 GB per array)
    integral = np.zeros((height + 1, width + 1), dtype=np.int32)
    integral[1:, 1:] = bad.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    ys = np.arange(0, height - WINDOW + 1, STRIDE)
    xs = np.arange(0, width - WINDOW + 1, STRIDE)
    y0, x0 = ys[:, None], xs[None, :]
    per_window = (
        integral[y0 + WINDOW, x0 + WINDOW]
        - integral[y0, x0 + WINDOW]
        - integral[y0 + WINDOW, x0]
        + integral[y0, x0]
    )
    # paint every window that holds a bad pixel (2-D difference array)
    paint = np.zeros((height + 1, width + 1), dtype=np.int32)
    for iy, ix in zip(*np.nonzero(per_window), strict=True):
        ya, xa = ys[iy], xs[ix]
        paint[ya, xa] += 1
        paint[ya, xa + WINDOW] -= 1
        paint[ya + WINDOW, xa] -= 1
        paint[ya + WINDOW, xa + WINDOW] += 1
    near = paint.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)[:height, :width] > 0
    flag = np.where(near, FLAG_NEAR_BAD, 0).astype(np.uint8)
    flag[bad] = FLAG_BAD
    return flag


def read_bad(lr_path) -> np.ndarray:
    """The lr step's BAD extension (an LR image without one: nothing flagged, with a warning)."""
    with fits.open(lr_path) as hdul:
        if "BAD" in hdul:
            return np.asarray(hdul["BAD"].data, dtype=bool)
        print(f"warning: {lr_path} has no BAD extension; no pixel flagged", flush=True)
        return np.zeros(hdul[0].data.shape, dtype=bool)


# ── finalize ───────────────────────────────────────────────────────────────────────────────────


def finalize(out_dir, lr_path, cards=None, rows_per_chunk: int = 600, seed: int = 0) -> dict:
    """morpheus output.fits (H, W, 1, [mean, var]) and n.fits -> nJy products with the HR WCS.

    Written in row chunks (StreamingHDU), so a mosaic of several Gpix never sits in memory.
    Bad LR pixels (the lr step's BAD extension) are NaN in sr_mean / sr_std / sr_paper; sr_flag
    marks them (FLAG_BAD) and the pixels a window holding one of them touched (FLAG_NEAR_BAD).
    Statistics are over the good pixels.
    """
    out_dir = Path(out_dir)
    lr_header = fits.getheader(lr_path)
    lr_wcs = WCS(lr_header).celestial
    hr_wcs = upsampled_wcs(lr_wcs, FACTOR)
    scale = hr_njy_per_px(lr_wcs)
    rng = np.random.default_rng(seed)
    flag_lr = lr_flag(read_bad(lr_path))
    # whole LR rows per chunk, so the flag upsamples by plain repetition
    rows_per_chunk = max(FACTOR, rows_per_chunk // FACTOR * FACTOR)
    with (
        fits.open(out_dir / "output.fits", memmap=True) as out_hdul,
        fits.open(out_dir / "n.fits", memmap=True) as n_hdul,
    ):
        mv, n = out_hdul[0].data, n_hdul[0].data
        height, width = n.shape
        if (height, width) != (flag_lr.shape[0] * FACTOR, flag_lr.shape[1] * FACTOR):
            raise ValueError(f"mosaic {n.shape} is not {FACTOR}x the LR image {flag_lr.shape}")
        header = fits.PrimaryHDU(np.zeros((1, 1), dtype=np.float32)).header
        header["NAXIS1"], header["NAXIS2"] = width, height
        header.extend(hr_wcs.to_header().cards)
        header["NJYSCALE"] = (scale, "nJy per HR px per e-/s per 0.03as px")
        for key, value in (cards or {}).items():
            header[key] = value
        header["FLAGNEAR"] = (FLAG_NEAR_BAD, "sr_flag: window held a bad LR px")
        header["FLAGBAD"] = (FLAG_BAD, "sr_flag: LR px bad (SR is NaN)")
        streams = {}
        for name in PRODUCTS:
            if name == "sr_flag":
                h = fits.PrimaryHDU(np.zeros((1, 1), dtype=np.uint8)).header
                h["NAXIS1"], h["NAXIS2"] = width, height
                h.extend(c for c in header.cards if c.keyword not in h)
            else:
                h = header.copy()
            h["BUNIT"] = {"sr_n": "count", "sr_flag": ""}.get(name, "nJy")
            h["PRODUCT"] = name
            path = out_dir / f"{name}.fits"
            path.unlink(missing_ok=True)
            streams[name] = fits.StreamingHDU(path, h)
        stats = {"pixels": 0, "good": 0, "near_bad": 0, "no_window": 0, "flux_mean_njy": 0.0}
        for y in range(0, height, rows_per_chunk):
            rows = slice(y, min(y + rows_per_chunk, height))
            lr_rows = flag_lr[y // FACTOR : rows.stop // FACTOR]
            flag = np.repeat(np.repeat(lr_rows, FACTOR, axis=0), FACTOR, axis=1)
            mean = np.asarray(mv[rows, :, 0, 0], dtype=np.float64)
            var = np.clip(np.asarray(mv[rows, :, 0, 1], dtype=np.float64), 0, None)
            count = np.asarray(n[rows], dtype=np.float32)
            draw = mean.copy()
            many = count > 1
            draw[many] = rng.normal(mean[many], np.sqrt(var[many]))
            std = np.sqrt(var)
            bad = flag == FLAG_BAD
            for image in (mean, std, draw):
                image[bad] = np.nan
            streams["sr_mean"].write((mean * scale).astype(np.float32))
            streams["sr_std"].write((std * scale).astype(np.float32))
            streams["sr_n"].write(count)
            streams["sr_paper"].write((draw * scale).astype(np.float32))
            streams["sr_flag"].write(flag)
            good = ~bad
            stats["pixels"] += count.size
            stats["good"] += int(good.sum())
            stats["near_bad"] += int((flag == FLAG_NEAR_BAD).sum())
            stats["no_window"] += int((good & (count == 0)).sum())
            stats["flux_mean_njy"] += float((mean[good] * scale).sum())
        for stream in streams.values():
            stream.close()
    stats["flux_mean_njy"] /= max(stats["good"], 1)
    stats["coverage"] = stats["good"] / max(stats["pixels"], 1)  # 1 - the LR BADFRAC
    stats["near_bad_frac"] = stats["near_bad"] / max(stats["pixels"], 1)
    if stats["no_window"]:
        raise RuntimeError(f"{stats['no_window']} good pixels got no window prediction")
    return stats


def preview(out_dir, lr_path, max_px: int = 2400) -> Path:
    """LR | SR mean side by side (SR subsampled to <= max_px), one log scale in nJy/arcsec²."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    lr = fits.getdata(lr_path, 0).astype(np.float64)
    lr_njy = lr * fits.getheader(lr_path)["NJYPERPX"]
    with fits.open(out_dir / "sr_mean.fits", memmap=True) as hdul:
        sr = hdul[0].data
        step = max(1, int(np.ceil(max(sr.shape) / max_px)))
        sr_small = np.asarray(sr[::step, ::step], dtype=np.float64)
    lr_wcs = WCS(fits.getheader(lr_path)).celestial
    lr_area = (float(np.mean(proj_plane_pixel_scales(lr_wcs))) * 3600) ** 2
    sr_area = lr_area / FACTOR**2
    lr_sb, sr_sb = lr_njy / lr_area, sr_small / sr_area
    vmin, vmax = np.percentile(sr_sb[np.isfinite(sr_sb)], [5, 99.8])
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, img, title in zip(axes, (lr_sb, sr_sb), ("LSST i (LR)", "NEO SR mosaic"), strict=True):
        ax.imshow(
            np.log10(np.clip(img, vmin, None) - vmin + 1e-3 * (vmax - vmin)),
            origin="lower",
            cmap="gray_r",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    path = out_dir / "preview.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────


def cmd_lr(args) -> None:
    paths = sorted(p for pattern in args.patches for p in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no patch files match {args.patches}")
    stored, bad, wcs, cards = build_lr(
        paths, args.ra, args.dec, args.size_arcmin, reject=tuple(args.reject)
    )
    write_lr(args.out, stored, bad, wcs, cards)
    print(
        f"LR {stored.shape[1]} x {stored.shape[0]} px from {len(paths)} patches -> {args.out} "
        f"(bad/uncovered {cards['BADFRAC']:.2%})"
    )


def cmd_sr(args) -> None:
    import torch
    from morpheus_core import morpheus_core

    from neo.data.floors import any_floor, floors
    from neo.eval.predict import pick_device
    from neo.eval.predictors import build_predictor

    if args.gen_mode == "train" and args.gen_batch != 1:
        raise SystemExit("--gen-mode train uses per-window batch statistics: use --gen-batch 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    # every check before anything in `out` is touched, so a refused run destroys nothing
    lr = fits.getdata(args.lr, 0).astype(np.float32)
    check_tiling(lr.shape)
    badfrac = float(read_bad(args.lr).mean())
    print(f"LR bad/uncovered fraction {badfrac:.3%}", flush=True)
    if badfrac > args.max_badfrac:
        raise SystemExit(
            f"{badfrac:.2%} of the LR image is bad or uncovered (> --max-badfrac "
            f"{args.max_badfrac:.2%}): check the patches cover the box, or raise --max-badfrac"
        )
    previous = [out / name for name in MOSAIC_FILES if (out / name).exists()]
    if previous and not args.overwrite:
        raise SystemExit(f"{out} holds a mosaic ({previous[0].name}); pass --overwrite to redo it")
    config = configparser.ConfigParser()
    config.read(args.config)
    device = pick_device(args.device)
    sky_floors = floors(config)
    if any_floor(sky_floors) and args.negative_noise == "paper":
        raise SystemExit(
            f"{args.config} trains with sky floors {sky_floors}: the model predicts negative sky "
            "itself, so pass --negative-noise none (mosaic.sbatch: NEG_NOISE=none)"
        )
    predict = build_predictor(config, args.checkpoint, device, mode=args.gen_mode)
    model = make_window_model(
        lambda x: predict(x, None),
        tta=args.tta,
        reduce=args.reduce,
        batch=args.gen_batch,
        negative_noise=args.negative_noise,
        seed=args.seed,
        **sky_floors,
    )
    # only now, after the config, floor and checkpoint checks (build_predictor refuses another
    # generator's or another floor's checkpoint), is the old run removed
    for path in previous:  # all of the old run, so no old product outlives a failed rerun
        path.unlink()
    out.mkdir(parents=True, exist_ok=True)
    n_windows = ((lr.shape[0] - WINDOW) // STRIDE + 1) * ((lr.shape[1] - WINDOW) // STRIDE + 1)
    passes = 4 if args.tta == "rot4" else 1
    print(
        f"device {device} | LR {lr.shape} -> HR {(lr.shape[0] * FACTOR, lr.shape[1] * FACTOR)} | "
        f"{n_windows} windows x {passes} pass(es) | tta {args.tta} ({args.reduce}) | "
        f"gen mode {args.gen_mode} | negative noise {args.negative_noise}",
        flush=True,
    )
    meta = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "lr": str(args.lr),
        "tta": args.tta,
        "reduce": args.reduce,
        "negative_noise": args.negative_noise,
        "gen_mode": args.gen_mode,
        "gen_batch": args.gen_batch,
        "seed": args.seed,
        "window": WINDOW,
        "stride": STRIDE,
        "dilation": FACTOR,
        "aggregate": "mean_var",
        "windows": n_windows,
        "lr_badfrac": badfrac,
        **sky_floors,
    }
    (out / "mosaic.json").write_text(json.dumps(meta, indent=1))
    experiment = None
    if not args.no_comet:
        from neo.eval.tracking import start_experiment

        run = Path(args.checkpoint).parent.name
        experiment = start_experiment(
            f"mosaic {out.name} {run} {Path(args.checkpoint).stem} tta={args.tta}",
            ["mosaic", f"tta_{args.tta}", run, *args.tag],
        )
        experiment.log_parameters(meta)
    start = time.time()
    hduls, _ = morpheus_core.predict(
        model,
        [lr],
        1,
        args.batch_size,
        (WINDOW, WINDOW),
        dilation=FACTOR,
        stride=(STRIDE, STRIDE),
        aggregate_method=morpheus_core.AGGREGATION_METHODS.MEAN_VAR,
        out_dir=str(out),
    )
    for hdul in hduls:
        if hdul is not None:
            hdul.close()
    meta["predict_seconds"] = round(time.time() - start, 1)
    zero_stats = dict(model.stats)
    zero_stats["zero_frac"] = zero_stats["zero_pixels"] / max(
        zero_stats["windows"] * (WINDOW * FACTOR) ** 2, 1
    )
    meta["negative_noise_stats"] = zero_stats
    print(f"negative noise: {zero_stats}", flush=True)
    cards = {
        "NEOCKPT": Path(args.checkpoint).name[:68],
        "NEORUN": Path(args.checkpoint).parent.name[:68],
        "NEOTTA": args.tta,
        "NEOREDUC": args.reduce,
        "NEONEG": args.negative_noise,
        "NEOGMODE": args.gen_mode,
        "NEOHRFL": (sky_floors["hr_floor"], "[DATASET] hr_floor the model was trained with"),
        "NEOLRFL": (sky_floors["lr_floor"], "[DATASET] lr_floor the model was trained with"),
        "MSTRIDE": STRIDE,
        "MWINDOW": WINDOW,
    }
    meta["finalize"] = finalize(out, args.lr, cards=cards, seed=args.seed)
    meta["preview"] = str(preview(out, args.lr))
    (out / "mosaic.json").write_text(json.dumps(meta, indent=1))
    if experiment is not None:
        experiment.log_metrics(
            meta["finalize"]
            | {"predict_seconds": meta["predict_seconds"]}
            | {f"negnoise_{k}": v for k, v in zero_stats.items()}
        )
        experiment.log_image(meta["preview"], name="preview")
        experiment.log_asset(str(out / "mosaic.json"))
        experiment.end()
    print(json.dumps(meta, indent=1))


def cmd_finalize(args) -> None:
    meta_path = Path(args.out) / "mosaic.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["finalize"] = finalize(args.out, args.lr, seed=args.seed)
    meta["preview"] = str(preview(args.out, args.lr))
    meta_path.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["finalize"], indent=1))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("lr", help="stitch one tract's patches into the LR image")
    p.add_argument("--patches", nargs="+", required=True, help="deep_coadd FITS files or globs")
    p.add_argument("--ra", type=float, required=True, help="centre RA, deg")
    p.add_argument("--dec", type=float, required=True, help="centre Dec, deg")
    p.add_argument("--size-arcmin", type=float, required=True, help="side length, arcmin")
    p.add_argument("--reject", nargs="*", default=["NO_DATA"], help="mask planes to zero out")
    p.add_argument("--out", required=True, help="output LR FITS")
    p.set_defaults(func=cmd_lr)

    p = sub.add_parser("sr", help="super-resolve the LR image with morpheus-core")
    p.add_argument("--config", required=True, help="the .ini the model was trained with")
    p.add_argument("--checkpoint", required=True, help="step_*.pt")
    p.add_argument("--lr", required=True, help="LR FITS from the lr step")
    p.add_argument("--out", required=True, help="output directory for this mosaic")
    p.add_argument("--tta", choices=TTA_CHOICES, default="none", help="rot4: 4 rotations/window")
    p.add_argument("--reduce", choices=REDUCE_CHOICES, default="median", help="rot4 combination")
    p.add_argument(
        "--negative-noise",
        choices=NEGATIVE_NOISE_CHOICES,
        default="paper",
        help="paper: refill each window's exact zeros with negated sky noise (paper Sec. 3)",
    )
    p.add_argument("--gen-mode", choices=["eval", "train"], default="train")
    p.add_argument("--gen-batch", type=int, default=1, help="windows per generator call")
    p.add_argument("--batch-size", type=int, default=16, help="windows per morpheus batch")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--max-badfrac",
        type=float,
        default=0.02,
        help="refuse an LR image with more bad/uncovered pixels than this fraction",
    )
    p.add_argument("--tag", action="append", default=[], help="extra Comet tag (repeatable)")
    p.add_argument("--no-comet", action="store_true", help="do not track the run on Comet")
    p.set_defaults(func=cmd_sr)

    p = sub.add_parser("finalize", help="rewrite the nJy products from output.fits / n.fits")
    p.add_argument("--lr", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_finalize)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
