"""Gaia DR3 stars in the COSMOS field: where each one sits in a cutout at the HST and LSST epochs.

Stars move between the two surveys. COSMOS ACS F814W was taken Oct 2003 - Nov 2005 (Cycle 12 Oct
2003 - Jun 2004, Cycle 13 Oct 2004 - Nov 2005). HST_EPOCH = 2004.3 is the project's convention;
the orbit-weighted midpoint of the campaign is ~2004.8, and the 0.5 yr difference moves the median
star by 5 mas and the fastest catalog star by 0.12", both inside the 0.3" minimum mask radius. The
LSST DP2 i-band visits ran Apr 2025 - Jan 2026 (LSST_EPOCH = 2025.5; the midpoint is ~2025.7, a
2 mas difference for the median star). Over those ~21 years Gaia DR3 proper motions give a median
motion of 0.2" (6 HR px) and 13% of the stars move more than 0.5", so a star is at one place in
the LR input and at another in the HST target: an SR image that puts it where the LR image shows
it reads as one ghost plus one miss in the realism check and as a misaligned source in Table 4.
This module gives the catalog stars' pixel positions in a cutout at either epoch (star_positions)
and a mask covering a disk around BOTH positions (star_mask), so that neither the moved star nor
the place HST shows it is counted.

Catalog: neo/data/gaia_dr3_cosmos_stars.csv, fetched once with
    python -m neo.eval.stars fetch --out neo/data/gaia_dr3_cosmos_stars.csv
Gaia DR3 (gaiadr3.gaia_source via TAP at gea.esac.esa.int): sources with parallax > 0.5 mas or
proper motion > 3 mas/yr in BOX, which covers the COSMOS-Web pairs (RA 149.4-150.8, Dec 1.6-2.8)
and the local CANDELS-strip pairs (RA 150.0-150.2, Dec 2.1-2.6). The Lux compute nodes have no
network, so the file ships with the code and nothing here queries anything at run time. Positions
are propagated linearly from ref_epoch (2016.0); the parallactic wobble is at most the parallax,
a few mas for these stars against 33 mas per HR pixel, and is ignored. A cutout without a WCS or
outside the catalog box gets an empty mask and Tracks.notice says why.
"""

import argparse
import csv
import functools
import sys
from pathlib import Path

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

CATALOG = Path(__file__).resolve().parents[1] / "data" / "gaia_dr3_cosmos_stars.csv"
BOX = (149.3, 151.0, 1.5, 2.9)  # RA min, RA max, Dec min, Dec max (deg)
HST_EPOCH = 2004.3  # COSMOS ACS F814W, Oct 2003 - Nov 2005
LSST_EPOCH = 2025.5  # LSST DP2 i visits, Apr 2025 - Jan 2026
MIN_PARALLAX = 0.5  # mas
MIN_PM = 3.0  # mas / yr
MIN_RADIUS_ARCSEC = 0.3  # the mask is at least this wide (3 HST PSF FWHM) around each position
MOTION_FACTOR = 1.5  # and at least this many times the star's motion between the epochs
TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
COLUMNS = ("ra", "dec", "pmra", "pmdec", "parallax", "phot_g_mean_mag", "ref_epoch")


class Catalog:
    """The star catalog: arrays of COLUMNS plus the box it was queried in."""

    def __init__(self, rows: np.ndarray, box, query: str = ""):
        self.rows = rows
        self.box = tuple(float(v) for v in box)
        self.query = query

    def __len__(self):
        return len(self.rows)

    @classmethod
    def read(cls, path=None) -> "Catalog":
        """Read a catalog CSV; the box (and query) come from its '# key: value' header lines."""
        path = Path(CATALOG if path is None else path)
        meta = {}
        with open(path, newline="") as f:
            lines = f.readlines()
        body = []
        for line in lines:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                meta[key.strip()] = value.strip()
            else:
                body.append(line)
        rows = np.array(
            [
                [float(row[c]) if row[c] != "" else np.nan for c in COLUMNS]
                for row in csv.DictReader(body)
            ],
            dtype=np.float64,
        ).reshape(-1, len(COLUMNS))
        box = [float(v) for v in meta.get("box", ",".join(map(str, BOX))).split(",")]
        return cls(rows, box, meta.get("query", ""))

    def column(self, name: str) -> np.ndarray:
        return self.rows[:, COLUMNS.index(name)]

    def covers(self, ra: float, dec: float) -> bool:
        ra_lo, ra_hi, dec_lo, dec_hi = self.box
        return ra_lo <= ra <= ra_hi and dec_lo <= dec <= dec_hi

    def at(self, epoch: float):
        """(ra, dec) in degrees of every star at `epoch`, proper motion propagated linearly from
        ref_epoch; a star without a proper motion is left where the catalog has it."""
        ra, dec = self.column("ra"), self.column("dec")
        dt = epoch - self.column("ref_epoch")
        pmra = np.nan_to_num(self.column("pmra"))  # Gaia pmra is already pmra* = d(ra)/dt cos(dec)
        pmdec = np.nan_to_num(self.column("pmdec"))
        mas = 1 / 3.6e6
        return ra + pmra * dt * mas / np.cos(np.radians(dec)), dec + pmdec * dt * mas

    def motion_arcsec(self, epoch_a: float, epoch_b: float) -> np.ndarray:
        """How far every star moves between two epochs, in arcsec."""
        pm = np.hypot(np.nan_to_num(self.column("pmra")), np.nan_to_num(self.column("pmdec")))
        return pm * abs(epoch_b - epoch_a) / 1000


@functools.lru_cache(maxsize=4)
def _cached(path: str, mtime_ns: int, size: int) -> Catalog:
    return Catalog.read(path)


def load_catalog(path=None) -> Catalog:
    """The catalog at `path` (default CATALOG), read once per process (again if the file
    changes). A missing file raises FileNotFoundError naming the remedy: the catalog ships with
    the code (a checkout without it cannot be fixed from a Lux compute node, which has no
    network), so compare.py and sweep.py check for it before any work."""
    path = Path(CATALOG if path is None else path).resolve()
    try:
        stat = path.stat()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Gaia catalog {path} not found: run `python -m neo.eval.stars fetch --out {path}` on "
            "a machine with network and commit the file"
        ) from None
    return _cached(str(path), stat.st_mtime_ns, stat.st_size)


def cutout_wcs(header, shape=None):
    """Celestial WCS of the central `shape` (ny, nx) crop of the image `header` describes (the
    crop postprocess.center_crop makes), or None when the header carries no celestial WCS."""
    try:
        wcs = WCS(header).celestial
    except Exception:  # noqa: BLE001 - a malformed header is "no WCS" here
        return None
    if not wcs.has_celestial or wcs.naxis != 2:
        return None
    if shape is None:
        return wcs
    ny, nx = shape
    full_ny, full_nx = int(header.get("NAXIS2", ny)), int(header.get("NAXIS1", nx))
    y0, x0 = (full_ny - ny) // 2, (full_nx - nx) // 2
    if (y0, x0) == (0, 0):
        return wcs
    return wcs[y0 : y0 + ny, x0 : x0 + nx]


class Tracks:
    """The catalog stars in or near one cutout: pixel positions (0-based x, y) at the HST and
    the LSST epoch, their motion between the two, and the reason when there are none."""

    def __init__(
        self,
        shape,
        pixel_arcsec,
        x_hst,
        y_hst,
        x_lsst,
        y_lsst,
        motion,
        gmag,
        reason=None,
        radius_arcsec=MIN_RADIUS_ARCSEC,
        motion_factor=MOTION_FACTOR,
    ):
        self.shape = tuple(shape)
        self.pixel_arcsec = pixel_arcsec
        self.radius_arcsec, self.motion_factor = radius_arcsec, motion_factor
        self.x_hst, self.y_hst = np.asarray(x_hst, float), np.asarray(y_hst, float)
        self.x_lsst, self.y_lsst = np.asarray(x_lsst, float), np.asarray(y_lsst, float)
        self.motion = np.asarray(motion, float)  # arcsec
        self.gmag = np.asarray(gmag, float)
        self.reason = reason

    def __len__(self):
        return len(self.x_hst)

    @property
    def notice(self):
        """One line to print when the cutout could not be checked against the catalog."""
        if self.reason is None:
            return None
        return f"no Gaia star mask ({self.reason})"

    def radii_px(self, radius_arcsec=None, motion_factor=None):
        """Mask radius per star in pixels: at least `radius_arcsec`, and at least
        `motion_factor` times the star's motion between the epochs (defaults: the values the
        stars were selected with)."""
        radius_arcsec = self.radius_arcsec if radius_arcsec is None else radius_arcsec
        motion_factor = self.motion_factor if motion_factor is None else motion_factor
        return np.maximum(radius_arcsec, motion_factor * self.motion) / self.pixel_arcsec

    def mask(self, radius_arcsec=None, motion_factor=None) -> np.ndarray:
        """Boolean image: a disk of radii_px around both epoch positions of every star."""
        out = np.zeros(self.shape, dtype=bool)
        if not len(self):
            return out
        yy, xx = np.ogrid[: self.shape[0], : self.shape[1]]
        for x0, y0, x1, y1, r in zip(
            self.x_hst,
            self.y_hst,
            self.x_lsst,
            self.y_lsst,
            self.radii_px(radius_arcsec, motion_factor),
            strict=True,
        ):
            for x, y in ((x0, y0), (x1, y1)):
                out |= (xx - x) ** 2 + (yy - y) ** 2 <= r**2
        return out


def star_tracks(
    header,
    shape=None,
    hst_epoch: float = HST_EPOCH,
    lsst_epoch: float = LSST_EPOCH,
    catalog=None,
    radius_arcsec: float = MIN_RADIUS_ARCSEC,
    motion_factor: float = MOTION_FACTOR,
) -> Tracks:
    """Catalog stars whose mask disk (radius_arcsec, motion_factor) reaches into the central
    `shape` crop of the cutout `header` describes (default: the whole image).

    `catalog` is a Catalog, a path, or None for the shipped file. Without a celestial WCS, or
    with the cutout centre outside the catalog box, no star is returned and Tracks.reason says so.
    """
    if shape is None:
        shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
    shape = tuple(int(v) for v in shape)
    empty = ([],) * 6
    wcs = cutout_wcs(header, shape)
    if wcs is None:
        return Tracks(shape, np.nan, *empty, reason="no celestial WCS")
    pixel_arcsec = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600
    cat = catalog if isinstance(catalog, Catalog) else load_catalog(catalog)
    ra_c, dec_c = (float(v) for v in wcs.wcs_pix2world(shape[1] / 2, shape[0] / 2, 0))
    if not cat.covers(ra_c, dec_c):
        reason = f"cutout centre ({ra_c:.3f}, {dec_c:.3f}) outside the catalog box {cat.box}"
        return Tracks(shape, pixel_arcsec, *empty, reason=reason)
    if not len(cat):
        return Tracks(
            shape, pixel_arcsec, *empty, radius_arcsec=radius_arcsec, motion_factor=motion_factor
        )
    # pixel positions at both epochs; a star counts when either position lies within its mask
    # radius of the image, so a disk reaching in from just outside is not lost
    ra0, dec0 = cat.at(hst_epoch)
    ra1, dec1 = cat.at(lsst_epoch)
    x0, y0 = wcs.wcs_world2pix(ra0, dec0, 0)
    x1, y1 = wcs.wcs_world2pix(ra1, dec1, 0)
    motion = cat.motion_arcsec(hst_epoch, lsst_epoch)
    r = np.maximum(radius_arcsec, motion_factor * motion) / pixel_arcsec
    ny, nx = shape

    def inside(x, y):
        return (x >= -r) & (x < nx + r) & (y >= -r) & (y < ny + r)

    keep = np.isfinite(x0) & np.isfinite(x1) & (inside(x0, y0) | inside(x1, y1))
    return Tracks(
        shape,
        pixel_arcsec,
        x0[keep],
        y0[keep],
        x1[keep],
        y1[keep],
        motion[keep],
        cat.column("phot_g_mean_mag")[keep],
        radius_arcsec=radius_arcsec,
        motion_factor=motion_factor,
    )


def star_positions(
    header,
    epoch: float,
    shape=None,
    catalog=None,
    hst_epoch: float = HST_EPOCH,
    lsst_epoch: float = LSST_EPOCH,
    **options,
) -> np.ndarray:
    """(N, 2) pixel positions (x, y; 0-based) at `epoch` of the catalog stars in the central
    `shape` crop of the cutout (those star_tracks finds between hst_epoch and lsst_epoch).

    The position at any epoch lies on the straight track between the two tracked epochs (proper
    motion is linear, and the tangent plane is flat over a cutout), so it is interpolated or
    extrapolated along it. `options` go to star_tracks (radius, motion factor).
    """
    tracks = star_tracks(header, shape, hst_epoch, lsst_epoch, catalog, **options)
    if not len(tracks):
        return np.zeros((0, 2))
    t = (epoch - hst_epoch) / (lsst_epoch - hst_epoch)
    x = tracks.x_hst + t * (tracks.x_lsst - tracks.x_hst)
    y = tracks.y_hst + t * (tracks.y_lsst - tracks.y_hst)
    return np.column_stack([x, y])


def star_mask(
    header,
    shape,
    radius_arcsec: float = MIN_RADIUS_ARCSEC,
    hst_epoch: float = HST_EPOCH,
    lsst_epoch: float = LSST_EPOCH,
    motion_factor: float = MOTION_FACTOR,
    catalog=None,
) -> np.ndarray:
    """Boolean mask of `shape` covering a disk of radius max(radius_arcsec, motion_factor x the
    star's motion between the epochs) around both epoch positions of every catalog star in the
    central `shape` crop of the cutout; all False without a WCS or outside the catalog box."""
    tracks = star_tracks(
        header, shape, hst_epoch, lsst_epoch, catalog, radius_arcsec, motion_factor
    )
    return tracks.mask(radius_arcsec, motion_factor)


def fetch(out: Path, box=BOX, min_parallax=MIN_PARALLAX, min_pm=MIN_PM, url=TAP_URL) -> int:
    """Query Gaia DR3 (asynchronous TAP job, so the archive's synchronous row cap does not apply)
    and write the catalog CSV; returns the number of stars. Needs the network: run once on the
    Mac, commit the file."""
    from pyvo.dal import TAPService

    ra_lo, ra_hi, dec_lo, dec_hi = box
    query = (
        f"SELECT {', '.join(COLUMNS)} FROM gaiadr3.gaia_source "
        f"WHERE ra BETWEEN {ra_lo} AND {ra_hi} AND dec BETWEEN {dec_lo} AND {dec_hi} "
        f"AND (parallax > {min_parallax} OR pm > {min_pm}) ORDER BY ra"
    )
    job = TAPService(url).submit_job(query, maxrec=1_000_000)
    job.run()
    job.wait()
    job.raise_if_error()
    table = job.fetch_result().to_table()
    job.delete()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        f.write(
            "# Gaia DR3 stars in the COSMOS field (neo/eval/stars.py fetch): parallax > "
            f"{min_parallax} mas or proper motion > {min_pm} mas/yr; ra, dec in deg at ref_epoch, "
            "pmra (= pmra*), pmdec in mas/yr, parallax in mas, phot_g_mean_mag in mag\n"
        )
        f.write(f"# box: {ra_lo},{ra_hi},{dec_lo},{dec_hi}\n")
        f.write(f"# query: {query}\n")
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        digits = {
            "ra": 7,
            "dec": 7,
            "pmra": 3,
            "pmdec": 3,
            "parallax": 4,
            "phot_g_mean_mag": 3,
            "ref_epoch": 1,
        }
        for row in table:
            cells = []
            for name in COLUMNS:
                value = row[name]
                if value is np.ma.masked or (isinstance(value, float) and not np.isfinite(value)):
                    cells.append("")
                else:
                    cells.append(f"{float(value):.{digits[name]}f}")
            writer.writerow(cells)
    return len(table)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="query Gaia DR3 once and write the catalog CSV")
    f.add_argument("--out", default=str(CATALOG), help=f"output CSV (default {CATALOG})")
    f.add_argument(
        "--box",
        type=float,
        nargs=4,
        default=list(BOX),
        metavar=("RA_MIN", "RA_MAX", "DEC_MIN", "DEC_MAX"),
        help="sky box in degrees (default covers COSMOS-Web and the local CANDELS strip)",
    )
    f.add_argument("--min-parallax", type=float, default=MIN_PARALLAX, help="mas")
    f.add_argument("--min-pm", type=float, default=MIN_PM, help="mas/yr")
    f.add_argument("--url", default=TAP_URL)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "fetch":
        n = fetch(args.out, args.box, args.min_parallax, args.min_pm, args.url)
        size = Path(args.out).stat().st_size / 1024
        print(f"{n} Gaia DR3 stars in box {tuple(args.box)} -> {args.out} ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
