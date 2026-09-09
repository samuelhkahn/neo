"""Query the Rubin Science Platform SIA service for deep coadds over a sky position.

Lists the matching patches (band, tract, patch, access URL) without downloading anything.
Needs an RSP token with the read:image scope, supplied via RSP_TOKEN or --token-file.
"""

import argparse
import os
from pathlib import Path

import requests
from astropy.table import Table
from pyvo.dal import SIA2Service

SIA_URL_TEMPLATE = "https://data.lsst.cloud/api/sia/{collection}"
DEEP_COADD_SUBTYPE = "lsst.deep_coadd"

# DP2 deep drilling field center and the radius enclosing all visit boresights, in degrees.
COSMOS_RA, COSMOS_DEC, COSMOS_RADIUS = 150.1, 2.1, 1.0

LSST_BANDS = ("u", "g", "r", "i", "z", "y")
DISPLAY_COLUMNS = ("lsst_band", "lsst_tract", "lsst_patch", "s_ra", "s_dec", "access_url")


def load_token(token_file: str | None = None) -> str:
    if token_file:
        return Path(token_file).expanduser().read_text().strip()
    token = os.environ.get("RSP_TOKEN", "").strip()
    if not token:
        raise SystemExit("No RSP token found: set RSP_TOKEN or pass --token-file")
    return token


def make_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return session


def sia_service(token: str, collection: str = "dp2") -> SIA2Service:
    url = SIA_URL_TEMPLATE.format(collection=collection)
    return SIA2Service(url, session=make_session(token))


def filter_bands(table: Table, bands: list[str] | None) -> Table:
    if not bands or len(table) == 0:
        return table
    keep = [str(b) in bands for b in table["lsst_band"]]
    return table[keep]


def query_deep_coadds(
    service: SIA2Service,
    ra: float,
    dec: float,
    radius: float,
    bands: list[str] | None = None,
) -> Table:
    results = service.search(pos=(ra, dec, radius), calib_level=3, dpsubtype=DEEP_COADD_SUBTYPE)
    table = results.to_table()
    if "dataproduct_subtype" in table.colnames:
        table = table[[str(s) == DEEP_COADD_SUBTYPE for s in table["dataproduct_subtype"]]]
    table = filter_bands(table, bands)
    table.sort([c for c in ("lsst_band", "lsst_tract", "lsst_patch") if c in table.colnames])
    return table


def estimated_size_gb(table: Table) -> float | None:
    """ObsCore access_estsize is in kilobytes; returns None if the column is absent."""
    if "access_estsize" not in table.colnames or len(table) == 0:
        return None
    sizes = table["access_estsize"]
    if hasattr(sizes, "filled"):
        sizes = sizes.filled(0)
    return float(sum(sizes)) / 1e6


def summarize(table: Table) -> str:
    if len(table) == 0:
        return "0 deep_coadd patches"
    lines = [f"{len(table)} deep_coadd patches"]
    for band in sorted({str(b) for b in table["lsst_band"]}):
        sub = table[[str(b) == band for b in table["lsst_band"]]]
        tracts = sorted({int(t) for t in sub["lsst_tract"]})
        lines.append(f"  {band}: {len(sub)} patches in tracts {tracts}")
    size = estimated_size_gb(table)
    if size is not None:
        lines.append(f"  estimated download: {size:.1f} GB")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ra", type=float, default=COSMOS_RA, help="center RA, deg")
    parser.add_argument("--dec", type=float, default=COSMOS_DEC, help="center Dec, deg")
    parser.add_argument("--radius", type=float, default=COSMOS_RADIUS, help="search radius, deg")
    parser.add_argument(
        "--band",
        nargs="+",
        default=["i"],
        help="bands to keep, e.g. --band i r z; use 'all' for every band",
    )
    parser.add_argument("--collection", default="dp2", help="SIA collection (dp1, dp2)")
    parser.add_argument("--token-file", help="file containing an RSP token (else RSP_TOKEN)")
    parser.add_argument("--out", help="write the full result table to this CSV path")
    parser.add_argument("--max-rows", type=int, default=30, help="rows to print (-1 for all)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    bands = None if "all" in args.band else args.band

    service = sia_service(load_token(args.token_file), args.collection)
    table = query_deep_coadds(service, args.ra, args.dec, args.radius, bands)

    print(summarize(table))
    if len(table):
        shown = [c for c in DISPLAY_COLUMNS if c in table.colnames]
        limit = None if args.max_rows < 0 else args.max_rows
        table[shown][:limit].pprint(max_lines=-1, max_width=-1)
    if args.out:
        table.write(args.out, format="csv", overwrite=True)
        print(f"wrote {len(table)} rows to {args.out}")


if __name__ == "__main__":
    main()
