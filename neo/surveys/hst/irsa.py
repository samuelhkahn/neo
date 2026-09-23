"""Download tiles of the IRSA COSMOS ACS F814W v2.0 mosaic (0.03"/px, gzipped FITS).

The 81 tiles are numbered on a 12-column grid (13-21, 25-33, ..., 109-117); adjacent
numbers are adjacent in RA and numbers 12 apart are adjacent in Dec.
"""

import argparse
import gzip
import shutil
from pathlib import Path

import requests

TILE_URL = (
    "https://irsa.ipac.caltech.edu/data/COSMOS/images/acs_mosaic_2.0/tiles/"
    "acs_I_030mas_{tile:03d}_{kind}.fits.gz"
)
GRID_COLUMNS = 12
ALL_TILES = [n for row in range(13, 110, GRID_COLUMNS) for n in range(row, row + 9)]
COSMOS_CENTER_TILE = 65
CHUNK_BYTES = 1 << 20


def block(center: int = COSMOS_CENTER_TILE, size: int = 3) -> list[int]:
    """Tile numbers of a size x size block centred on `center` (size 4 extends right/down)."""
    row, col = divmod(center - 1, GRID_COLUMNS)
    lo = -(size // 2) + (1 if size % 2 == 0 else 0)
    hi = size // 2
    tiles = []
    for dr in range(lo, hi + 1):
        for dc in range(lo, hi + 1):
            n = (row + dr) * GRID_COLUMNS + (col + dc) + 1
            if n in ALL_TILES:
                tiles.append(n)
    return tiles


def tile_filename(tile: int, kind: str = "sci") -> str:
    return f"acs_I_030mas_{tile:03d}_{kind}.fits"


def download_tile(tile: int, dest: Path, kind: str = "sci") -> tuple[str, str]:
    """Fetch one tile into `dest` as an uncompressed FITS; skips tiles already present."""
    final = dest / tile_filename(tile, kind)
    if final.exists():
        return final.name, "skipped"
    gz = final.with_suffix(".fits.gz")
    part = gz.with_name(gz.name + ".part")
    url = TILE_URL.format(tile=tile, kind=kind)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        expected = int(response.headers.get("Content-Length", 0)) or None
        with open(part, "wb") as f:
            for chunk in response.iter_content(CHUNK_BYTES):
                f.write(chunk)
    if expected is not None and part.stat().st_size != expected:
        part.unlink()
        raise OSError(f"size mismatch for {gz.name}: expected {expected} bytes")
    part.replace(gz)
    with gzip.open(gz, "rb") as src, open(final, "wb") as out:
        shutil.copyfileobj(src, out, CHUNK_BYTES)
    gz.unlink()
    return final.name, f"{final.stat().st_size / 1e9:.2f} GB"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default="data/hst/cosmos_acs", help="output directory")
    parser.add_argument("--tiles", nargs="*", type=int, help="explicit tile numbers")
    parser.add_argument("--block", type=int, help="size of a block centred on --center")
    parser.add_argument("--center", type=int, default=COSMOS_CENTER_TILE)
    parser.add_argument("--kind", default="sci", choices=["sci", "wht"])
    parser.add_argument("--dry-run", action="store_true", help="list tiles, fetch nothing")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    tiles = args.tiles or (block(args.center, args.block) if args.block else [])
    if not tiles:
        raise SystemExit("give --tiles or --block")
    print(f"{len(tiles)} tiles: {tiles}")
    if args.dry_run:
        return
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for i, tile in enumerate(tiles, 1):
        name, status = download_tile(tile, dest, args.kind)
        print(f"[{i}/{len(tiles)}] {name}: {status}")


if __name__ == "__main__":
    main()
