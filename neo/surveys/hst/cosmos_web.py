"""Download tiles of the COSMOS-Web DR1 HST/ACS F814W mosaic (Gaia DR3-aligned, gzipped FITS).

20 tiles named A1-A10 and B1-B10, at 30mas (0.03") or 60mas (0.06"), each with sci/wht/err.
Headers carry no PHOTFLAM/PHOTPLAM (units electron/s); pair with the F814W AB zeropoint 25.94.
Source: https://cosmos2025.iap.fr/hst.html
"""

import argparse
import gzip
import shutil
from pathlib import Path

import requests

BASE_URL = "https://cosmos2025.iap.fr/data/hst"
TILE_TEMPLATE = "mosaic_acs_f814w_COSMOS-Web_{scale}_{tile}_v1.0_{kind}.fits.gz"
ALL_TILES = [f"{row}{i}" for row in "AB" for i in range(1, 11)]
CHUNK_BYTES = 1 << 20


def tile_filename(tile: str, scale: str = "30mas", kind: str = "sci") -> str:
    """Local (unzipped) filename for a tile."""
    return TILE_TEMPLATE.format(scale=scale, tile=tile, kind=kind).removesuffix(".gz")


def download_tile(
    tile: str, dest: Path, scale: str = "30mas", kind: str = "sci"
) -> tuple[str, str]:
    """Fetch one tile into `dest` as an uncompressed FITS; skips tiles already present.

    astropy cannot memmap a gzipped FITS, so tiles are gunzipped on arrival.
    """
    final = dest / tile_filename(tile, scale, kind)
    if final.exists():
        return final.name, "skipped"
    gz = final.with_name(final.name + ".gz")
    part = gz.with_name(gz.name + ".part")
    url = f"{BASE_URL}/{TILE_TEMPLATE.format(scale=scale, tile=tile, kind=kind)}"
    with requests.get(url, stream=True, timeout=180) as response:
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
    parser.add_argument("--dest", default="data/hst/cosmos_web", help="output directory")
    parser.add_argument("--tiles", nargs="*", help="tile names A1..B10 (default: all 20)")
    parser.add_argument("--scale", default="30mas", choices=["30mas", "60mas"])
    parser.add_argument("--kind", default="sci", choices=["sci", "wht", "err"])
    parser.add_argument("--dry-run", action="store_true", help="list tiles, fetch nothing")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    tiles = args.tiles or ALL_TILES
    unknown = [t for t in tiles if t not in ALL_TILES]
    if unknown:
        raise SystemExit(f"unknown tiles {unknown}; valid: {ALL_TILES}")
    print(f"{len(tiles)} tiles ({args.scale}, {args.kind}): {tiles}")
    if args.dry_run:
        return
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for i, tile in enumerate(tiles, 1):
        name, status = download_tile(tile, dest, args.scale, args.kind)
        print(f"[{i}/{len(tiles)}] {name}: {status}")


if __name__ == "__main__":
    main()
