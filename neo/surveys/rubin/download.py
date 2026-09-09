"""Download the Rubin deep_coadd patches listed by `neo.surveys.rubin.sia --out`.

Resolves each row's DataLink to the file URL and streams the FITS into --dest.
Files already present are skipped, so an interrupted run can simply be re-run.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from astropy.table import Table
from pyvo.dal.adhoc import DatalinkResults

from neo.surveys.rubin.sia import load_token, make_session

CHUNK_BYTES = 1 << 20


def patch_filename(row) -> str:
    return f"deep_coadd_{int(row['lsst_tract'])}_{int(row['lsst_patch'])}_{row['lsst_band']}.fits"


def select_rows(table: Table, tracts: list[int] | None = None, limit: int | None = None) -> Table:
    if tracts:
        table = table[[int(t) in tracts for t in table["lsst_tract"]]]
    if limit is not None:
        table = table[:limit]
    return table


def resolve_file(datalink_url: str, session: requests.Session) -> tuple[str, int | None]:
    """Return the '#this' record's URL and size. The URL is pre-signed and expires in hours."""
    links = DatalinkResults.from_result_url(datalink_url, session=session)
    for rec in links:
        if rec.semantics == "#this":
            size = rec.content_length
            return rec.access_url, int(size) if size else None
    raise ValueError(f"no '#this' record in {datalink_url}")


def download(url: str, path: Path, size: int | None = None) -> None:
    """Fetch without the RSP token: the URL is pre-signed for a third-party bucket."""
    part = path.with_name(path.name + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(part, "wb") as f:
            for chunk in response.iter_content(CHUNK_BYTES):
                f.write(chunk)
    if size is not None and part.stat().st_size != size:
        part.unlink()
        raise OSError(f"size mismatch for {path.name}: expected {size} bytes")
    part.replace(path)


def fetch_row(row, dest: Path, session: requests.Session) -> tuple[str, str]:
    name = patch_filename(row)
    path = dest / name
    if path.exists():
        return name, "skipped"
    url, size = resolve_file(str(row["access_url"]), session)
    download(url, path, size)
    return name, f"{path.stat().st_size / 1e6:.0f} MB"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", required=True, help="CSV written by neo.surveys.rubin.sia")
    parser.add_argument("--dest", default="data/rubin/dp2", help="output directory")
    parser.add_argument("--token-file", help="file containing an RSP token (else RSP_TOKEN)")
    parser.add_argument("--tract", nargs="+", type=int, help="only these tract numbers")
    parser.add_argument("--limit", type=int, help="stop after this many rows")
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads")
    parser.add_argument("--dry-run", action="store_true", help="list files, fetch nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = select_rows(Table.read(args.table, format="csv"), args.tract, args.limit)
    dest = Path(args.dest)
    print(f"{len(rows)} patches -> {dest}")

    if args.dry_run:
        for row in rows:
            print(" ", patch_filename(row))
        return

    dest.mkdir(parents=True, exist_ok=True)
    session = make_session(load_token(args.token_file))
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_row, row, dest, session): row for row in rows}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                name, status = future.result()
                print(f"[{i}/{len(rows)}] {name}: {status}")
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                name = patch_filename(futures[future])
                failures.append((name, exc))
                print(f"[{i}/{len(rows)}] {name}: FAILED {exc}")
    if failures:
        raise SystemExit(f"{len(failures)} downloads failed; re-run to retry them")


if __name__ == "__main__":
    main()
