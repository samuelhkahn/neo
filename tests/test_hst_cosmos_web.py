import gzip
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from neo.surveys.hst import cosmos_web


def test_tile_grid_is_20_named_tiles():
    assert len(cosmos_web.ALL_TILES) == 20
    assert cosmos_web.ALL_TILES[:10] == [f"A{i}" for i in range(1, 11)]
    assert cosmos_web.ALL_TILES[-1] == "B10"


def test_tile_filename_drops_gz_and_defaults_to_30mas_sci():
    assert cosmos_web.tile_filename("A1") == "mosaic_acs_f814w_COSMOS-Web_30mas_A1_v1.0_sci.fits"
    assert cosmos_web.tile_filename("B10", "60mas", "wht") == (
        "mosaic_acs_f814w_COSMOS-Web_60mas_B10_v1.0_wht.fits"
    )


def test_download_tile_unzips_and_verifies_size(tmp_path, monkeypatch):
    payload = gzip.compress(b"SIMPLE  =" + b" " * 100)

    @contextmanager
    def fake_get(url, **kwargs):
        assert url.endswith("mosaic_acs_f814w_COSMOS-Web_30mas_A1_v1.0_sci.fits.gz")
        yield SimpleNamespace(
            raise_for_status=lambda: None,
            headers={"Content-Length": str(len(payload))},
            iter_content=lambda n: iter([payload]),
        )

    monkeypatch.setattr(cosmos_web.requests, "get", fake_get)
    name, status = cosmos_web.download_tile("A1", tmp_path)
    assert name == "mosaic_acs_f814w_COSMOS-Web_30mas_A1_v1.0_sci.fits"
    assert (tmp_path / name).read_bytes().startswith(b"SIMPLE  =")
    assert not list(tmp_path.glob("*.gz*"))
    assert cosmos_web.download_tile("A1", tmp_path) == (name, "skipped")


def test_download_tile_size_mismatch_removes_partial(tmp_path, monkeypatch):
    payload = gzip.compress(b"x" * 10)

    @contextmanager
    def fake_get(url, **kwargs):
        yield SimpleNamespace(
            raise_for_status=lambda: None,
            headers={"Content-Length": str(len(payload) + 5)},
            iter_content=lambda n: iter([payload]),
        )

    monkeypatch.setattr(cosmos_web.requests, "get", fake_get)
    with pytest.raises(OSError):
        cosmos_web.download_tile("A1", tmp_path)
    assert not list(tmp_path.iterdir())


def test_parse_args_defaults_and_unknown_tile():
    args = cosmos_web.parse_args([])
    assert args.tiles is None and args.scale == "30mas" and args.kind == "sci"
    with pytest.raises(SystemExit):
        cosmos_web.main(["--tiles", "Z9", "--dry-run"])
