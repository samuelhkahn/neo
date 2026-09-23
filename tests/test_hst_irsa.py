import gzip
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from neo.surveys.hst import irsa


def test_grid_has_81_tiles_in_nine_rows():
    assert len(irsa.ALL_TILES) == 81
    assert irsa.ALL_TILES[:9] == list(range(13, 22))
    assert irsa.ALL_TILES[-9:] == list(range(109, 118))


def test_block_around_center():
    assert irsa.block(65, 3) == [52, 53, 54, 64, 65, 66, 76, 77, 78]
    assert irsa.block(65, 1) == [65]
    assert len(irsa.block(65, 4)) == 16 and 65 in irsa.block(65, 4)
    assert len(irsa.block(65, 5)) == 25
    assert irsa.block(13, 3) == [13, 14, 25, 26]  # clipped at the grid edge


def test_tile_url_and_filename():
    assert irsa.TILE_URL.format(tile=65, kind="sci").endswith("acs_I_030mas_065_sci.fits.gz")
    assert irsa.tile_filename(7) == "acs_I_030mas_007_sci.fits"


def test_download_tile_unzips_and_verifies_size(tmp_path, monkeypatch):
    payload = gzip.compress(b"SIMPLE  =" + b" " * 100)

    @contextmanager
    def fake_get(url, **kwargs):
        yield SimpleNamespace(
            raise_for_status=lambda: None,
            headers={"Content-Length": str(len(payload))},
            iter_content=lambda n: iter([payload]),
        )

    monkeypatch.setattr(irsa.requests, "get", fake_get)
    name, status = irsa.download_tile(65, tmp_path)
    assert name == "acs_I_030mas_065_sci.fits" and status.endswith("GB")
    assert (tmp_path / name).read_bytes().startswith(b"SIMPLE  =")
    assert not list(tmp_path.glob("*.gz*"))
    assert irsa.download_tile(65, tmp_path) == (name, "skipped")


def test_download_tile_size_mismatch(tmp_path, monkeypatch):
    payload = gzip.compress(b"x" * 10)

    @contextmanager
    def fake_get(url, **kwargs):
        yield SimpleNamespace(
            raise_for_status=lambda: None,
            headers={"Content-Length": str(len(payload) + 5)},
            iter_content=lambda n: iter([payload]),
        )

    monkeypatch.setattr(irsa.requests, "get", fake_get)
    with pytest.raises(OSError):
        irsa.download_tile(65, tmp_path)
    assert not list(tmp_path.iterdir())
