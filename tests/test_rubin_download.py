from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from astropy.table import Table

from neo.surveys.rubin import download


@pytest.fixture
def rows():
    return Table(
        {
            "lsst_band": ["i", "i", "i"],
            "lsst_tract": [9813, 9813, 9570],
            "lsst_patch": [15, 16, 70],
            "access_url": ["dl1", "dl2", "dl3"],
        }
    )


def test_patch_filename(rows):
    assert download.patch_filename(rows[0]) == "deep_coadd_9813_15_i.fits"


def test_select_rows_by_tract_and_limit(rows):
    assert len(download.select_rows(rows, tracts=[9813])) == 2
    assert len(download.select_rows(rows, limit=1)) == 1
    assert len(download.select_rows(rows)) == 3


def test_resolve_file_picks_this_record(monkeypatch):
    records = [
        SimpleNamespace(semantics="#cutout", access_url="", content_length=0),
        SimpleNamespace(
            semantics="#this", access_url="https://bucket/file.fits", content_length=42
        ),
    ]
    monkeypatch.setattr(
        download.DatalinkResults, "from_result_url", staticmethod(lambda url, session: records)
    )
    assert download.resolve_file("dl", session=None) == ("https://bucket/file.fits", 42)


def test_resolve_file_without_this_record_raises(monkeypatch):
    monkeypatch.setattr(
        download.DatalinkResults, "from_result_url", staticmethod(lambda *a, **k: [])
    )
    with pytest.raises(ValueError):
        download.resolve_file("dl", session=None)


def _fake_get(payload):
    calls = []

    @contextmanager
    def get(url, **kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(
            raise_for_status=lambda: None,
            iter_content=lambda n: iter([payload]),
        )

    return get, calls


def test_download_writes_file_without_token(tmp_path, monkeypatch):
    get, calls = _fake_get(b"x" * 10)
    monkeypatch.setattr(download.requests, "get", get)
    path = tmp_path / "a.fits"
    download.download("https://bucket/a.fits", path, size=10)
    assert path.read_bytes() == b"x" * 10
    assert not path.with_name("a.fits.part").exists()
    assert "headers" not in calls[0] and "auth" not in calls[0]


def test_download_size_mismatch_removes_partial(tmp_path, monkeypatch):
    get, _ = _fake_get(b"x" * 5)
    monkeypatch.setattr(download.requests, "get", get)
    path = tmp_path / "a.fits"
    with pytest.raises(OSError):
        download.download("https://bucket/a.fits", path, size=10)
    assert not path.exists() and not path.with_name("a.fits.part").exists()


def test_fetch_row_skips_existing_without_network(tmp_path, rows, monkeypatch):
    (tmp_path / "deep_coadd_9813_15_i.fits").write_bytes(b"done")

    def boom(*a, **k):
        raise AssertionError("network should not be touched")

    monkeypatch.setattr(download, "resolve_file", boom)
    assert download.fetch_row(rows[0], tmp_path, session=None) == (
        "deep_coadd_9813_15_i.fits",
        "skipped",
    )
