import pytest
from astropy.table import Table

from neo.surveys.rubin import sia


@pytest.fixture
def patches():
    return Table(
        {
            "lsst_band": ["i", "i", "r", "z"],
            "lsst_tract": [5063, 5064, 5063, 5063],
            "lsst_patch": [15, 3, 15, 15],
            "access_estsize": [200_000, 200_000, 200_000, 200_000],
            "access_url": ["u1", "u2", "u3", "u4"],
        }
    )


def test_load_token_from_file_strips_whitespace(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("  abc123\n")
    assert sia.load_token(str(token_file)) == "abc123"


def test_load_token_from_env(monkeypatch):
    monkeypatch.setenv("RSP_TOKEN", "envtoken")
    assert sia.load_token() == "envtoken"


def test_load_token_missing_exits(monkeypatch):
    monkeypatch.delenv("RSP_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        sia.load_token()


def test_make_session_sets_bearer_header():
    session = sia.make_session("tok")
    assert session.headers["Authorization"] == "Bearer tok"


def test_filter_bands_keeps_requested(patches):
    out = sia.filter_bands(patches, ["i", "z"])
    assert list(out["lsst_band"]) == ["i", "i", "z"]


def test_filter_bands_none_returns_all(patches):
    assert len(sia.filter_bands(patches, None)) == 4


def test_estimated_size_gb(patches):
    assert sia.estimated_size_gb(patches) == pytest.approx(0.8)


def test_estimated_size_gb_without_column():
    assert sia.estimated_size_gb(Table({"lsst_band": ["i"]})) is None


def test_summarize_counts_per_band_and_tracts(patches):
    text = sia.summarize(patches)
    assert "4 deep_coadd patches" in text
    assert "i: 2 patches in tracts [5063, 5064]" in text
    assert "r: 1 patches in tracts [5063]" in text
    assert "estimated download: 0.8 GB" in text


def test_summarize_empty():
    assert sia.summarize(Table()) == "0 deep_coadd patches"


def test_parse_args_defaults_to_cosmos_i_band():
    args = sia.parse_args([])
    assert (args.ra, args.dec, args.radius) == (sia.COSMOS_RA, sia.COSMOS_DEC, sia.COSMOS_RADIUS)
    assert args.band == ["i"]
    assert args.collection == "dp2"
