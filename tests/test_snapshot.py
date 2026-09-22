import pytest

from evotrader import data
from tests.conftest import make_bars


@pytest.fixture
def cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(data, "CACHE_DIR", cache)
    monkeypatch.setattr(data, "SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(data, "_frozen", False)
    make_bars(n=50).to_csv(cache / "AAA.csv")
    (cache / "ml_AAA_x.csv").write_text("derived")
    return cache


def test_snapshot_is_fingerprinted_and_immutable(cache):
    m = data.create_snapshot("s1")
    snap = data.SNAPSHOT_DIR / "s1"
    assert (snap / "AAA.csv").exists() and not (snap / "ml_AAA_x.csv").exists()
    assert m["fingerprint"] == data.fingerprint(snap)
    with pytest.raises(RuntimeError, match="immutable"):
        data.create_snapshot("s1")


def test_fingerprint_changes_with_a_single_byte(cache):
    data.create_snapshot("s1")
    snap = data.SNAPSHOT_DIR / "s1"
    before = data.fingerprint(snap)
    text = (snap / "AAA.csv").read_text()
    (snap / "AAA.csv").write_text(text.replace("1", "2", 1))
    assert data.fingerprint(snap) != before


def test_reading_a_snapshot_never_downloads(cache):
    data.create_snapshot("s1")
    fp = data.use_snapshot("s1")
    assert data.CACHE_DIR == data.SNAPSHOT_DIR / "s1" and fp
    assert len(data.load("AAA", start="1990-01-01")) == 50
    with pytest.raises(RuntimeError, match="frozen snapshot"):
        data.load("BBB", start="1990-01-01")  # not in the snapshot: refuse, don't fetch
    with pytest.raises(RuntimeError, match="frozen snapshot"):
        data.load("AAA", start="1990-01-01", refresh=True)


def test_missing_snapshot_is_an_error(cache):
    with pytest.raises(FileNotFoundError):
        data.use_snapshot("nope")
