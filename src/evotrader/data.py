"""Market data loading with a local CSV cache.

All prices are split/dividend adjusted so that returns computed from them are
total returns. Columns are always: open, high, low, close, volume, indexed by date.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
SNAPSHOT_DIR = CACHE_DIR.parent / "snapshots"
_frozen = False  # True while reading from a snapshot: never download, never overwrite
COLUMNS = ["open", "high", "low", "close", "volume"]
MARKET_CLOSE_MINUTES = 16 * 60 + 30  # 16:30 ET: close plus a buffer for final prices

# Liquid ETFs that have existed for the whole test window. Using ETFs rather than
# today's hand-picked winning stocks avoids most survivorship bias.
DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "TLT", "GLD"]


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.csv"


def download(ticker: str, start: str = "2005-01-01", end: str | None = None) -> pd.DataFrame:
    """Download daily adjusted OHLCV bars from Yahoo Finance."""
    import yfinance as yf  # imported lazily so tests never need the network

    raw = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False
    )
    if raw.empty:
        raise ValueError(f"No data returned for {ticker!r}")
    df = raw.rename(columns=str.lower)[COLUMNS]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return drop_incomplete(df.dropna())


def drop_incomplete(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove today's bar while the US market is still open.

    During trading hours Yahoo returns a partial bar whose 'close' is just the
    latest price. Treating that as a real close would be a subtle live-only form
    of look-ahead, so only completed sessions are kept (close 16:00 ET + buffer).
    """
    now = now if now is not None else pd.Timestamp.now(tz="America/New_York")
    today = now.tz_localize(None).normalize() if now.tzinfo else now.normalize()
    session_done = now.hour * 60 + now.minute >= MARKET_CLOSE_MINUTES
    return df if session_done else df[df.index < today]


def load(
    ticker: str, start: str = "2005-01-01", end: str | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Load bars for `ticker`, using the cache when possible."""
    path = _cache_path(ticker)
    if refresh or not path.exists():
        ensure_writable(path)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        download(ticker, start="1990-01-01").to_csv(path)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return validate(df.loc[start:end])


RATES_TICKER = "^IRX"  # 13-week US Treasury bill yield, in percent


def load_rates(refresh: bool = False) -> pd.Series:
    """Daily 3-month T-bill yield as a decimal (0.05 = 5%), used to charge for borrowing.

    Not a tradable price (it can be zero or slightly negative), so it skips `validate`.
    """
    path = _cache_path(RATES_TICKER)
    if refresh or not path.exists():
        ensure_writable(path)
        import yfinance as yf

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        raw = yf.download(RATES_TICKER, start="1990-01-01", auto_adjust=False, progress=False,
                          multi_level_index=False)
        rates = raw["Close"].rename("rate") / 100
        rates.index = pd.to_datetime(rates.index).tz_localize(None)
        rates.index.name = "date"
        drop_incomplete(rates.to_frame()).to_csv(path)
    return pd.read_csv(path, index_col="date", parse_dates=True)["rate"].dropna()


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check a bar DataFrame is well formed; raise early rather than backtest garbage."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Index must be sorted, unique dates")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    return df[COLUMNS]


# --- frozen data snapshots -----------------------------------------------------------
# Yahoo revises historical prices slightly on every download, and the genetic algorithm
# is chaotic: a one-in-a-million change can evolve a different champion. So any result
# meant to be reproducible runs on a frozen, fingerprinted copy of the data.

def ensure_writable(path: Path) -> None:
    if _frozen:
        raise RuntimeError(f"{path.name} is not in the frozen snapshot, and snapshots are "
                           "never downloaded into or refreshed")


def create_snapshot(name: str) -> dict:
    """Copy the current cache into data/snapshots/<name> and fingerprint every file."""
    import shutil

    target = SNAPSHOT_DIR / name
    if target.exists():
        raise RuntimeError(f"snapshot {name!r} already exists; snapshots are immutable")
    target.mkdir(parents=True)
    for f in sorted(CACHE_DIR.glob("*.csv")):
        if not f.name.startswith("ml_"):  # derived ML outputs are recomputed, not frozen
            shutil.copy2(f, target / f.name)
    manifest = {"name": name, "created": pd.Timestamp.now().isoformat(timespec="seconds"),
                "fingerprint": fingerprint(target)}
    (target / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2),
                                          encoding="utf-8")
    return manifest


def fingerprint(directory: Path) -> str:
    """SHA-256 over every data file's name and exact bytes."""
    import hashlib

    h = hashlib.sha256()
    for f in sorted(directory.glob("*.csv")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def use_snapshot(name: str) -> str:
    """Read all data from a frozen snapshot from now on; returns its fingerprint."""
    global CACHE_DIR, _frozen
    target = SNAPSHOT_DIR / name
    if not target.exists():
        raise FileNotFoundError(f"no snapshot {name!r}; create it with `evotrader snapshot`")
    CACHE_DIR, _frozen = target, True
    return fingerprint(target)
