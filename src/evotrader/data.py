"""Market data loading with a local CSV cache.

All prices are split/dividend adjusted so that returns computed from them are
total returns. Columns are always: open, high, low, close, volume, indexed by date.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
COLUMNS = ["open", "high", "low", "close", "volume"]

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
    return df.dropna()


def load(
    ticker: str, start: str = "2005-01-01", end: str | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Load bars for `ticker`, using the cache when possible."""
    path = _cache_path(ticker)
    if refresh or not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        download(ticker, start="1990-01-01").to_csv(path)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return validate(df.loc[start:end])


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
