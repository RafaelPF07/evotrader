"""Market-wide signals beyond each ETF's own price: fear, rates, credit, the dollar,
commodities and market internals.

Each source is a daily Yahoo Finance series, cached like the ETF data. They are
attached to every ETF's bars as extra `m_*` columns, forward-filled onto that ETF's
trading days (never back-filled), so each value is the last one published at or
before that day's close. Decisions are made at the close and filled at the next
open, as everywhere else, so none of these signals can see the future.

Only accounts and experiments that call `attach_macro` see these columns; the live
evolved bot does not, so its behaviour is unchanged.
"""

from __future__ import annotations

import pandas as pd

from evotrader import data as _data
from evotrader.data import drop_incomplete, load
from evotrader.evolution.fitness import Dataset

# name -> Yahoo ticker. Closes only; these are indicators, not things we trade.
SOURCES = {
    "vix": "^VIX", "vix3m": "^VIX3M", "vvix": "^VVIX", "skew": "^SKEW",
    "tnx": "^TNX", "irx": "^IRX",
    "hyg": "HYG", "ief": "IEF", "tip": "TIP",
    "dxy": "DX-Y.NYB", "oil": "CL=F", "copper": "HG=F", "gold": "GC=F",
    "rsp": "RSP", "spy": "SPY", "iwm": "IWM", "xly": "XLY", "xlp": "XLP", "eem": "EEM",
}


def _load_close(ticker: str, refresh: bool = False) -> pd.Series:
    path = _data.CACHE_DIR / f"close_{ticker.replace('^', 'IDX_').replace('=', '_')}.csv"
    if refresh or not path.exists():
        _data.ensure_writable(path)
        import yfinance as yf

        _data.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        raw = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False,
                          multi_level_index=False)
        close = raw["Close"].rename("close")
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close.index.name = "date"
        drop_incomplete(close.to_frame()).to_csv(path)
    return pd.read_csv(path, index_col="date", parse_dates=True)["close"].dropna()


def load_macro(refresh: bool = False) -> pd.DataFrame:
    """Raw macro series, one column per source, on the union of their dates."""
    cols = {}
    for name, ticker in SOURCES.items():
        if name in ("spy", "iwm", "xly", "xlp", "eem"):
            cols[name] = load(ticker, start="1990-01-01", refresh=refresh)["close"]
        else:
            cols[name] = _load_close(ticker, refresh)
    return pd.DataFrame(cols).sort_index()


def derive(raw: pd.DataFrame) -> pd.DataFrame:
    """The indicator series the genetic algorithm can use, prefixed `m_`."""
    safe = lambda s: s.clip(lower=1e-6)  # noqa: E731 - oil futures went negative in 2020
    return pd.DataFrame({
        "m_vix": raw["vix"],
        "m_vix_term": raw["vix"] / raw["vix3m"],  # > 1: short-term fear above long-term
        "m_vvix": raw["vvix"],
        "m_skew": raw["skew"],
        "m_curve": raw["tnx"] - raw["irx"],  # 10-year minus 3-month yield, % points
        "m_tnx": raw["tnx"],
        "m_credit": raw["hyg"] / raw["ief"],  # junk bonds vs Treasuries
        "m_inflation": raw["tip"] / raw["ief"],  # inflation-protected vs nominal
        "m_dollar": raw["dxy"],
        "m_oil": safe(raw["oil"]),
        "m_copper_gold": raw["copper"] / raw["gold"],
        "m_breadth": raw["rsp"] / raw["spy"],  # equal-weight vs cap-weight S&P 500
        "m_small_large": raw["iwm"] / raw["spy"],
        "m_cyclical": raw["xly"] / raw["xlp"],  # discretionary vs staples
        "m_em": raw["eem"] / raw["spy"],
        "m_spy": raw["spy"],
    })


def attach_macro(datasets: list[Dataset], macro: pd.DataFrame | None = None) -> list[Dataset]:
    """New datasets whose bars carry the macro columns, aligned without look-ahead."""
    derived = derive(load_macro()) if macro is None else macro
    out = []
    for ds in datasets:
        aligned = derived.reindex(derived.index.union(ds.bars.index)).ffill()
        out.append(Dataset(ds.ticker, ds.bars.join(aligned.reindex(ds.bars.index))))
    return out
