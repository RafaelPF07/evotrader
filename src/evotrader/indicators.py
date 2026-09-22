"""Technical indicators.

Every function here uses only data up to and including each row, so the value
at date t is knowable at the close of day t. The no-look-ahead test suite checks this.
These are also the building blocks the evolutionary engine will combine in Phase 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def momentum(close: pd.Series, window: int) -> pd.Series:
    """Percentage change over the last `window` bars."""
    return close.pct_change(window)


def zscore(series: pd.Series, window: int) -> pd.Series:
    """How many rolling standard deviations the series is from its rolling mean."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0, np.nan)


def volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualised rolling volatility of daily returns."""
    return close.pct_change().rolling(window, min_periods=window).std() * np.sqrt(252)
