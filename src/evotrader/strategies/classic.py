"""Hand-written baseline strategies.

These are the benchmarks the learned strategies must beat. A self-improving bot
that can't outperform a 50-line moving-average rule isn't improving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evotrader import indicators as ind
from evotrader.strategies.base import Strategy


class BuyAndHold(Strategy):
    name = "buy_and_hold"

    def _signal(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index)


class SmaCrossover(Strategy):
    """Trend following: invested while the fast average is above the slow one."""

    name = "sma_cross"

    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        super().__init__(fast=fast, slow=slow)

    def _signal(self, bars: pd.DataFrame) -> pd.Series:
        fast = ind.sma(bars["close"], self.params["fast"])
        slow = ind.sma(bars["close"], self.params["slow"])
        return (fast > slow).astype(float).where(slow.notna())


class RsiMeanReversion(Strategy):
    """Buy when oversold (RSI < entry), sell once it recovers (RSI > exit)."""

    name = "rsi_reversion"

    def __init__(self, window: int = 14, entry: float = 30, exit: float = 55) -> None:
        if entry >= exit:
            raise ValueError("entry threshold must be below exit threshold")
        super().__init__(window=window, entry=entry, exit=exit)

    def _signal(self, bars: pd.DataFrame) -> pd.Series:
        r = ind.rsi(bars["close"], self.params["window"])
        # 1 on entry days, 0 on exit days, NaN otherwise -> forward-fill holds the position.
        events = pd.Series(np.nan, index=bars.index)
        events[r < self.params["entry"]] = 1.0
        events[r > self.params["exit"]] = 0.0
        return events.ffill().fillna(0.0).where(r.notna())


class Momentum(Strategy):
    """Time-series momentum: invested when the trailing return is positive."""

    name = "momentum"

    def __init__(self, lookback: int = 126, threshold: float = 0.0) -> None:
        super().__init__(lookback=lookback, threshold=threshold)

    def _signal(self, bars: pd.DataFrame) -> pd.Series:
        mom = ind.momentum(bars["close"], self.params["lookback"])
        return (mom > self.params["threshold"]).astype(float).where(mom.notna())
