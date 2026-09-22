"""Leveraged trend following ("Leverage for the Long Run", Gayed & Bilello, 2016).

Each ETF gets an equal slot. At every close:

    price above its 200-day moving average  ->  hold the slot at `leverage`
    price below it                          ->  hold T-bills (cash earning the T-bill rate)

The idea: leverage is dangerous in volatile, falling markets and profitable in
calm, rising ones, and a price above its 200-day average is a simple sign of the
latter. Crucially for this project, a trend filter also exits *slow, grinding*
declines, which is exactly where experiment 2's volatility targeting stayed
levered and failed.

Optionally the whole book is scaled by volatility targeting as well (arm L3).
Execution follows the backtester: decide at the close, fill at the next open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evotrader.evolution.rotation import Panel
from evotrader.metrics import TRADING_DAYS
from evotrader.sizing import VolTarget, exposure_path


@dataclass(frozen=True)
class TrendLeverage:
    sma: int = 200
    leverage: float = 1.5
    vol_target: VolTarget | None = None  # scale the whole book by vol targeting too
    borrow_spread: float = 0.01  # annual spread over T-bills on borrowed money
    cash_earns_rate: bool = True  # idle money sits in T-bills, as in the paper

    def __str__(self) -> str:
        vt = f", {self.vol_target}" if self.vol_target else ""
        return f"trend(sma={self.sma}, leverage={self.leverage}{vt})"


def trend_mask(panel: Panel, sma: int) -> np.ndarray:
    """1 where an ETF closed above its moving average, else 0 (and 0 during warm-up)."""
    closes = pd.DataFrame(panel.close, index=panel.index)
    average = closes.rolling(sma, min_periods=sma).mean()
    return (closes > average).to_numpy(dtype=float)


def weights(panel: Panel, tl: TrendLeverage) -> np.ndarray:
    n = len(panel.tickers)
    w = trend_mask(panel, tl.sma) * tl.leverage / n
    if tl.vol_target is not None:
        w = w * exposure_path(panel, tl.vol_target)[:, None] / tl.vol_target.cap
    return w


def financing(held: pd.Series, rates: pd.Series, spread: float, cash_earns: bool) -> pd.Series:
    """Daily financing return: interest on idle cash, minus the cost of borrowed money."""
    rate = rates.reindex(held.index).ffill().fillna(0.0).clip(lower=0.0)
    borrowed = (held - 1.0).clip(lower=0.0)
    idle = (1.0 - held).clip(lower=0.0) if cash_earns else 0.0
    return (idle * rate - borrowed * (rate + spread)) / TRADING_DAYS


def simulate(panel: Panel, tl: TrendLeverage, cost_bps: float,
             rates: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Daily returns after trading costs and financing, and the exposure held."""
    returns, held = panel.simulate(weights(panel, tl), cost_bps)
    return returns + financing(held, rates, tl.borrow_spread, tl.cash_earns_rate), held
