"""Volatility targeting: size the whole portfolio by how turbulent the market is.

Each day the equal-weight ETF basket is held at

    exposure = target_scale * long-run volatility / recent volatility

capped at `cap` (1.5 = up to 50% borrowed) and floored at 0. When markets are
calm, recent volatility is below its long-run level and the bot holds *more*;
when they are turbulent, it holds less. Crashes tend to happen in turbulent
periods, so this cuts risk more than it cuts return, and the spare risk budget
can then be spent on modest leverage.

Everything is causal: exposure at day t uses returns up to the close of t, and
the backtester fills it at the next open. Borrowing is charged daily at the
3-month T-bill rate plus `borrow_spread` on the levered part. Cash earns nothing
(a deliberately conservative simplification).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evotrader.evolution.rotation import Panel
from evotrader.metrics import TRADING_DAYS


@dataclass(frozen=True)
class VolTarget:
    lookback: int = 20  # days of returns for "recent" volatility
    target_scale: float = 1.0  # 1.0 = aim for the basket's own long-run volatility
    band: float = 0.10  # only re-size when the target moves more than 10% (limits costs)
    cap: float = 1.5  # maximum exposure; > 1 means borrowing
    min_history: int = 252  # days needed before the long-run estimate is trusted
    borrow_spread: float = 0.01  # annual spread over T-bills paid on borrowed money

    def __str__(self) -> str:
        return (f"voltarget(lookback={self.lookback}, scale={self.target_scale}, "
                f"band={self.band}, cap={self.cap})")


def basket_returns(panel: Panel) -> pd.Series:
    """Close-to-close returns of the equal-weight basket."""
    closes = pd.DataFrame(panel.close, index=panel.index)
    return closes.pct_change().mean(axis=1)


def exposure_path(panel: Panel, vt: VolTarget) -> np.ndarray:
    """Target exposure decided at each close. Holds 1.0 until enough history exists."""
    r = basket_returns(panel)
    recent = r.rolling(vt.lookback, min_periods=vt.lookback).std()
    long_run = r.expanding(min_periods=vt.min_history).std()
    raw = (vt.target_scale * long_run / recent).clip(0.0, vt.cap).fillna(1.0).to_numpy()

    out = np.empty_like(raw)
    current = raw[0]
    for t, target in enumerate(raw):  # the no-trade band: ignore small changes
        if abs(target - current) > vt.band * max(current, 1e-9):
            current = target
        out[t] = current
    return out


def simulate(
    panel: Panel, vt: VolTarget, cost_bps: float, rates: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Daily returns (after trading and borrowing costs) and the exposure actually held."""
    n = len(panel.tickers)
    weights = exposure_path(panel, vt)[:, None] * np.full((1, n), 1.0 / n)
    returns, held = panel.simulate(weights, cost_bps)
    rate = rates.reindex(panel.index).ffill().fillna(0.0).clip(lower=0.0)
    borrowed = (held - 1.0).clip(lower=0.0)
    returns = returns - borrowed * (rate + vt.borrow_spread) / TRADING_DAYS
    return returns, held
