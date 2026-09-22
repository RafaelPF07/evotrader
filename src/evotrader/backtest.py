"""Daily bar backtester with realistic execution timing and costs.

Timeline for each day d:
    open[d]   orders decided at close[d-1] are filled here, paying costs
    close[d]  the strategy sees today's bar and sets a new target for tomorrow

So the position held during day d is target[d-1], and the overnight gap from
close[d-1] to open[d] is still earned by the *previous* position target[d-2].
A strategy therefore can never trade on a price it has not yet seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from evotrader.metrics import compute_metrics
from evotrader.strategies.base import Strategy

DEFAULT_COST_BPS = 5.0  # per side: commission + spread + slippage, in basis points


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)


def run_backtest(
    strategy: Strategy,
    bars: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    target = strategy.target_position(bars)
    returns, held = simulate(bars, target, cost_bps)
    equity = initial_capital * (1 + returns).cumprod()
    trades = extract_trades(bars, held, cost_bps)
    return BacktestResult(
        strategy=strategy.describe(),
        equity=equity.rename("equity"),
        returns=returns.rename("returns"),
        position=held.rename("position"),
        trades=trades,
        metrics=compute_metrics(returns, held, trades),
    )


def simulate(
    bars: pd.DataFrame, target: pd.Series, cost_bps: float = DEFAULT_COST_BPS
) -> tuple[pd.Series, pd.Series]:
    """Turn target positions into daily strategy returns. Returns (returns, held_position)."""
    held = target.shift(1).fillna(0.0)  # filled at next open
    held_prev = held.shift(1).fillna(0.0)  # position carried through the overnight gap

    gap = (bars["open"] / bars["close"].shift(1) - 1).fillna(0.0)
    intraday = bars["close"] / bars["open"] - 1
    cost = (held - held_prev).abs() * cost_bps / 10_000

    returns = (1 + held_prev * gap) * (1 - cost) * (1 + held * intraday) - 1
    return returns, held


def extract_trades(bars: pd.DataFrame, held: pd.Series, cost_bps: float) -> pd.DataFrame:
    """One row per round trip (flat -> invested -> flat), priced at the fill opens.

    This trade log is what the learning engine will study to find its mistakes.
    """
    invested = held > 0
    entries = bars.index[invested & ~invested.shift(1, fill_value=False)]
    exits = bars.index[~invested & invested.shift(1, fill_value=False)]

    rows = []
    for i, entry in enumerate(entries):
        still_open = i >= len(exits)
        # An open trade is marked to market at the last close, without exit costs.
        exit_date = bars.index[-1] if still_open else exits[i]
        exit_price = bars["close"].iloc[-1] if still_open else bars.at[exit_date, "open"]
        entry_price = bars.at[entry, "open"]
        sides = 1 if still_open else 2
        rows.append(
            {
                "entry_date": entry,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "bars_held": int(invested.loc[entry:exit_date].sum()),
                "return": exit_price / entry_price * (1 - cost_bps / 10_000) ** sides - 1,
                "open": still_open,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["entry_date", "exit_date", "entry_price", "exit_price", "bars_held", "return",
                 "open"],
    )
