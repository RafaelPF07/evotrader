import numpy as np
import pandas as pd
import pytest

from evotrader.backtest import run_backtest, simulate
from evotrader.evolution import GeneticStrategy, Genome
from evotrader.evolution.genome import Compare, Not, Or
from evotrader.features import FeatureSpec
from evotrader.strategies import REGISTRY, BuyAndHold, SmaCrossover
from evotrader.strategies.base import Strategy


def default_strategies() -> list[Strategy]:
    evolved = GeneticStrategy(Genome(
        entry=Or(Compare(FeatureSpec("rsi", (2,)), "<", 25.0),
                 Compare(FeatureSpec("ma_spread", (10, 50)), ">", 0.01)),
        exit=Not(Compare(FeatureSpec("zscore", (20,)), "<", 1.0)),
    ))
    return [cls() for cls in REGISTRY.values()] + [SmaCrossover(fast=5, slow=20), evolved]


@pytest.mark.parametrize("strategy", default_strategies(), ids=repr)
def test_no_lookahead(bars, strategy):
    """Scrambling the future must not change any decision or return up to the cutoff."""
    cutoff = 400
    future = bars.copy()
    rng = np.random.default_rng(42)
    future.iloc[cutoff + 1 :, :4] *= rng.uniform(0.5, 1.5, size=(len(bars) - cutoff - 1, 1))

    original = run_backtest(strategy, bars)
    scrambled = run_backtest(strategy, future)

    pd.testing.assert_series_equal(
        original.position.iloc[: cutoff + 2], scrambled.position.iloc[: cutoff + 2]
    )
    pd.testing.assert_series_equal(
        original.returns.iloc[: cutoff + 1], scrambled.returns.iloc[: cutoff + 1]
    )


def test_cannot_profit_from_same_day_signal(bars):
    """A signal that 'knows' today's close > open must not earn today's move."""
    today_up = (bars["close"] > bars["open"]).astype(float)
    returns, held = simulate(bars, today_up, cost_bps=0)
    # Position during day d was set from day d-1's bar, so it is uncorrelated with day d.
    assert (held == today_up.shift(1).fillna(0)).all()
    intraday = bars["close"] / bars["open"] - 1
    assert (held * intraday).mean() < intraday[intraday > 0].mean() / 2


def test_buy_and_hold_matches_price_ratio(bars):
    cost_bps = 10
    result = run_backtest(BuyAndHold(), bars, cost_bps=cost_bps)
    # Bought at the open of day 1 (decided at close of day 0), held to the last close.
    expected = bars["close"].iloc[-1] / bars["open"].iloc[1] * (1 - cost_bps / 10_000) - 1
    assert result.metrics["total_return"] == pytest.approx(expected, rel=1e-9)


def test_costs_reduce_returns(bars):
    strat = SmaCrossover(fast=5, slow=20)
    free = run_backtest(strat, bars, cost_bps=0).metrics["total_return"]
    cheap = run_backtest(strat, bars, cost_bps=5).metrics["total_return"]
    pricey = run_backtest(strat, bars, cost_bps=50).metrics["total_return"]
    assert free > cheap > pricey


def test_trades_are_consistent_with_positions(bars):
    result = run_backtest(SmaCrossover(fast=5, slow=20), bars)
    trades = result.trades
    assert len(trades) > 0
    assert (trades["exit_date"] >= trades["entry_date"]).all()
    assert trades["bars_held"].sum() == int((result.position > 0).sum())
    assert trades["open"].sum() <= 1


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        SmaCrossover(fast=50, slow=20)
