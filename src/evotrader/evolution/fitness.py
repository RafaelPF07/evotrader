"""How good is a genome? Scored across every ticker at once, over one date range.

Two objectives:

`sharpe` (the original):
    fitness = mean Sharpe across tickers
            - consistency_weight * std of Sharpe across tickers   (don't rely on one lucky ticker)
            - complexity_penalty * rule size                       (Occam's razor vs overfitting)
            - penalties for barely trading or barely being invested (a rule that trades twice
              in ten years has a meaningless Sharpe ratio)

`excess` (experiment idea 1): score each ticker by the *information ratio* of the
strategy against simply holding that ticker, i.e. how consistently it beats buy &
hold. Holding all the time scores exactly 0, so the search can fall back to buy &
hold when no timing helps; that's why the trading/exposure penalties are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from evotrader.backtest import DEFAULT_COST_BPS, simulate
from evotrader.evolution.genome import Genome
from evotrader.features import FeatureStore
from evotrader.metrics import TRADING_DAYS, information_ratio, sharpe

OBJECTIVES = ("sharpe", "excess")


@dataclass
class Dataset:
    ticker: str
    bars: pd.DataFrame
    store: FeatureStore = field(init=False)
    _buy_and_hold: dict[float, pd.Series] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.store = FeatureStore(self.bars)

    def buy_and_hold_returns(self, cost_bps: float) -> pd.Series:
        if cost_bps not in self._buy_and_hold:
            always = pd.Series(1.0, index=self.bars.index)
            self._buy_and_hold[cost_bps] = simulate(self.bars, always, cost_bps)[0]
        return self._buy_and_hold[cost_bps]


def truncate(datasets: list[Dataset], end: pd.Timestamp | str) -> list[Dataset]:
    """Datasets that physically contain no bars after `end`, so no code path can peek."""
    return [Dataset(ds.ticker, ds.bars.loc[:pd.Timestamp(end)]) for ds in datasets]


@dataclass(frozen=True)
class FitnessConfig:
    objective: str = "sharpe"
    consistency_weight: float = 0.5
    complexity_penalty: float = 0.01
    min_trades_per_year: float = 1.0
    min_exposure: float = 0.10
    cost_bps: float = DEFAULT_COST_BPS

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")


@dataclass(frozen=True)
class Evaluation:
    fitness: float
    score_mean: float  # mean per-ticker Sharpe (or information ratio for `excess`)
    score_std: float
    trades_per_year: float
    exposure: float


def evaluate(
    genome: Genome, datasets: list[Dataset], start: str, end: str, config: FitnessConfig
) -> Evaluation:
    scores, trade_rates, exposures = [], [], []
    for ds in datasets:
        target = pd.Series(genome.target_position(ds.store), index=ds.bars.index)
        returns, held = simulate(ds.bars, target, config.cost_bps)
        returns, held = returns.loc[start:end], held.loc[start:end]
        if len(returns) < TRADING_DAYS // 4:  # ticker has too little data in this window
            continue
        invested = held.to_numpy() > 0
        entries = np.count_nonzero(invested[1:] & ~invested[:-1]) + int(invested[0])
        if config.objective == "excess":
            active = returns - ds.buy_and_hold_returns(config.cost_bps).loc[start:end]
            scores.append(information_ratio(active))
        else:
            scores.append(sharpe(returns))
        trade_rates.append(entries / (len(returns) / TRADING_DAYS))
        exposures.append(invested.mean())

    if not scores:
        return Evaluation(-10.0, 0.0, 0.0, 0.0, 0.0)

    s_mean, s_std = float(np.mean(scores)), float(np.std(scores))
    tpy, exposure = float(np.mean(trade_rates)), float(np.mean(exposures))
    fitness = (
        s_mean
        - config.consistency_weight * s_std
        - config.complexity_penalty * genome.size
    )
    if config.objective == "sharpe":
        fitness -= max(0.0, 1 - tpy / config.min_trades_per_year)
        fitness -= max(0.0, 1 - exposure / config.min_exposure)
    return Evaluation(fitness, s_mean, s_std, tpy, exposure)
