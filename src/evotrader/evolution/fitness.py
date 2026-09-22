"""How good is a genome? Scored across every ticker at once, over one date range.

fitness = mean Sharpe across tickers
        - consistency_weight * std of Sharpe across tickers   (don't rely on one lucky ticker)
        - complexity_penalty * rule size                       (Occam's razor vs overfitting)
        - penalties for barely trading or barely being invested (a rule that trades twice
          in ten years has a meaningless Sharpe ratio)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from evotrader.backtest import DEFAULT_COST_BPS, simulate
from evotrader.evolution.genome import Genome
from evotrader.features import FeatureStore
from evotrader.metrics import TRADING_DAYS, sharpe


@dataclass
class Dataset:
    ticker: str
    bars: pd.DataFrame
    store: FeatureStore = field(init=False)

    def __post_init__(self) -> None:
        self.store = FeatureStore(self.bars)


@dataclass(frozen=True)
class FitnessConfig:
    consistency_weight: float = 0.5
    complexity_penalty: float = 0.01
    min_trades_per_year: float = 1.0
    min_exposure: float = 0.10
    cost_bps: float = DEFAULT_COST_BPS


@dataclass(frozen=True)
class Evaluation:
    fitness: float
    sharpe_mean: float
    sharpe_std: float
    trades_per_year: float
    exposure: float


def evaluate(
    genome: Genome, datasets: list[Dataset], start: str, end: str, config: FitnessConfig
) -> Evaluation:
    sharpes, trade_rates, exposures = [], [], []
    for ds in datasets:
        target = pd.Series(genome.target_position(ds.store), index=ds.bars.index)
        returns, held = simulate(ds.bars, target, config.cost_bps)
        returns, held = returns.loc[start:end], held.loc[start:end]
        if len(returns) < TRADING_DAYS // 4:  # ticker has too little data in this window
            continue
        invested = held.to_numpy() > 0
        entries = np.count_nonzero(invested[1:] & ~invested[:-1]) + int(invested[0])
        sharpes.append(sharpe(returns))
        trade_rates.append(entries / (len(returns) / TRADING_DAYS))
        exposures.append(invested.mean())

    if not sharpes:
        return Evaluation(-10.0, 0.0, 0.0, 0.0, 0.0)

    s_mean, s_std = float(np.mean(sharpes)), float(np.std(sharpes))
    tpy, exposure = float(np.mean(trade_rates)), float(np.mean(exposures))
    fitness = (
        s_mean
        - config.consistency_weight * s_std
        - config.complexity_penalty * genome.size
        - max(0.0, 1 - tpy / config.min_trades_per_year)
        - max(0.0, 1 - exposure / config.min_exposure)
    )
    return Evaluation(fitness, s_mean, s_std, tpy, exposure)
