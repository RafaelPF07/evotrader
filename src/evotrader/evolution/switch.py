"""Risk-on / risk-off switching (exploration round 4).

One evolved rule, read on the market as a whole (SPY's price plus the macro
signals), decides each day between two fully invested baskets:

    rule says "in"   ->  risk-on:  equal weight across the stock ETFs
    rule says "out"  ->  risk-off: equal weight across the defensive ETFs (TLT, GLD)

Unlike the in/out timing rules, money never sits idle in cash, which is what cost
those rules most of their return. With `max_exposure` in the fitness, the rule must
spend at least that share of the time out of stocks, so it cannot become buy & hold.
Execution follows the backtester: decide at the close, switch at the next open,
paying costs on every change of weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evotrader.evolution.fitness import Dataset, Evaluation, FitnessConfig
from evotrader.evolution.genome import Genome
from evotrader.evolution.rotation import Panel
from evotrader.features import FeatureStore
from evotrader.metrics import TRADING_DAYS, information_ratio, sharpe

DEFENSIVE = ("TLT", "GLD")
MARKET = "SPY"


class SwitchBook:
    def __init__(self, datasets: list[Dataset], defensive: tuple[str, ...] = DEFENSIVE,
                 market: str = MARKET) -> None:
        self.panel = Panel(datasets)
        tickers = self.panel.tickers
        if not set(defensive) <= set(tickers) or market not in tickers:
            raise ValueError(f"switching needs {market} and {defensive} in the universe")
        is_def = np.array([t in defensive for t in tickers])
        self.risk_on = np.where(is_def, 0.0, 1.0 / (~is_def).sum())
        self.risk_off = np.where(is_def, 1.0 / is_def.sum(), 0.0)
        market_bars = next(ds.bars for ds in datasets if ds.ticker == market)
        self.store = FeatureStore(market_bars)  # macro columns come along if attached
        self.market_index = market_bars.index

    def signal(self, genome: Genome) -> np.ndarray:
        """1 = risk-on, 0 = risk-off, decided at each close, on the panel's days."""
        s = pd.Series(genome.target_position(self.store), index=self.market_index)
        return s.reindex(self.panel.index).ffill().fillna(0.0).to_numpy()

    def weights(self, genome: Genome) -> np.ndarray:
        p = self.signal(genome)[:, None]
        return p * self.risk_on + (1 - p) * self.risk_off

    def portfolio(self, genome: Genome, start, end, cost_bps: float
                  ) -> tuple[pd.Series, pd.Series]:
        """Daily returns and the share held in stocks (the 'exposure' that is capped)."""
        returns, _ = self.panel.simulate(self.weights(genome), cost_bps)
        in_stocks = pd.Series(self.signal(genome), index=self.panel.index).shift(1).fillna(0.0)
        return returns.loc[start:end], in_stocks.loc[start:end]

    def buy_and_hold(self, start, end, cost_bps: float) -> pd.Series:
        return self.panel.buy_and_hold(start, end, cost_bps)


def evaluate_switch(genome: Genome, book: SwitchBook, start: str, end: str,
                    config: FitnessConfig) -> Evaluation:
    returns, in_stocks = book.portfolio(genome, start, end, config.cost_bps)
    if len(returns) < TRADING_DAYS // 4:
        return Evaluation(-10.0, 0.0, 0.0, 0.0, 0.0)
    if config.objective == "excess":
        score = information_ratio(returns - book.buy_and_hold(start, end, config.cost_bps))
    else:
        score = sharpe(returns)
    exposure = float(in_stocks.mean())
    switches = float((in_stocks.diff().abs() > 0).sum() / (len(in_stocks) / TRADING_DAYS))
    fitness = score - config.complexity_penalty * genome.size
    if config.max_exposure is not None and exposure > config.max_exposure:
        fitness -= 1.0 + 10 * (exposure - config.max_exposure)
    if config.objective == "sharpe":  # a rule that never switches has no timing to judge
        fitness -= max(0.0, 1 - switches / config.min_trades_per_year)
    return Evaluation(fitness, score, 0.0, switches, exposure)
