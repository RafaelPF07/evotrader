"""ETF rotation genomes (experiment idea 2).

Instead of timing each ETF in or out (which leaves cash idle ~25% of the time),
a rotation strategy is always fully invested. Every `rebalance` days it ranks
the ETFs by a weighted score and holds the top N equally:

    ROTATE top 3 every 21d by +0.80*mom(126) -0.40*vol(20)

Each feature is converted to a cross-sectional percentile rank (0..1 across the
tickers on that day), so weights are comparable across features. When stocks
weaken, money moves to whatever ranks best, which may be bonds (TLT) or gold (GLD).
Execution timing is the same as everywhere else: decide at the close, fill at the
next open, pay costs on every change in weight.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from evotrader.evolution.fitness import Dataset, Evaluation, FitnessConfig
from evotrader.features import KINDS, FeatureSpec
from evotrader.metrics import TRADING_DAYS, information_ratio, sharpe

REBALANCE_DAYS = (5, 10, 21, 63)
MAX_TERMS = 3


@dataclass(frozen=True)
class Term:
    feature: FeatureSpec
    weight: float


@dataclass(frozen=True)
class RotationGenome:
    terms: tuple[Term, ...]
    top_n: int
    rebalance: int

    @property
    def size(self) -> int:
        return len(self.terms) + 1

    def __str__(self) -> str:
        score = " ".join(f"{t.weight:+.2f}*{t.feature}" for t in self.terms)
        return f"ROTATE top {self.top_n} every {self.rebalance}d by {score}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "rotation", "top_n": self.top_n, "rebalance": self.rebalance,
            "terms": [{"feature": t.feature.kind, "windows": list(t.feature.windows),
                       "weight": t.weight} for t in self.terms],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RotationGenome:
        terms = tuple(Term(FeatureSpec(t["feature"], tuple(t["windows"])), float(t["weight"]))
                      for t in d["terms"])
        return cls(terms, int(d["top_n"]), int(d["rebalance"]))


class Panel:
    """All tickers aligned on their common trading days, as (days x tickers) arrays."""

    def __init__(self, datasets: list[Dataset]) -> None:
        start = max(ds.bars.index[0] for ds in datasets)
        index = datasets[0].bars.index[datasets[0].bars.index >= start]
        for ds in datasets[1:]:
            index = index.intersection(ds.bars.index)
        self.index = index
        self.tickers = [ds.ticker for ds in datasets]
        self._rows = [ds.bars.index.get_indexer(index) for ds in datasets]
        self._stores = [ds.store for ds in datasets]
        self.open = self._stack([ds.bars["open"].to_numpy(float) for ds in datasets])
        self.close = self._stack([ds.bars["close"].to_numpy(float) for ds in datasets])
        self._ranks: dict[FeatureSpec, np.ndarray] = {}

    def _stack(self, columns: list[np.ndarray]) -> np.ndarray:
        return np.column_stack([col[rows] for col, rows in zip(columns, self._rows, strict=True)])

    def rank(self, spec: FeatureSpec) -> np.ndarray:
        """Cross-sectional percentile rank of a feature on each day (NaN if missing)."""
        if spec not in self._ranks:
            raw = self._stack([store.get(spec) for store in self._stores])
            self._ranks[spec] = pd.DataFrame(raw).rank(axis=1, pct=True).to_numpy()
        return self._ranks[spec]

    def weights(self, genome: RotationGenome) -> np.ndarray:
        """Target weights decided at each close (held from the next open)."""
        score = sum(t.weight * self.rank(t.feature) for t in genome.terms)
        n_days, n_tickers = score.shape
        top_n = min(genome.top_n, n_tickers)
        weights = np.zeros((n_days, n_tickers))
        current = np.zeros(n_tickers)
        for t in range(0, n_days, genome.rebalance):  # anchored to the panel start
            row = score[t]
            valid = ~np.isnan(row)
            if valid.sum() >= top_n:
                best = np.argsort(-np.where(valid, row, -np.inf), kind="stable")[:top_n]
                current = np.zeros(n_tickers)
                current[best] = 1.0 / top_n
            weights[t:t + genome.rebalance] = current
        return weights

    def simulate(self, weights: np.ndarray, cost_bps: float) -> tuple[pd.Series, pd.Series]:
        """Daily portfolio returns and exposure, with the backtester's timing and costs."""
        held = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
        held_prev = np.vstack([np.zeros((1, weights.shape[1])), held[:-1]])
        prev_close = np.vstack([self.open[:1], self.close[:-1]])
        gap = self.open / prev_close - 1
        intraday = self.close / self.open - 1
        cost = np.abs(held - held_prev) * cost_bps / 10_000
        per_ticker = (1 + held_prev * gap) * (1 - cost) * (1 + held * intraday) - 1
        returns = pd.Series(per_ticker.sum(axis=1), index=self.index)
        exposure = pd.Series(held.sum(axis=1), index=self.index)
        return returns, exposure

    def portfolio(
        self, genome: RotationGenome, start: str | pd.Timestamp, end: str | pd.Timestamp,
        cost_bps: float,
    ) -> tuple[pd.Series, pd.Series]:
        returns, exposure = self.simulate(self.weights(genome), cost_bps)
        return returns.loc[start:end], exposure.loc[start:end]

    def buy_and_hold(self, start: str | pd.Timestamp, end: str | pd.Timestamp,
                     cost_bps: float) -> pd.Series:
        equal = np.full(self.open.shape, 1.0 / self.open.shape[1])
        return self.simulate(equal, cost_bps)[0].loc[start:end]


def evaluate_rotation(
    genome: RotationGenome, panel: Panel, start: str, end: str, config: FitnessConfig
) -> Evaluation:
    """Portfolio-level fitness: Sharpe, or information ratio vs equal-weight buy & hold."""
    returns, _ = panel.portfolio(genome, start, end, config.cost_bps)
    if len(returns) < TRADING_DAYS // 4:
        return Evaluation(-10.0, 0.0, 0.0, 0.0, 0.0)
    if config.objective == "excess":
        score = information_ratio(returns - panel.buy_and_hold(start, end, config.cost_bps))
    else:
        score = sharpe(returns)
    rebalances_per_year = TRADING_DAYS / genome.rebalance
    fitness = score - config.complexity_penalty * genome.size
    return Evaluation(fitness, score, 0.0, rebalances_per_year, 1.0)


class RotationFactory:
    """Random generation and variation for rotation genomes (same interface as GenomeFactory)."""

    def __init__(self, rng: np.random.Generator, kinds: tuple[str, ...], max_n: int = 5) -> None:
        self.rng = rng
        self.kinds = kinds
        self.max_n = max_n

    def _feature(self) -> FeatureSpec:
        kind = KINDS[str(self.rng.choice(self.kinds))]
        if kind.n_windows == 0:
            return FeatureSpec(kind.name)
        windows = sorted(self.rng.choice(kind.windows, size=kind.n_windows, replace=False))
        return FeatureSpec(kind.name, tuple(int(w) for w in windows))

    def _weight(self) -> float:
        return round(float(self.rng.uniform(-1, 1)), 2)

    def _term(self) -> Term:
        return Term(self._feature(), self._weight())

    def random_genome(self) -> RotationGenome:
        n_terms = int(self.rng.integers(1, MAX_TERMS + 1))
        return RotationGenome(
            tuple(self._term() for _ in range(n_terms)),
            int(self.rng.integers(1, self.max_n + 1)),
            int(self.rng.choice(REBALANCE_DAYS)),
        )

    def mutate(self, g: RotationGenome) -> RotationGenome:
        terms = list(g.terms)
        i = int(self.rng.integers(len(terms)))
        roll = self.rng.random()
        if roll < 0.35:  # nudge a weight
            w = round(float(np.clip(terms[i].weight + self.rng.normal(0, 0.25), -1, 1)), 2)
            terms[i] = Term(terms[i].feature, w)
        elif roll < 0.5:  # new feature for an existing term
            terms[i] = Term(self._feature(), terms[i].weight)
        elif roll < 0.6 and len(terms) < MAX_TERMS:
            terms.append(self._term())
        elif roll < 0.7 and len(terms) > 1:
            terms.pop(i)
        elif roll < 0.85:
            step = int(self.rng.choice([-1, 1]))
            return replace(g, top_n=int(np.clip(g.top_n + step, 1, self.max_n)))
        else:
            return replace(g, rebalance=int(self.rng.choice(REBALANCE_DAYS)))
        return replace(g, terms=tuple(terms))

    def crossover(self, a: RotationGenome, b: RotationGenome) -> RotationGenome:
        pool = list(a.terms) + list(b.terms)
        k = int(self.rng.integers(1, min(MAX_TERMS, len(pool)) + 1))
        picks = self.rng.choice(len(pool), size=k, replace=False)
        parent = a if self.rng.random() < 0.5 else b
        return replace(parent, terms=tuple(pool[int(i)] for i in sorted(picks)))
