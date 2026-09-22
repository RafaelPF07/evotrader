"""Normalised features that evolved rules are built from.

Every feature is scale-free (a ratio, z-score, percentage or probability), so a
rule like `trend(200) > 0.02` means the same thing on SPY at $100 or $600, and on
any ticker. That lets one rule be scored across the whole universe at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evotrader import indicators as ind

ML_COLUMN = "ml_prob"


@dataclass(frozen=True)
class FeatureKind:
    name: str
    n_windows: int
    windows: tuple[int, ...]  # allowed window lengths
    lo: float  # sensible threshold range
    hi: float
    decimals: int  # thresholds are rounded, keeping rules readable and cacheable


KINDS: dict[str, FeatureKind] = {
    k.name: k
    for k in (
        FeatureKind("rsi", 1, (2, 5, 14, 28), 5.0, 95.0, 1),
        FeatureKind("zscore", 1, (10, 20, 50, 100), -2.5, 2.5, 2),
        FeatureKind("mom", 1, (5, 20, 63, 126, 252), -0.15, 0.15, 3),
        FeatureKind("trend", 1, (10, 20, 50, 100, 200), -0.10, 0.10, 3),
        FeatureKind("ma_spread", 2, (5, 10, 20, 50, 100, 200), -0.08, 0.08, 3),
        FeatureKind("vol", 1, (10, 20, 63), 0.05, 0.50, 2),
        FeatureKind(ML_COLUMN, 0, (), 0.35, 0.65, 2),
    )
}
PRICE_KINDS = tuple(k for k in KINDS if k != ML_COLUMN)


@dataclass(frozen=True)
class FeatureSpec:
    kind: str
    windows: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        spec = KINDS[self.kind]
        if len(self.windows) != spec.n_windows:
            raise ValueError(f"{self.kind} needs {spec.n_windows} window(s), got {self.windows}")
        if self.kind == "ma_spread" and self.windows[0] >= self.windows[1]:
            raise ValueError("ma_spread needs fast < slow")

    def __str__(self) -> str:
        return f"{self.kind}({','.join(map(str, self.windows))})" if self.windows else self.kind


def compute(spec: FeatureSpec, bars: pd.DataFrame) -> pd.Series:
    close = bars["close"]
    w = spec.windows
    match spec.kind:
        case "rsi":
            return ind.rsi(close, w[0])
        case "zscore":
            return ind.zscore(close, w[0])
        case "mom":
            return ind.momentum(close, w[0])
        case "trend":
            return close / ind.sma(close, w[0]) - 1
        case "ma_spread":
            return ind.sma(close, w[0]) / ind.sma(close, w[1]) - 1
        case "vol":
            return ind.volatility(close, w[0])
        case "ml_prob":
            if ML_COLUMN not in bars:
                return pd.Series(np.nan, index=bars.index)
            return bars[ML_COLUMN]
    raise KeyError(spec.kind)


class FeatureStore:
    """Computes each feature once per dataset. The GA scores thousands of rules
    that share a few dozen features, so this cache is what makes evolution fast."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars = bars
        self._cache: dict[FeatureSpec, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.bars)

    def get(self, spec: FeatureSpec) -> np.ndarray:
        if spec not in self._cache:
            self._cache[spec] = compute(spec, self.bars).to_numpy(dtype=float)
        return self._cache[spec]
