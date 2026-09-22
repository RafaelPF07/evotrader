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
    requires: str | None = None  # a bars column this feature needs (macro signals)


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
        # --- market-wide signals (see macro.py); only offered when attached -------
        FeatureKind("vix", 0, (), 10.0, 60.0, 1, "m_vix"),
        FeatureKind("vix_mom", 1, (5, 20, 63), -0.5, 1.0, 2, "m_vix"),
        FeatureKind("vix_term", 0, (), 0.70, 1.30, 2, "m_vix_term"),
        FeatureKind("vvix", 0, (), 70.0, 160.0, 0, "m_vvix"),
        FeatureKind("skew", 0, (), 110.0, 160.0, 0, "m_skew"),
        FeatureKind("curve", 0, (), -1.5, 4.0, 2, "m_curve"),
        FeatureKind("curve_chg", 1, (20, 63, 126, 252), -1.5, 1.5, 2, "m_curve"),
        FeatureKind("rates_chg", 1, (20, 63, 126, 252), -1.5, 1.5, 2, "m_tnx"),
        FeatureKind("credit_mom", 1, (5, 20, 63, 126), -0.10, 0.10, 3, "m_credit"),
        FeatureKind("inflation_mom", 1, (20, 63, 126), -0.05, 0.05, 3, "m_inflation"),
        FeatureKind("dollar_mom", 1, (20, 63, 126), -0.10, 0.10, 3, "m_dollar"),
        FeatureKind("oil_mom", 1, (20, 63, 126), -0.50, 0.50, 2, "m_oil"),
        FeatureKind("copper_gold_mom", 1, (20, 63, 126), -0.20, 0.20, 3, "m_copper_gold"),
        FeatureKind("breadth_mom", 1, (20, 63, 126), -0.08, 0.08, 3, "m_breadth"),
        FeatureKind("small_large_mom", 1, (20, 63, 126), -0.15, 0.15, 3, "m_small_large"),
        FeatureKind("cyclical_mom", 1, (20, 63, 126), -0.15, 0.15, 3, "m_cyclical"),
        FeatureKind("em_mom", 1, (20, 63, 126), -0.15, 0.15, 3, "m_em"),
        FeatureKind("rel_strength", 1, (20, 63, 126, 252), -0.20, 0.20, 3, "m_spy"),
        FeatureKind("tom", 0, (), 0.0, 1.0, 1),  # turn of the month: 1 on those days
        FeatureKind("halloween", 0, (), 0.0, 1.0, 1),  # 1 from May to October
    )
}
# The original building blocks, in their original order. The live evolved bot draws
# from exactly this tuple, so adding kinds above never changes its behaviour.
PRICE_KINDS = ("rsi", "zscore", "mom", "trend", "ma_spread", "vol")
MACRO_KINDS = tuple(k for k in KINDS if k not in PRICE_KINDS and k != ML_COLUMN)


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
    kind = KINDS[spec.kind]
    if kind.requires is not None and kind.requires not in bars:
        return pd.Series(np.nan, index=bars.index)  # macro data not attached
    match spec.kind:
        case "vix" | "vix_term" | "vvix" | "skew" | "curve":
            return bars[kind.requires]
        case "curve_chg" | "rates_chg":  # yields: change in percentage points
            return bars[kind.requires].diff(w[0])
        case "rel_strength":  # this ETF's return minus the S&P 500's
            return close.pct_change(w[0]) - bars["m_spy"].pct_change(w[0])
        case "tom":
            return _next_day_calendar(bars.index)[0]
        case "halloween":
            return _next_day_calendar(bars.index)[1]
    if spec.kind.endswith("_mom"):  # momentum of a macro level or ratio
        return bars[kind.requires].pct_change(w[0])
    raise KeyError(spec.kind)


def _next_day_calendar(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Calendar flags for the *next* business day, known in advance from the calendar.

    A rule reads them at today's close to decide tomorrow's position, so the flag
    says whether tomorrow is a turn-of-the-month day (the last business day of a
    month or one of the first three) and whether tomorrow falls in May to October.
    """
    from pandas.tseries.offsets import BDay

    nxt = index + BDay(1)
    last_of_month = (nxt + BDay(1)).month != nxt.month
    month_start = nxt.to_period("M").to_timestamp()
    bdays_in = np.array([len(pd.bdate_range(s, d)) for s, d in zip(month_start, nxt,
                                                                   strict=True)])
    tom = pd.Series((last_of_month | (bdays_in <= 3)).astype(float), index=index)
    halloween = pd.Series(nxt.month.isin(range(5, 11)).astype(float), index=index)
    return tom, halloween


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
