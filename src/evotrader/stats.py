"""Is an edge real, or is it luck from trying many things? The deflated Sharpe ratio.

Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection
Bias, Backtest Overfitting and Non-Normality".

If you try N strategies that have no real edge, the *best* of them will still show a
positive Sharpe ratio by luck, and the more you try, the higher that lucky best gets.
The deflated Sharpe ratio (DSR) is the probability that a strategy's true Sharpe is
above what the best of N skill-less trials would show by chance, allowing for how
much data there is and for fat-tailed or skewed returns.

    DSR > 0.95   evidence of a real edge at the usual 95% confidence
    DSR ~ 0.5    indistinguishable from the luckiest of N coin-flippers

Applied here to *active* returns (strategy minus buy & hold), it answers the question
this project keeps asking: does it really beat buy & hold?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import pandas as pd

EULER_GAMMA = 0.5772156649015329
Z = NormalDist()


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe (per period) among `n_trials` skill-less strategies."""
    if n_trials <= 1:
        return 0.0
    a = Z.inv_cdf(1 - 1 / n_trials)
    b = Z.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def probabilistic_sharpe(sr: float, benchmark: float, n_obs: int, skew: float,
                         kurtosis: float) -> float:
    """P(true Sharpe > benchmark), from a per-period Sharpe estimated on n_obs returns.

    `kurtosis` is the raw kurtosis (3 for a normal distribution), not the excess.
    """
    denom = 1 - skew * sr + (kurtosis - 1) / 4 * sr ** 2
    if n_obs < 2 or denom <= 0:
        return float("nan")
    return Z.cdf((sr - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom))


@dataclass(frozen=True)
class Significance:
    n_obs: int
    n_trials: int
    sharpe_annual: float  # annualised Sharpe of the tested series
    skew: float
    kurtosis: float
    luck_threshold_annual: float  # annualised Sharpe the luckiest of N nulls would show
    psr: float  # P(true Sharpe > 0), ignoring how many things were tried
    dsr: float  # P(true Sharpe > luck threshold), the deflated Sharpe ratio


def deflated_sharpe(returns: pd.Series, n_trials: int, periods_per_year: int = 252
                    ) -> Significance:
    """Deflated Sharpe ratio of a daily return series (e.g. strategy minus benchmark).

    The variance of Sharpe estimates across trials is taken as its value under the
    null of no skill, 1 / (T - 1), i.e. independent trials. Real trials here are
    correlated (similar ideas on similar data), so the effective number of trials is
    lower than the count; report several N to show how much that matters.
    """
    r = returns.dropna()
    n = len(r)
    std = r.std()
    sr = float(r.mean() / std) if std > 0 else 0.0
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0
    threshold = expected_max_sharpe(n_trials, 1 / (n - 1))
    root = math.sqrt(periods_per_year)
    return Significance(
        n_obs=n, n_trials=n_trials, sharpe_annual=sr * root, skew=skew, kurtosis=kurt,
        luck_threshold_annual=threshold * root,
        psr=probabilistic_sharpe(sr, 0.0, n, skew, kurt),
        dsr=probabilistic_sharpe(sr, threshold, n, skew, kurt),
    )
