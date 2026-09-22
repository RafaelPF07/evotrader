import numpy as np
import pandas as pd
import pytest

from evotrader.stats import deflated_sharpe, expected_max_sharpe, probabilistic_sharpe


def series(mean: float, n: int = 2500, seed: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mean, 0.01, n))


def test_expected_max_sharpe_grows_with_trials():
    v = 1 / 2500
    values = [expected_max_sharpe(n, v) for n in (1, 2, 10, 100, 1000)]
    assert values[0] == 0.0
    assert all(b > a for a, b in zip(values, values[1:], strict=False))


def test_expected_max_matches_simulation():
    """The formula should match the average best Sharpe of N pure-noise strategies."""
    rng = np.random.default_rng(1)
    n_trials, n_obs = 20, 1000
    best = [
        max((x := rng.normal(0, 1, n_obs)).mean() / x.std() for _ in range(n_trials))
        for _ in range(300)
    ]
    assert np.mean(best) == pytest.approx(expected_max_sharpe(n_trials, 1 / (n_obs - 1)),
                                          rel=0.1)


def test_psr_basics():
    assert probabilistic_sharpe(0.0, 0.0, 1000, 0.0, 3.0) == pytest.approx(0.5)
    assert probabilistic_sharpe(0.1, 0.0, 1000, 0.0, 3.0) > 0.99
    assert probabilistic_sharpe(-0.1, 0.0, 1000, 0.0, 3.0) < 0.01


def test_deflation_lowers_confidence_as_trials_grow():
    r = series(0.0005)  # a modest real edge: ~0.8 annual Sharpe
    few, many = deflated_sharpe(r, 1), deflated_sharpe(r, 1000)
    assert few.dsr == pytest.approx(few.psr)  # one trial: nothing to deflate
    assert many.dsr < few.dsr
    assert many.luck_threshold_annual > 0


def test_noise_is_not_significant_after_deflation():
    for seed in range(5):
        assert deflated_sharpe(series(0.0, seed=seed), 36).dsr < 0.95


def test_strong_edge_survives_deflation():
    assert deflated_sharpe(series(0.002), 36).dsr > 0.99
