import numpy as np
import pandas as pd
import pytest

from evotrader import indicators as ind
from evotrader.data import drop_incomplete, validate
from evotrader.metrics import cagr, max_drawdown, sharpe


def test_max_drawdown_known_path():
    # Equity: 1 -> 1.1 -> 0.88 -> 0.968 ; worst fall is 1.1 -> 0.88 = -20%
    returns = pd.Series([0.10, -0.20, 0.10])
    assert max_drawdown(returns) == pytest.approx(-0.20)


def test_sharpe_zero_for_flat_returns():
    assert sharpe(pd.Series([0.0] * 100)) == 0.0


def test_cagr_one_year():
    daily = (1.10) ** (1 / 252) - 1
    assert cagr(pd.Series([daily] * 252)) == pytest.approx(0.10)


def test_sma_values():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert ind.sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    assert ind.sma(s, 3).isna().sum() == 2


def test_rsi_bounds(bars):
    r = ind.rsi(bars["close"]).dropna()
    assert r.between(0, 100).all()


def test_rsi_all_gains_is_100():
    r = ind.rsi(pd.Series(np.arange(1.0, 40.0)), window=14)
    assert r.dropna().eq(100).all()


def test_partial_bar_dropped_while_market_open(bars):
    today = bars.index[-1]
    during = pd.Timestamp(today.date()).tz_localize("America/New_York") + pd.Timedelta(hours=14)
    after = during + pd.Timedelta(hours=3)
    assert drop_incomplete(bars, now=during).index[-1] < today
    assert drop_incomplete(bars, now=after).index[-1] == today


def test_validate_rejects_unsorted(bars):
    with pytest.raises(ValueError):
        validate(bars.iloc[::-1])
