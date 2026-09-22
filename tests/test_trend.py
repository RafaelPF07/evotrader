import numpy as np
import pandas as pd
import pytest

from evotrader import experiments3 as ex3
from evotrader.evolution import Dataset
from evotrader.evolution.rotation import Panel
from evotrader.sizing import VolTarget
from evotrader.trend import TrendLeverage, financing, simulate, trend_mask, weights
from tests.conftest import make_bars

NO_RATES = pd.Series(dtype=float)


def make_panel(n: int = 900, k: int = 4) -> Panel:
    return Panel([Dataset(f"T{i}", make_bars(n=n, seed=i)) for i in range(k)])


def test_mask_matches_moving_average_and_waits_for_warmup():
    panel = make_panel()
    mask = trend_mask(panel, 50)
    closes = pd.DataFrame(panel.close)
    expected = (closes > closes.rolling(50).mean()).to_numpy(dtype=float)
    assert np.array_equal(mask, expected)
    assert mask[:49].sum() == 0


def test_weights_are_leverage_over_n_when_above_average():
    panel = make_panel()
    w = weights(panel, TrendLeverage(sma=50, leverage=1.5))
    assert set(np.unique(np.round(w * 4, 12))) <= {0.0, 1.5}
    assert w.sum(axis=1).max() <= 1.5 + 1e-12


def test_vol_target_only_scales_down_from_the_cap():
    panel = make_panel()
    vt = VolTarget(min_history=100)
    plain = weights(panel, TrendLeverage(sma=50, leverage=1.5))
    scaled = weights(panel, TrendLeverage(sma=50, leverage=1.5, vol_target=vt))
    assert (scaled <= plain + 1e-12).all()
    assert (scaled[plain == 0] == 0).all()


def test_trend_strategy_is_causal():
    base = [Dataset(f"T{i}", make_bars(n=900, seed=i)) for i in range(4)]
    cutoff = 600
    rng = np.random.default_rng(2)
    scrambled = []
    for ds in base:
        bars = ds.bars.copy()
        bars.iloc[cutoff + 1:, :4] *= rng.uniform(0.5, 1.5, size=(len(bars) - cutoff - 1, 1))
        scrambled.append(Dataset(ds.ticker, bars))
    tl = TrendLeverage(sma=50, leverage=1.5)
    rates = pd.Series(0.03, index=Panel(base).index)
    a, _ = simulate(Panel(base), tl, 5, rates)
    b, _ = simulate(Panel(scrambled), tl, 5, rates)
    pd.testing.assert_series_equal(a.iloc[: cutoff + 1], b.iloc[: cutoff + 1])


def test_financing_pays_on_cash_and_charges_on_borrowing():
    held = pd.Series([0.0, 0.5, 1.0, 1.5], index=pd.bdate_range("2020-01-01", periods=4))
    rates = pd.Series(0.0252, index=held.index)  # 2.52% a year = 0.0001 a day
    f = financing(held, rates, spread=0.0252, cash_earns=True)
    assert np.allclose(f.to_numpy(), [0.0001, 0.00005, 0.0, -0.5 * 0.0002])
    assert financing(held, rates, 0.0, cash_earns=False).iloc[:3].eq(0).all()


def test_never_above_average_earns_exactly_t_bills():
    panel = make_panel()
    tl = TrendLeverage(sma=1, leverage=1.5)  # a 1-day average: price is never strictly above it
    rates = pd.Series(0.0252, index=panel.index)
    returns, held = simulate(panel, tl, 5, rates)
    assert (held == 0).all() and np.allclose(returns, 0.0001)


def test_always_above_average_without_leverage_is_buy_and_hold():
    idx = pd.bdate_range("2015-01-01", periods=400, name="date")
    rising = []
    for i in range(3):
        close = 100 * np.exp(np.cumsum(np.full(len(idx), 0.002 + 0.001 * i)))
        rising.append(Dataset(f"R{i}", pd.DataFrame(
            {"open": close * 0.999, "high": close * 1.001, "low": close * 0.998,
             "close": close, "volume": 1e6}, index=idx)))
    panel = Panel(rising)
    returns, held = simulate(panel, TrendLeverage(sma=50, leverage=1.0), 5, NO_RATES)
    start = panel.index[52]  # after the warm-up and the first fill
    bh = panel.buy_and_hold(start, panel.index[-1], 5)
    assert np.allclose(held.loc[start:], 1.0)
    assert (returns.loc[start:] - bh).abs().max() < 1e-12


def test_sharpe_is_measured_over_t_bills():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rates = pd.Series(0.0252, index=idx)
    cash_only = pd.Series(0.0001, index=idx)  # exactly the T-bill return
    s = ex3.score(cash_only, pd.Series(0.0, index=idx), rates)
    assert s["sharpe"] == 0.0 and s["days_in_cash"] == 1.0


def test_final_universes_were_never_used_before():
    from evotrader.data import DEFAULT_UNIVERSE
    from evotrader.experiments2 import INTERNATIONAL, US_2000
    used = set(DEFAULT_UNIVERSE) | set(INTERNATIONAL) | set(US_2000)
    assert used.isdisjoint(ex3.SECTORS) and used.isdisjoint(ex3.COUNTRIES)
    assert not ex3.STAGES["post-publication"].judged


def test_finals_require_dev_and_run_once(tmp_path, monkeypatch):
    monkeypatch.setattr(ex3, "RESULTS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="dev stage first"):
        ex3.check_allowed("final-sectors")
    ex3.result_path("dev").write_text("x")
    ex3.check_allowed("final-sectors")
    ex3.result_path("final-sectors").write_text("x")
    with pytest.raises(RuntimeError, match="exactly once"):
        ex3.check_allowed("final-sectors")
