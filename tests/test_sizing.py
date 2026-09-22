import numpy as np
import pandas as pd
import pytest

from evotrader import experiments2 as ex2
from evotrader.evolution import Dataset
from evotrader.evolution.rotation import Panel
from evotrader.sizing import VolTarget, exposure_path, simulate
from tests.conftest import make_bars

NO_RATES = pd.Series(dtype=float)


def make_panel(n: int = 900, k: int = 4, bars=None) -> Panel:
    return Panel(bars or [Dataset(f"T{i}", make_bars(n=n, seed=i)) for i in range(k)])


def test_exposure_respects_cap_and_floor():
    for cap in (1.0, 1.5):
        e = exposure_path(make_panel(), VolTarget(cap=cap, target_scale=3.0, min_history=100))
        assert (e >= 0).all() and (e <= cap + 1e-12).all()
        assert np.isclose(e.max(), cap)  # a big target scale must hit the cap


def test_band_suppresses_small_changes():
    panel = make_panel()
    tight = exposure_path(panel, VolTarget(band=0.0, min_history=100))
    banded = exposure_path(panel, VolTarget(band=0.25, min_history=100))
    changes = lambda e: int((np.diff(e) != 0).sum())  # noqa: E731
    assert changes(banded) < changes(tight) / 3
    moves = np.abs(np.diff(banded))[np.diff(banded) != 0]
    before = banded[:-1][np.diff(banded) != 0]
    assert (moves > 0.25 * before - 1e-12).all()  # every change clears the band


def test_exposure_is_causal():
    base = [Dataset(f"T{i}", make_bars(n=900, seed=i)) for i in range(4)]
    cutoff = 600
    rng = np.random.default_rng(1)
    scrambled = []
    for ds in base:
        bars = ds.bars.copy()
        bars.iloc[cutoff + 1:, :4] *= rng.uniform(0.5, 1.5, size=(len(bars) - cutoff - 1, 1))
        scrambled.append(Dataset(ds.ticker, bars))
    vt = VolTarget(min_history=100)
    a, b = exposure_path(Panel(base), vt), exposure_path(Panel(scrambled), vt)
    assert np.array_equal(a[: cutoff + 1], b[: cutoff + 1])
    ra, _ = simulate(Panel(base), vt, 5, NO_RATES)
    rb, _ = simulate(Panel(scrambled), vt, 5, NO_RATES)
    pd.testing.assert_series_equal(ra.iloc[: cutoff + 1], rb.iloc[: cutoff + 1])


def test_full_exposure_without_leverage_is_buy_and_hold():
    panel = make_panel()
    vt = VolTarget(cap=1.0, target_scale=1e6, min_history=100)  # always pinned at 1.0
    returns, held = simulate(panel, vt, 5, NO_RATES)
    start = panel.index[5]
    bh = panel.buy_and_hold(start, panel.index[-1], 5)
    assert (returns.loc[start:] - bh).abs().max() < 1e-12
    assert np.allclose(held.iloc[1:], 1.0)


def test_borrowing_is_charged_only_on_the_levered_part():
    panel = make_panel()
    vt = VolTarget(cap=1.5, target_scale=1.3, min_history=100, borrow_spread=0.01)
    free_vt = VolTarget(cap=1.5, target_scale=1.3, min_history=100, borrow_spread=0.0)
    rates = pd.Series(0.04, index=panel.index)
    free, held = simulate(panel, free_vt, 5, NO_RATES)  # borrowing costs nothing
    paid, _ = simulate(panel, vt, 5, rates)  # T-bill 4% + 1% spread
    borrowed = (held - 1).clip(lower=0)
    assert (free - paid - borrowed * 0.05 / 252).abs().max() < 1e-12
    assert (held > 1).any() and ((free - paid)[held <= 1] == 0).all()


def test_final_stages_physically_exclude_the_future():
    stage = ex2.STAGES["final-us2000"]
    assert stage.data_end == "2006-12-31" and "TLT" not in stage.tickers
    assert set(ex2.INTERNATIONAL).isdisjoint(ex2.DEFAULT_UNIVERSE)
    assert ex2.STAGES["dev"].data_end == "2021-12-31"


def test_grid_is_the_preregistered_27_settings():
    assert len(ex2.GRID) == 27 and all(vt.cap == 1.5 for vt in ex2.GRID)


def test_finals_require_dev_and_run_once(tmp_path, monkeypatch):
    monkeypatch.setattr(ex2, "RESULTS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="dev stage first"):
        ex2.check_allowed("final-intl")
    ex2.result_path("dev").write_text("x")
    ex2.tuned_path().write_text("{}")
    ex2.check_allowed("final-intl")
    ex2.result_path("final-intl").write_text("x")
    with pytest.raises(RuntimeError, match="exactly once"):
        ex2.check_allowed("final-intl")
