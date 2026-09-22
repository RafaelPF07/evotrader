import hashlib

import numpy as np
import pandas as pd
import pytest

from evotrader import explore, ledger
from evotrader.evolution import Dataset, EvolutionConfig, Evolver, FitnessConfig, Genome, evaluate
from evotrader.evolution.engine import available_kinds
from evotrader.evolution.genome import Compare
from evotrader.features import MACRO_KINDS, PRICE_KINDS, FeatureSpec, FeatureStore
from evotrader.macro import attach_macro
from tests.conftest import make_bars


def datasets(n: int = 900) -> list[Dataset]:
    return [Dataset(f"T{i}", make_bars(n=n, seed=i)) for i in range(4)]


def fake_macro(index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    walk = lambda: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index)))),  # noqa: E731
                             index=index)
    cols = {c: walk() for c in ("m_credit", "m_inflation", "m_dollar", "m_oil",
                                "m_copper_gold", "m_breadth", "m_small_large", "m_cyclical",
                                "m_em", "m_spy")}
    cols.update({"m_vix": 20 + walk() / 10, "m_vix_term": 0.9 + walk() / 1000,
                 "m_vvix": 90 + walk() / 10, "m_skew": 120 + walk() / 10,
                 "m_curve": 1 + walk() / 100, "m_tnx": 3 + walk() / 100})
    return pd.DataFrame(cols)


def test_live_bot_evolution_is_unchanged():
    """Golden fingerprint recorded before the macro building blocks were added."""
    out = []
    for obj in ("sharpe", "excess"):
        r = Evolver(datasets(), "2015-06-01", "2017-12-31",
                    EvolutionConfig(population=20, generations=8, seed=3),
                    FitnessConfig(objective=obj)).run()
        out.append(";".join(f"{s.best_fitness:.10f}|{s.best_rule}" for s in r.history))
    digest = hashlib.sha1("\n".join(out).encode()).hexdigest()
    assert digest == "132fadb16885894eaf4ac427e4b7456863f0a01a"


def test_macro_kinds_only_offered_when_attached():
    plain = datasets()
    assert available_kinds(plain) == PRICE_KINDS
    with_macro = attach_macro(plain, fake_macro(plain[0].bars.index))
    kinds = available_kinds(with_macro)
    assert kinds[:len(PRICE_KINDS)] == PRICE_KINDS and set(MACRO_KINDS) <= set(kinds)


def test_every_macro_feature_computes_and_is_causal():
    base = datasets()
    idx = base[0].bars.index
    macro = fake_macro(idx)
    cutoff = 600
    changed = macro.copy()
    changed.iloc[cutoff + 1:] *= 1.7
    a, b = attach_macro(base, macro)[0], attach_macro(base, changed)[0]
    for kind in MACRO_KINDS:
        from evotrader.features import KINDS
        k = KINDS[kind]
        spec = FeatureSpec(kind, (k.windows[0],) if k.n_windows else ())
        va, vb = FeatureStore(a.bars).get(spec), FeatureStore(b.bars).get(spec)
        assert np.isfinite(va[300:]).any(), kind
        np.testing.assert_array_equal(va[: cutoff + 1], vb[: cutoff + 1], err_msg=kind)


def test_macro_alignment_only_carries_values_forward():
    idx = pd.bdate_range("2020-01-01", periods=10, name="date")
    bars = make_bars(n=10)
    bars.index = idx
    macro = pd.DataFrame({"m_spy": [1.0, 2.0, 3.0]},
                         index=pd.DatetimeIndex(["2020-01-01", "2020-01-04", "2020-01-08"]))
    out = attach_macro([Dataset("X", bars)], macro)[0].bars["m_spy"]
    # 2020-01-04 is a Saturday: its value first appears on Monday 01-06, never earlier.
    assert out.loc["2020-01-03"] == 1.0 and out.loc["2020-01-06"] == 2.0
    assert out.loc["2020-01-07"] == 2.0 and out.loc["2020-01-08"] == 3.0


@pytest.mark.parametrize("today,tom,halloween", [
    ("2024-01-30", 1.0, 0.0),  # tomorrow (Jan 31) is the last business day of January
    ("2024-01-10", 0.0, 0.0),
    ("2024-02-01", 1.0, 0.0),  # tomorrow is the 2nd business day of February
    ("2024-02-05", 0.0, 0.0),  # tomorrow is the 4th business day
    ("2024-04-30", 1.0, 1.0),  # tomorrow is May 1st: turn of month and May-October
    ("2024-10-31", 1.0, 0.0),  # tomorrow is November 1st
])
def test_calendar_flags_describe_tomorrow(today, tom, halloween):
    idx = pd.DatetimeIndex([pd.Timestamp(today)])
    bars = pd.DataFrame({"close": [1.0]}, index=idx)
    store = FeatureStore(bars)
    assert store.get(FeatureSpec("tom"))[0] == tom
    assert store.get(FeatureSpec("halloween"))[0] == halloween


def test_exposure_cap_penalises_buy_and_hold_clones():
    always = Genome(entry=Compare(FeatureSpec("rsi", (2,)), ">", -1.0),
                    exit=Compare(FeatureSpec("rsi", (2,)), "<", -1.0))
    ds = datasets()
    free = evaluate(always, ds, "2016-01-01", "2018-01-01", FitnessConfig(objective="excess"))
    capped = evaluate(always, ds, "2016-01-01", "2018-01-01",
                      FitnessConfig(objective="excess", max_exposure=0.8))
    assert free.exposure > 0.95 and capped.fitness <= free.fitness - 1.0


def test_ledger_counts_every_trial(tmp_path):
    path = tmp_path / "trials.csv"
    assert ledger.total_trials(path) == ledger.HISTORICAL_TRIALS
    assert ledger.next_trial(path) == ledger.HISTORICAL_TRIALS + 1
    t = ledger.record("r1", "test", {"a": 1}, {"sharpe": 0.5, "dsr": 0.1}, path)
    assert t == ledger.HISTORICAL_TRIALS + 1 and ledger.total_trials(path) == t


def test_summary_flags_buy_and_hold_clones():
    idx = pd.bdate_range("2015-01-01", periods=500)
    bh = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, len(idx)), index=idx)
    s = explore.summarise(bh, bh, pd.Series(1.0, index=idx), 40)
    assert s["corr_with_bh"] == pytest.approx(1.0) and s["exposure"] == 1.0
    assert not s["beats_bh"] and s["active_sharpe"] == 0.0
