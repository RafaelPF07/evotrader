import numpy as np
import pandas as pd
import pytest

from evotrader import experiments as ex
from evotrader.evolution import Dataset, EvolutionConfig, Evolver, FitnessConfig, Genome, evaluate
from evotrader.evolution.fitness import truncate
from evotrader.evolution.genome import Compare
from evotrader.evolution.rotation import Panel, RotationFactory, RotationGenome, Term
from evotrader.features import PRICE_KINDS, FeatureSpec
from tests.conftest import make_bars

MOM = FeatureSpec("mom", (20,))
VOL = FeatureSpec("vol", (20,))


def make_datasets(n: int = 900, k: int = 4) -> list[Dataset]:
    return [Dataset(f"T{i}", make_bars(n=n, seed=i)) for i in range(k)]


def rotation() -> RotationGenome:
    return RotationGenome((Term(MOM, 1.0), Term(VOL, -0.5)), top_n=2, rebalance=10)


def test_rotation_roundtrip_and_str():
    g = rotation()
    assert RotationGenome.from_dict(g.to_dict()) == g
    assert str(g) == "ROTATE top 2 every 10d by +1.00*mom(20) -0.50*vol(20)"


def test_weights_hold_top_n_and_only_change_on_rebalance_days():
    panel = Panel(make_datasets())
    w = panel.weights(rotation())
    invested = w.sum(axis=1)
    first = int(np.argmax(invested > 0))
    assert np.allclose(invested[first:], 1.0)  # always fully invested once features exist
    assert ((w[first:] > 0).sum(axis=1) == 2).all()
    changes = np.flatnonzero(np.abs(np.diff(w, axis=0)).sum(axis=1) > 0) + 1
    assert (changes % 10 == 0).all()


def test_rotation_has_no_lookahead():
    base = make_datasets()
    cutoff = 600
    rng = np.random.default_rng(3)
    scrambled = []
    for ds in base:
        bars = ds.bars.copy()
        bars.iloc[cutoff + 1:, :4] *= rng.uniform(0.6, 1.4, size=(len(bars) - cutoff - 1, 1))
        scrambled.append(Dataset(ds.ticker, bars))
    a = Panel(base).simulate(Panel(base).weights(rotation()), 5)[0]
    b = Panel(scrambled).simulate(Panel(scrambled).weights(rotation()), 5)[0]
    pd.testing.assert_series_equal(a.iloc[: cutoff + 1], b.iloc[: cutoff + 1])


def test_equal_weight_rotation_matches_buy_and_hold():
    """Holding every ticker, never trading, should be buy & hold (after the first fill)."""
    datasets = make_datasets()
    panel = Panel(datasets)
    everything = RotationGenome((Term(MOM, 1.0),), top_n=len(datasets), rebalance=5)
    start, end = panel.index[300], panel.index[-1]
    rot, _ = panel.portfolio(everything, start, end, 5)
    bh = panel.buy_and_hold(start, end, 5)
    assert (rot - bh).abs().max() < 1e-12


def test_rotation_factory_stays_valid():
    f = RotationFactory(np.random.default_rng(0), PRICE_KINDS, max_n=5)
    g = f.random_genome()
    for _ in range(300):
        g = f.mutate(f.crossover(g, f.random_genome()))
        assert 1 <= len(g.terms) <= 3 and 1 <= g.top_n <= 5 and g.rebalance in (5, 10, 21, 63)
        assert RotationGenome.from_dict(g.to_dict()) == g


def test_excess_objective_scores_buy_and_hold_as_zero():
    always = Genome(entry=Compare(FeatureSpec("rsi", (2,)), ">", -1.0),
                    exit=Compare(FeatureSpec("rsi", (2,)), "<", -1.0))
    ev = evaluate(always, make_datasets(), "2016-01-01", "2018-06-01",
                  FitnessConfig(objective="excess"))
    assert ev.score_mean == pytest.approx(0.0, abs=1e-9)


def test_invalid_objective_rejected():
    with pytest.raises(ValueError):
        FitnessConfig(objective="profit")


def test_rotation_evolution_smoke_and_elitism():
    config = EvolutionConfig(population=12, generations=4, seed=1)
    result = Evolver(make_datasets(), "2016-01-01", "2017-12-31", config, space="rotation").run()
    best = [s.best_fitness for s in result.history]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(best, best[1:], strict=False))
    assert str(result.best[0]).startswith("ROTATE")


def test_dev_stage_cannot_see_past_cutoff():
    datasets = [Dataset(f"T{i}", make_bars(n=2400, seed=i)) for i in range(3)]  # to ~2024
    data, folds = ex.stage_setup("dev", datasets)
    assert all(ds.bars.index[-1] <= pd.Timestamp(ex.DEV_END) for ds in data)
    assert folds[-1].test_end <= pd.Timestamp(ex.DEV_END)
    _, hold = ex.stage_setup("holdout", datasets)
    assert len(hold) == 1 and hold[0].test_start == pd.Timestamp("2022-01-01")
    assert hold[0].train_end == pd.Timestamp(ex.DEV_END)


def test_truncate_physically_removes_future():
    data = truncate(make_datasets(), "2016-01-01")
    assert all(ds.bars.index[-1] < pd.Timestamp("2016-01-02") for ds in data)


def test_holdout_refuses_to_run_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "RESULTS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="development"):
        ex.check_allowed("holdout", "A")
    for arm in ex.ARMS:
        ex.result_path("dev", arm).write_text("x")
    ex.check_allowed("holdout", "A")
    ex.result_path("holdout", "A").write_text("x")
    with pytest.raises(RuntimeError, match="exactly once"):
        ex.check_allowed("holdout", "A")


def test_pass_rule():
    res = pd.DataFrame({"excess_sharpe": [0.1, 0.2, -0.05, 0.3, 0.1]})
    assert ex.verdict(res)["passed"]
    res = pd.DataFrame({"excess_sharpe": [0.5, 0.5, -0.1, -0.1, 0.1]})
    assert not ex.verdict(res)["passed"]  # good mean, but only 3/5 seeds
