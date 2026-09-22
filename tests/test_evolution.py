import numpy as np
import pandas as pd
import pytest

from evotrader.evolution import (
    Dataset,
    EvolutionConfig,
    Evolver,
    FitnessConfig,
    GeneticStrategy,
    Genome,
    GenomeFactory,
    evaluate,
)
from evotrader.evolution.genome import And, Compare, Not
from evotrader.features import PRICE_KINDS, FeatureSpec, FeatureStore
from evotrader.walkforward import make_folds, run_walkforward
from tests.conftest import make_bars

RSI2 = FeatureSpec("rsi", (2,))
TREND = FeatureSpec("trend", (50,))


def sample_genome() -> Genome:
    return Genome(
        entry=And(Compare(RSI2, "<", 20.0), Compare(TREND, ">", 0.0)),
        exit=Compare(RSI2, ">", 70.0),
    )


@pytest.fixture
def factory() -> GenomeFactory:
    return GenomeFactory(np.random.default_rng(1), PRICE_KINDS, max_depth=3)


@pytest.fixture
def datasets() -> list[Dataset]:
    return [Dataset(f"T{i}", make_bars(n=900, seed=i)) for i in range(3)]


def test_serialisation_roundtrip(factory):
    for _ in range(50):
        g = factory.random_genome()
        assert str(Genome.from_dict(g.to_dict())) == str(g)


def test_variation_respects_max_depth(factory):
    g = factory.random_genome()
    for _ in range(300):
        other = factory.random_genome()
        g = factory.mutate(factory.crossover(g, other))
        assert g.depth <= factory.max_depth + 1


def test_factory_is_reproducible():
    def draw() -> list[str]:
        factory = GenomeFactory(np.random.default_rng(7), PRICE_KINDS)
        return [str(factory.random_genome()) for _ in range(20)]

    assert draw() == draw()


def test_positions_are_binary_and_wait_for_warmup(bars):
    g = Genome(entry=Not(Compare(TREND, ">", 5.0)), exit=Compare(RSI2, ">", 101.0))
    pos = g.target_position(FeatureStore(bars))
    assert set(np.unique(pos)) <= {0.0, 1.0}
    # NOT(trend > 5) is "true" on missing data; it must still not enter before 50 bars exist.
    assert pos[:49].sum() == 0 and pos[49] == 1.0


def test_genetic_strategy_save_load(tmp_path, bars):
    strat = GeneticStrategy(sample_genome())
    strat.save(tmp_path / "champ.json", {"note": "test"})
    loaded = GeneticStrategy.load(tmp_path / "champ.json")
    pd.testing.assert_series_equal(strat.target_position(bars), loaded.target_position(bars))


def test_fitness_penalises_rules_that_never_trade(datasets):
    never = Genome(entry=Compare(RSI2, "<", -1.0), exit=Compare(RSI2, ">", 50.0))
    cfg = FitnessConfig()
    ev = evaluate(never, datasets, "2015-06-01", "2018-06-01", cfg)
    assert ev.trades_per_year == 0
    assert ev.fitness < evaluate(sample_genome(), datasets, "2015-06-01", "2018-06-01", cfg).fitness


def test_elitism_never_loses_best(datasets):
    config = EvolutionConfig(population=16, generations=6, elite=2, seed=3)
    result = Evolver(datasets, "2015-06-01", "2017-12-31", config).run()
    best = [s.best_fitness for s in result.history]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(best, best[1:], strict=False))
    assert result.best[1].fitness == pytest.approx(best[-1])


def test_evolution_is_reproducible(datasets):
    config = EvolutionConfig(population=12, generations=3, seed=5)
    a = Evolver(datasets, "2015-06-01", "2017-12-31", config).run()
    b = Evolver(datasets, "2015-06-01", "2017-12-31", config).run()
    assert str(a.best[0]) == str(b.best[0])


def test_folds_never_overlap_test_with_training():
    folds = make_folds("2007-01-01", 2016, pd.Timestamp("2026-09-22"), test_years=2)
    assert [f.test_start.year for f in folds] == [2016, 2018, 2020, 2022, 2024]
    assert folds[-1].test_end == pd.Timestamp("2026-09-22")
    for f in folds:
        assert f.train_start < f.val_start < f.train_end < f.test_start <= f.test_end
    for prev, nxt in zip(folds, folds[1:], strict=False):
        assert prev.test_end < nxt.test_start


def test_walkforward_smoke(datasets):
    last = datasets[0].bars.index[-1]
    folds = make_folds("2015-01-01", 2017, last, test_years=1)
    report = run_walkforward(datasets, folds, EvolutionConfig(population=8, generations=2),
                             log=lambda _: None)
    assert len(report.folds) == len(folds)
    assert {"evolved", "buy_and_hold"} <= set(report.oos_metrics.index)
    assert report.oos_returns.index.is_monotonic_increasing
