import numpy as np
import pandas as pd
import pytest

from evotrader.evolution import Dataset, EvolutionConfig, Genome
from evotrader.evolution.genome import Compare
from evotrader.features import FeatureSpec, FeatureStore
from evotrader.paper import LearnerConfig, PaperStore, PaperTrader, analyse_mistakes
from tests.conftest import make_bars

LEARNER = LearnerConfig(
    train_start="2015-01-01",
    holdout_days=126,
    relearn_every=40,
    evolution=EvolutionConfig(population=10, generations=3),
)


def make_datasets(n: int = 1300) -> list[Dataset]:
    return [Dataset(t, make_bars(n=n, seed=i)) for i, t in enumerate(["AAA", "BBB", "CCC"])]


def make_trader(tmp_path, datasets, name="paper.db") -> PaperTrader:
    return PaperTrader(PaperStore(tmp_path / name), datasets, LEARNER, log=lambda _: None)


START = "2018-06-01"


def test_replay_accounting_is_exact(tmp_path):
    trader = make_trader(tmp_path, make_datasets())
    trader.init(100_000, START)
    days = trader.run()
    s = trader.store

    equity = s.frame("SELECT * FROM equity ORDER BY date")
    assert len(equity) == days + 1
    assert (equity["cash"] > -1e-6).all()

    # Every dollar is accounted for: realised P&L + unrealised P&L = equity change.
    last = pd.Timestamp(s.get("last_processed"))
    realised = s.frame("SELECT pnl FROM trades")["pnl"].sum()
    unrealised = sum(
        p.qty * (float(trader.bars[t]["close"].loc[:last].iloc[-1]) - p.avg_price)
        for t, p in s.positions().items()
    )
    assert equity["equity"].iloc[-1] - 100_000 == pytest.approx(realised + unrealised, abs=1e-6)
    assert len(s.frame("SELECT * FROM trades")) > 0


def test_no_future_information_in_replay(tmp_path):
    """Two worlds identical up to the cutoff must produce identical accounts up to it."""
    base = make_datasets()
    cutoff = pd.Timestamp("2019-03-01")
    rng = np.random.default_rng(9)
    scrambled = []
    for ds in base:
        bars = ds.bars.copy()
        after = bars.index > cutoff
        bars.loc[after, ["open", "high", "low", "close"]] *= rng.uniform(
            0.6, 1.4, size=(after.sum(), 1)
        )
        scrambled.append(Dataset(ds.ticker, bars))

    a = make_trader(tmp_path, base, "a.db")
    b = make_trader(tmp_path, scrambled, "b.db")
    for trader in (a, b):
        trader.init(100_000, START)
        trader.run(until=cutoff)

    for table in ("equity", "orders", "trades", "strategies", "learning_log"):
        pd.testing.assert_frame_equal(
            a.store.frame(f"SELECT * FROM {table}"), b.store.frame(f"SELECT * FROM {table}")
        )
    assert a.store.get("relearn_count") > 0  # learning happened inside the window


def test_run_is_resumable(tmp_path):
    """Running in two chunks gives the same account as running in one go."""
    once = make_trader(tmp_path, make_datasets(), "once.db")
    once.init(100_000, START)
    once.run()

    twice = make_trader(tmp_path, make_datasets(), "twice.db")
    twice.init(100_000, START)
    twice.run(until=pd.Timestamp("2019-01-15"))
    twice.run()

    pd.testing.assert_frame_equal(once.store.frame("SELECT * FROM equity"),
                                  twice.store.frame("SELECT * FROM equity"))


def test_learning_is_logged(tmp_path):
    trader = make_trader(tmp_path, make_datasets())
    trader.init(100_000, START)
    trader.run()
    log = trader.store.frame("SELECT * FROM learning_log")
    assert len(log) == trader.store.get("relearn_count") >= 2
    assert log["notes"].str.contains("incumbent").all()
    active = trader.store.frame("SELECT * FROM strategies WHERE active = 1")
    assert len(active) == 1


def test_orders_carry_reasons(tmp_path):
    trader = make_trader(tmp_path, make_datasets())
    trader.init(100_000, START)
    trader.run(until=pd.Timestamp("2019-01-01"))
    reasons = trader.store.frame("SELECT reason FROM orders")["reason"]
    assert len(reasons) > 0
    assert reasons.str.contains("entry").all() and reasons.str.contains("strategy #").all()


def test_cannot_init_twice(tmp_path):
    trader = make_trader(tmp_path, make_datasets())
    trader.init(100_000, START)
    with pytest.raises(RuntimeError):
        trader.init(100_000, START)


def test_mistake_analysis_finds_filters_that_help():
    datasets = make_datasets()
    genome = Genome(entry=Compare(FeatureSpec("rsi", (2,)), "<", 30.0),
                    exit=Compare(FeatureSpec("rsi", (2,)), ">", 60.0))
    lessons = analyse_mistakes(genome, datasets, "2015-06-01", "2019-12-31", cost_bps=5)
    assert lessons
    for lesson in lessons:
        assert lesson.return_gain > 0
        assert lesson.losers_avoided + lesson.winners_lost <= lesson.trades / 2


def test_explain_shows_live_values(bars):
    genome = Genome(entry=Compare(FeatureSpec("rsi", (14,)), "<", 30.0),
                    exit=Compare(FeatureSpec("rsi", (14,)), ">", 70.0))
    text = genome.explain(FeatureStore(bars), -1)
    assert text.startswith("entry") and "rsi(14)=" in text and "exit" in text
