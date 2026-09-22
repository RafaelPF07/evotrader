import numpy as np
import pandas as pd
import pytest

from evotrader.evolution import Dataset, Genome
from evotrader.evolution.genome import Compare
from evotrader.features import FeatureSpec
from evotrader.paper import PaperStore
from evotrader.paper.ensemble_trader import EnsemblePaperTrader
from tests.conftest import make_bars

START = "2016-06-01"
GENOMES = [
    Genome(entry=Compare(FeatureSpec("rsi", (5,)), "<", 45.0),
           exit=Compare(FeatureSpec("rsi", (5,)), ">", 60.0)),
    Genome(entry=Compare(FeatureSpec("trend", (50,)), ">", 0.0),
           exit=Compare(FeatureSpec("trend", (50,)), "<", 0.0)),
    Genome(entry=Compare(FeatureSpec("zscore", (20,)), "<", -0.5),
           exit=Compare(FeatureSpec("zscore", (20,)), ">", 0.8)),
]
MODEL = {"name": "test ensemble", "rules": [{"seed": i, "rule": str(g)}
                                            for i, g in enumerate(GENOMES)]}


def make_datasets(scramble_after=None):
    rng = np.random.default_rng(3)
    out = []
    for i in range(3):
        bars = make_bars(n=700, seed=i)
        if scramble_after is not None:
            after = bars.index > scramble_after
            bars.loc[after, ["open", "high", "low", "close"]] *= rng.uniform(
                0.6, 1.4, size=(after.sum(), 1))
        out.append(Dataset(f"T{i}", bars))
    return out


def make_trader(tmp_path, datasets, name="ens.db"):
    return EnsemblePaperTrader(PaperStore(tmp_path / name), datasets, GENOMES,
                               log=lambda _: None)


def test_accounting_is_exact_with_partial_trims(tmp_path):
    trader = make_trader(tmp_path, make_datasets())
    trader.init(100_000, START, MODEL)
    trader.run()
    s = trader.store
    last = pd.Timestamp(s.get("last_processed"))
    realised = s.frame("SELECT pnl FROM trades")["pnl"].sum()
    unrealised = sum(
        p.qty * (float(trader.bars[t]["close"].loc[:last].iloc[-1]) - p.avg_price)
        for t, p in s.positions().items())
    equity = s.frame("SELECT equity FROM equity ORDER BY date")["equity"].iloc[-1]
    assert equity - 100_000 == pytest.approx(realised + unrealised, abs=1e-6)
    orders = s.frame("SELECT side, reason FROM orders")
    # Partial moves happen: some buys top up and some sells trim (not only 0 <-> full).
    assert orders["reason"].str.contains("/3 rules say in").all()
    assert orders["reason"].str.contains(r"[12]/3").any()
    assert (s.frame("SELECT cash FROM equity")["cash"] > -1e-6).all()  # never borrows


def test_position_size_follows_the_votes(tmp_path):
    datasets = make_datasets()
    trader = make_trader(tmp_path, datasets)
    trader.init(100_000, START, MODEL)
    trader.run()
    s = trader.store
    last = pd.Timestamp(s.get("last_processed"))
    decided = trader.calendar[trader.calendar < last][-1]
    votes = s.get("votes")
    for ds in datasets:
        today = int(round(trader.votes(ds.ticker)[ds.bars.index.get_loc(last)]))
        filled = int(round(trader.votes(ds.ticker)[ds.bars.index.get_loc(decided)]))
        assert votes.get(ds.ticker, 0) == today  # latest decision (fills tomorrow)
        assert (ds.ticker in s.positions()) == (filled > 0)  # yesterday's, already filled


def test_no_future_information(tmp_path):
    cutoff = pd.Timestamp("2016-11-01")
    a = make_trader(tmp_path, make_datasets(), "a.db")
    b = make_trader(tmp_path, make_datasets(scramble_after=cutoff), "b.db")
    for t in (a, b):
        t.init(100_000, START, MODEL)
        t.run(until=cutoff)
    for table in ("equity", "orders", "trades"):
        pd.testing.assert_frame_equal(a.store.frame(f"SELECT * FROM {table}"),
                                      b.store.frame(f"SELECT * FROM {table}"))
