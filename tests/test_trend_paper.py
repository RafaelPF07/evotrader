import numpy as np
import pandas as pd
import pytest

from evotrader.evolution import Dataset
from evotrader.paper import PaperStore
from evotrader.paper.trend_trader import TrendPaperTrader
from evotrader.trend import TrendLeverage
from tests.conftest import make_bars

RULE = TrendLeverage(sma=50, leverage=1.5)
START = "2016-06-01"


def make_datasets(seed_offset: int = 0) -> list[Dataset]:
    return [Dataset(f"T{i}", make_bars(n=700, seed=i + seed_offset)) for i in range(3)]


def make_trader(tmp_path, datasets, name="trend.db", rate=0.03) -> TrendPaperTrader:
    rates = pd.Series(rate, index=datasets[0].bars.index)
    return TrendPaperTrader(PaperStore(tmp_path / name), datasets, RULE, rates,
                            log=lambda _: None)


def test_accounting_including_interest_is_exact(tmp_path):
    trader = make_trader(tmp_path, make_datasets())
    trader.init(100_000, START)
    trader.run()
    s = trader.store
    last = pd.Timestamp(s.get("last_processed"))
    realised = s.frame("SELECT pnl FROM trades")["pnl"].sum()
    unrealised = sum(
        p.qty * (float(trader.bars[t]["close"].loc[:last].iloc[-1]) - p.avg_price)
        for t, p in s.positions().items()
    )
    equity = s.frame("SELECT equity FROM equity ORDER BY date")["equity"].iloc[-1]
    assert equity - 100_000 == pytest.approx(realised + unrealised + s.get("financing_total"),
                                             abs=1e-6)


def test_it_borrows_and_pays_for_it(tmp_path):
    trader = make_trader(tmp_path, make_datasets())
    trader.init(100_000, START)
    trader.run()
    cash = trader.store.frame("SELECT cash FROM equity")["cash"]
    assert cash.min() < 0  # leverage means negative cash on some days
    free = make_trader(tmp_path, make_datasets(), "free.db", rate=0.0)
    free.init(100_000, START)
    free.run()
    # With a 0% T-bill rate, borrowing still costs the 1% spread and cash earns nothing.
    assert free.store.get("financing_total") < 0


def test_holdings_follow_the_rule(tmp_path):
    datasets = make_datasets()
    trader = make_trader(tmp_path, datasets)
    trader.init(100_000, START)
    trader.run()
    s = trader.store
    last = pd.Timestamp(s.get("last_processed"))
    decided = trader.calendar[trader.calendar < last][-1]  # orders from here are filled
    for ds in datasets:
        close = ds.bars["close"]
        above = bool(close.loc[decided] > close.rolling(50).mean().loc[decided])
        assert (ds.ticker in s.positions()) == above
    assert (s.frame("SELECT reason FROM orders")["reason"].str.contains("50-day average")).all()


def test_no_future_information(tmp_path):
    base = make_datasets()
    cutoff = pd.Timestamp("2016-11-01")
    rng = np.random.default_rng(4)
    scrambled = []
    for ds in base:
        bars = ds.bars.copy()
        after = bars.index > cutoff
        bars.loc[after, ["open", "high", "low", "close"]] *= rng.uniform(
            0.6, 1.4, size=(after.sum(), 1))
        scrambled.append(Dataset(ds.ticker, bars))
    a, b = make_trader(tmp_path, base, "a.db"), make_trader(tmp_path, scrambled, "b.db")
    for t in (a, b):
        t.init(100_000, START)
        t.run(until=cutoff)
    for table in ("equity", "orders", "trades"):
        pd.testing.assert_frame_equal(a.store.frame(f"SELECT * FROM {table}"),
                                      b.store.frame(f"SELECT * FROM {table}"))
