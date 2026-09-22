import numpy as np
import pandas as pd
import pytest

from evotrader.evolution import Dataset, EvolutionConfig, Evolver, FitnessConfig, Genome
from evotrader.evolution.genome import Compare
from evotrader.evolution.switch import SwitchBook, evaluate_switch
from evotrader.features import FeatureSpec
from tests.conftest import make_bars

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
RSI = FeatureSpec("rsi", (14,))


def datasets(n: int = 900, scramble_after: int | None = None) -> list[Dataset]:
    out = []
    rng = np.random.default_rng(7)
    for i, t in enumerate(TICKERS):
        bars = make_bars(n=n, seed=i)
        if scramble_after is not None:
            bars.iloc[scramble_after + 1:, :4] *= rng.uniform(
                0.6, 1.4, size=(n - scramble_after - 1, 1))
        out.append(Dataset(t, bars))
    return out


def always(on: bool) -> Genome:
    yes, no = Compare(RSI, ">", -1.0), Compare(RSI, "<", -1.0)
    return Genome(entry=yes if on else no, exit=no if on else yes)


def test_always_fully_invested_in_one_basket():
    book = SwitchBook(datasets())
    g = Genome(entry=Compare(RSI, "<", 40.0), exit=Compare(RSI, ">", 60.0))
    w = book.weights(g)
    assert np.allclose(w.sum(axis=1), 1.0)
    stocks, defensive = w[:, :3].sum(axis=1), w[:, 3:].sum(axis=1)
    assert set(np.round(stocks, 12)) <= {0.0, 1.0} and np.allclose(stocks + defensive, 1.0)


def test_extremes_equal_the_two_baskets():
    book = SwitchBook(datasets())
    start, end = book.panel.index[50], book.panel.index[-1]
    on, _ = book.portfolio(always(True), start, end, 0)
    off, _ = book.portfolio(always(False), start, end, 0)
    closes = pd.DataFrame(book.panel.close, index=book.panel.index,
                          columns=book.panel.tickers).pct_change().loc[start:end]
    # Close-to-close after the first day; the panel fills at the open, so compare gently.
    assert on.corr(closes[["SPY", "QQQ", "IWM"]].mean(axis=1)) > 0.99
    assert off.corr(closes[["TLT", "GLD"]].mean(axis=1)) > 0.99


def test_switch_is_causal():
    cutoff = 600
    a, b = SwitchBook(datasets()), SwitchBook(datasets(scramble_after=cutoff))
    g = Genome(entry=Compare(RSI, "<", 45.0), exit=Compare(RSI, ">", 55.0))
    ra, _ = a.portfolio(g, a.panel.index[0], a.panel.index[-1], 5)
    rb, _ = b.portfolio(g, b.panel.index[0], b.panel.index[-1], 5)
    pd.testing.assert_series_equal(ra.iloc[: cutoff + 1], rb.iloc[: cutoff + 1])


def test_cap_forbids_staying_in_stocks():
    book = SwitchBook(datasets())
    free = evaluate_switch(always(True), book, "2016-01-01", "2018-01-01",
                           FitnessConfig(objective="sharpe"))
    capped = evaluate_switch(always(True), book, "2016-01-01", "2018-01-01",
                             FitnessConfig(objective="sharpe", max_exposure=0.8))
    assert free.exposure == pytest.approx(1.0) and capped.fitness <= free.fitness - 1.0


def test_needs_defensive_assets():
    with pytest.raises(ValueError):
        SwitchBook([d for d in datasets() if d.ticker != "TLT"])


def test_switch_evolution_smoke():
    r = Evolver(datasets(), "2016-01-01", "2017-12-31",
                EvolutionConfig(population=10, generations=3, seed=1),
                FitnessConfig(objective="sharpe", max_exposure=0.8), space="switch").run()
    assert r.best[1].exposure <= 1.0 and r.history
