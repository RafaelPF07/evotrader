"""A live paper account for a fixed ensemble of evolved rules (forward test, account 3).

Exploration round 2's recipe: several independently evolved in/out rules using price
and macro building blocks, combined equally. For each ETF, at every close:

    k of K rules say "in"  ->  hold k/K of that ETF's equal slot (equity / N)

Positions are resized only when k changes, so the account trims or tops up part of a
position rather than trading daily. Cash earns nothing, as in the backtest. The rules
never change during the forward test.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from evotrader.backtest import DEFAULT_COST_BPS
from evotrader.evolution.fitness import Dataset
from evotrader.evolution.genome import Genome
from evotrader.paper.store import PaperStore
from evotrader.paper.trader import PaperTrader, _day

Log = Callable[[str], None]


class EnsemblePaperTrader(PaperTrader):
    def __init__(self, store: PaperStore, datasets: list[Dataset], genomes: list[Genome],
                 cost_bps: float = DEFAULT_COST_BPS, log: Log = print) -> None:
        super().__init__(store, datasets, cost_bps=cost_bps, log=log)
        self.genomes = genomes
        self._votes: dict[str, np.ndarray] = {}

    def votes(self, ticker: str) -> np.ndarray:
        """How many rules say "in" for this ETF on each day (causal features only)."""
        if ticker not in self._votes:
            store = self.stores[ticker]
            self._votes[ticker] = sum(g.target_position(store) for g in self.genomes)
        return self._votes[ticker]

    # --- setup ----------------------------------------------------------------
    def init(self, capital: float, start: str, model: dict) -> None:
        if self.store.initialised:
            raise RuntimeError(f"{self.store.path} already holds an account")
        before = self.calendar[self.calendar < pd.Timestamp(start)]
        if before.empty:
            raise ValueError("Start date is before the available data")
        anchor = before[-1]
        s = self.store
        s.set("strategy", "ensemble")
        s.set("model", model)
        s.set("initial_capital", capital)
        s.cash = capital
        s.set("tickers", [ds.ticker for ds in self.datasets])
        s.set("start", _day(anchor))
        s.set("last_processed", _day(anchor))
        s.set("votes", {})
        s.record_equity(_day(anchor), capital, 0.0)
        self._decide(anchor)
        s.commit()
        self.log(f"Ensemble account created at {_day(anchor)} with {len(self.genomes)} rules")

    # --- hooks ------------------------------------------------------------------
    def _relearn_due(self, d: pd.Timestamp) -> bool:
        return False  # fixed rules for the forward test

    def _strategy_id(self) -> int:
        return 0

    def _decide(self, d: pd.Timestamp) -> None:
        s = self.store
        held = s.get("votes", {})  # ticker -> votes behind the current position
        positions = s.positions()
        pending = {o.ticker for o in s.pending_orders()}
        equity = s.cash + self.holdings_value(d)
        k_total, n = len(self.genomes), len(self.bars)

        for ticker, bars in self.bars.items():
            if d not in bars.index or ticker in pending:
                continue
            i = bars.index.get_loc(d)
            k = int(round(self.votes(ticker)[i]))
            if k == held.get(ticker, 0):
                continue
            close = float(bars.at[d, "close"])
            target_qty = equity / n * k / k_total / close
            have = positions[ticker].qty if ticker in positions else 0.0
            agree = [str(g) for g in self.genomes if g.target_position(self.stores[ticker])[i]]
            reason = (f"ensemble on {_day(d)}: {k}/{k_total} rules say in"
                      + (f" ({'; '.join(agree)})" if agree else ""))
            if k == 0 and have > 0:
                s.add_order(_day(d), ticker, "sell", have, reason)
            elif target_qty > have:
                s.add_order(_day(d), ticker, "buy", target_qty - have, reason)
            elif target_qty < have:
                s.add_order(_day(d), ticker, "sell", have - target_qty, reason)
            held[ticker] = k
        s.set("votes", held)
