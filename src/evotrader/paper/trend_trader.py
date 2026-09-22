"""A live paper account for the leveraged trend strategy (the forward test).

Same daily loop and database as the evolved bot (fill at the open, mark at the
close, decide for tomorrow), but with experiment 3's rule instead of an evolved one:

    ETF closes above its 200-day average  ->  hold a slot of equity x leverage / N
    ETF closes below it                   ->  sell; the money waits in cash

Cash earns the 3-month T-bill rate; when leverage pushes cash below zero, the
borrowed amount costs the T-bill rate + 1% a year. Positions are sized when opened
and not rebalanced daily (unlike the research backtest), which is how a person
would actually run it. Nothing is re-learned: the point is to see how a fixed,
pre-registered rule does on days nobody has seen.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from evotrader.backtest import DEFAULT_COST_BPS
from evotrader.evolution.fitness import Dataset
from evotrader.metrics import TRADING_DAYS
from evotrader.paper.store import PaperStore
from evotrader.paper.trader import PaperTrader, _day
from evotrader.trend import TrendLeverage

Log = Callable[[str], None]


class TrendPaperTrader(PaperTrader):
    allow_borrowing = True

    def __init__(
        self,
        store: PaperStore,
        datasets: list[Dataset],
        rule: TrendLeverage,
        rates: pd.Series,
        cost_bps: float = DEFAULT_COST_BPS,
        log: Log = print,
    ) -> None:
        super().__init__(store, datasets, cost_bps=cost_bps, log=log)
        self.rule = rule
        self.rates = rates.sort_index()
        # Moving averages use only closes up to each day (rolling is causal).
        self.averages = {t: b["close"].rolling(rule.sma, min_periods=rule.sma).mean()
                         for t, b in self.bars.items()}

    def describe(self) -> str:
        return (f"hold each ETF at {self.rule.leverage}x while it closes above its "
                f"{self.rule.sma}-day average; cash below (earns T-bills, borrowing "
                f"costs T-bills + {self.rule.borrow_spread:.0%})")

    # --- setup ----------------------------------------------------------------
    def init(self, capital: float, start: str) -> None:
        if self.store.initialised:
            raise RuntimeError(f"{self.store.path} already holds an account")
        before = self.calendar[self.calendar < pd.Timestamp(start)]
        if before.empty:
            raise ValueError("Start date is before the available data")
        anchor = before[-1]
        s = self.store
        s.set("strategy", "trend")
        s.set("rule", {"sma": self.rule.sma, "leverage": self.rule.leverage,
                       "borrow_spread": self.rule.borrow_spread})
        s.set("initial_capital", capital)
        s.cash = capital
        s.set("financing_total", 0.0)
        s.set("tickers", [ds.ticker for ds in self.datasets])
        s.set("start", _day(anchor))
        s.set("last_processed", _day(anchor))
        s.record_equity(_day(anchor), capital, 0.0)
        self._decide(anchor)
        s.commit()
        self.log(f"Trend account created at {_day(anchor)}: {self.describe()}")

    # --- hooks ------------------------------------------------------------------
    def _relearn_due(self, d: pd.Timestamp) -> bool:
        return False  # a fixed rule: nothing to re-learn

    def _strategy_id(self) -> int:
        return 0

    def _accrue(self, d: pd.Timestamp) -> None:
        """One trading day of interest: earned on cash, paid on borrowing."""
        s = self.store
        rate = self.rates.loc[:d]
        rate = max(float(rate.iloc[-1]), 0.0) if len(rate) else 0.0
        cash = s.cash
        yearly = rate if cash >= 0 else rate + self.rule.borrow_spread
        interest = cash * yearly / TRADING_DAYS
        s.cash = cash + interest
        s.set("financing_total", s.get("financing_total", 0.0) + interest)

    def _decide(self, d: pd.Timestamp) -> None:
        s = self.store
        positions = s.positions()
        pending = {o.ticker for o in s.pending_orders()}
        equity = s.cash + self.holdings_value(d)
        slot = equity * self.rule.leverage / len(self.bars)

        for ticker, bars in self.bars.items():
            if d not in bars.index or ticker in pending:
                continue
            close = float(bars.at[d, "close"])
            average = self.averages[ticker].get(d, np.nan)
            want = bool(close > average)  # NaN (not enough history) -> False
            have = ticker in positions
            if want == have:
                continue
            side = "above" if want else "below"
            reason = (f"trend rule on {_day(d)}: close {close:.2f} is {side} its "
                      f"{self.rule.sma}-day average {average:.2f}")
            if want:
                s.add_order(_day(d), ticker, "buy", slot / close, reason)
            else:
                s.add_order(_day(d), ticker, "sell", positions[ticker].qty, reason)
