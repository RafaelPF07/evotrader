"""The daily paper-trading loop.

Each trading day d, in the same order the backtester assumes:

    1. fill     orders decided yesterday execute at today's open, with slippage
    2. mark     value the account at today's close
    3. learn    every N days, try to evolve a better strategy (see learner.py)
    4. decide   the active strategy looks at bars up to today's close and queues
                orders for tomorrow, each with a human-readable reason

`run()` processes every trading day since the last run, so the same code works
for a daily scheduled job and for replaying the bot through past months.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from evotrader.backtest import DEFAULT_COST_BPS
from evotrader.evolution.fitness import Dataset
from evotrader.evolution.genome import Genome
from evotrader.paper.learner import LearnerConfig, relearn, train_initial
from evotrader.paper.store import PaperStore, Position

Log = Callable[[str], None]


def _day(ts: pd.Timestamp) -> str:
    return ts.date().isoformat()


class PaperTrader:
    def __init__(
        self,
        store: PaperStore,
        datasets: list[Dataset],
        learner: LearnerConfig | None = None,
        cost_bps: float = DEFAULT_COST_BPS,
        log: Log = print,
    ) -> None:
        self.store = store
        self.datasets = datasets
        self.bars = {ds.ticker: ds.bars for ds in datasets}
        self.stores = {ds.ticker: ds.store for ds in datasets}
        self._target_cache: dict[tuple[int, str], np.ndarray] = {}
        self.calendar = datasets[0].bars.index
        self.learner = learner or LearnerConfig()
        self.cost = cost_bps / 10_000
        self.log = log

    # --- setup ----------------------------------------------------------------
    def init(self, capital: float, start: str) -> None:
        if self.store.initialised:
            raise RuntimeError(f"{self.store.path} already holds an account")
        before = self.calendar[self.calendar < pd.Timestamp(start)]
        if before.empty:
            raise ValueError("Start date is before the available data")
        anchor = before[-1]  # the bot "wakes up" at this close, knowing nothing later
        self.log(f"Training the first champion on data up to {_day(anchor)}")
        champion, hall = train_initial(self.datasets, anchor, self.learner, self.log)

        s = self.store
        s.set("initial_capital", capital)
        s.cash = capital
        s.set("tickers", [ds.ticker for ds in self.datasets])
        s.set("start", _day(anchor))
        s.set("last_processed", _day(anchor))
        s.set("last_relearn", _day(anchor))
        s.set("relearn_count", 0)
        s.set("objective", self.learner.fitness.objective)
        s.set_hall_of_fame(hall)
        s.activate_strategy(champion, _day(anchor), "initial")
        s.record_equity(_day(anchor), capital, 0.0)
        self._decide(anchor)
        s.commit()
        self.log(f"Champion: {champion}")

    # --- main loop ------------------------------------------------------------
    def run(self, until: pd.Timestamp | None = None) -> int:
        last = pd.Timestamp(self.store.get("last_processed"))
        days = self.calendar[self.calendar > last]
        if until is not None:
            days = days[days <= until]
        for d in days:
            self.step(d)
        return len(days)

    def step(self, d: pd.Timestamp) -> None:
        self._fill(d)
        self._mark(d)
        if self._relearn_due(d):
            self._relearn(d)
        self._decide(d)
        self.store.set("last_processed", _day(d))
        self.store.commit()

    # --- 1. fill --------------------------------------------------------------
    def _fill(self, d: pd.Timestamp) -> None:
        s = self.store
        positions = s.positions()
        for order in s.pending_orders():
            bars = self.bars[order.ticker]
            if d not in bars.index:  # ticker didn't trade today; try again tomorrow
                continue
            open_ = float(bars.at[d, "open"])
            if order.side == "buy":
                price = open_ * (1 + self.cost)
                qty = min(order.qty, s.cash / price)
                s.cash = s.cash - qty * price
                s.upsert_position(Position(order.ticker, qty, price, _day(d), order.reason))
            else:
                pos = positions[order.ticker]
                qty, price = pos.qty, open_ * (1 - self.cost)
                s.cash = s.cash + qty * price
                s.add_trade(
                    ticker=order.ticker, strategy_id=s.active_strategy()[0],
                    entry_date=pos.entry_date, entry_price=pos.avg_price, exit_date=_day(d),
                    exit_price=price, qty=qty, pnl=qty * (price - pos.avg_price),
                    **{"return": price / pos.avg_price - 1},
                    entry_reason=pos.entry_reason, exit_reason=order.reason,
                )
                s.delete_position(order.ticker)
            s.mark_filled(order.id, _day(d), price, qty * open_ * self.cost)
            side = order.side.upper()
            self.log(f"{_day(d)}  {side:4} {order.ticker:4} {qty:10.3f} @ {price:.2f}")

    # --- 2. mark --------------------------------------------------------------
    def holdings_value(self, d: pd.Timestamp) -> float:
        total = 0.0
        for ticker, pos in self.store.positions().items():
            closes = self.bars[ticker]["close"].loc[:d]
            total += pos.qty * float(closes.iloc[-1])
        return total

    def _mark(self, d: pd.Timestamp) -> None:
        self.store.record_equity(_day(d), self.store.cash, self.holdings_value(d))

    # --- 3. learn -------------------------------------------------------------
    def _relearn_due(self, d: pd.Timestamp) -> bool:
        last = pd.Timestamp(self.store.get("last_relearn"))
        elapsed = ((self.calendar > last) & (self.calendar <= d)).sum()
        return elapsed >= self.learner.relearn_every

    def _relearn(self, d: pd.Timestamp) -> None:
        s = self.store
        _, incumbent = s.active_strategy()
        count = s.get("relearn_count", 0) + 1
        self.log(f"{_day(d)}  re-learning (#{count})")
        decision = relearn(self.datasets, d, incumbent, s.hall_of_fame(), self.learner,
                           seed=count, log=self.log)
        s.log_learning(_day(d), decision.incumbent_score.fitness,
                       decision.challenger_score.fitness, str(decision.challenger),
                       decision.promoted, decision.notes)
        s.set_hall_of_fame(decision.hall_of_fame)
        if decision.promoted:
            s.activate_strategy(decision.challenger, _day(d), f"relearn #{count}")
        self.log(f"  {decision.notes.splitlines()[0]}")
        s.set("last_relearn", _day(d))
        s.set("relearn_count", count)

    # --- 4. decide ------------------------------------------------------------
    def _decide(self, d: pd.Timestamp) -> None:
        s = self.store
        strategy_id, genome = s.active_strategy()
        positions = s.positions()
        pending = {o.ticker for o in s.pending_orders()}
        equity = s.cash + self.holdings_value(d)
        slot = equity / len(self.bars)  # equal weight per ticker

        for ticker, bars in self.bars.items():
            if d not in bars.index or ticker in pending:
                continue
            i = bars.index.get_loc(d)
            want = self._targets(strategy_id, genome, ticker)[i] > 0
            have = ticker in positions
            if want == have:
                continue
            explanation = genome.explain(self.stores[ticker], i)
            reason = f"strategy #{strategy_id} on {_day(d)}: {explanation}"
            if want:
                s.add_order(_day(d), ticker, "buy", slot / float(bars.at[d, "close"]), reason)
            else:
                s.add_order(_day(d), ticker, "sell", positions[ticker].qty, reason)

    def _targets(self, strategy_id: int, genome: Genome, ticker: str) -> np.ndarray:
        """Target positions for the whole series, computed once per strategy and ticker.

        Every feature is causal (value at i uses bars <= i only, enforced by the
        no-look-ahead tests), so reading element i equals recomputing on bars[:i+1].
        """
        key = (strategy_id, ticker)
        if key not in self._target_cache:
            self._target_cache[key] = genome.target_position(self.stores[ticker])
        return self._target_cache[key]
