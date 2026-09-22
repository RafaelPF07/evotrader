"""Plain-text views of the paper account for the CLI."""

from __future__ import annotations

import pandas as pd

from evotrader.metrics import max_drawdown, sharpe
from evotrader.paper.store import PaperStore


def benchmark_return(bars: dict[str, pd.DataFrame], start: str, end: str) -> float:
    """Equal-weight buy & hold of the same universe over the same dates."""
    rets = [
        float(b["close"].loc[:end].iloc[-1] / b["close"].loc[:start].iloc[-1] - 1)
        for b in bars.values()
    ]
    return sum(rets) / len(rets)


def equity_frame(store: PaperStore, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily account equity next to an equal-weight buy & hold of the same universe,
    both starting from the same capital. Used by the charts and the dashboard."""
    eq = store.frame("SELECT date, equity FROM equity ORDER BY date")
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date")["equity"]
    start = eq.index[0]
    growth = pd.concat(
        [b["close"].loc[start:eq.index[-1]] / b["close"].loc[:start].iloc[-1]
         for b in bars.values()], axis=1
    ).ffill().mean(axis=1)
    benchmark = growth.reindex(eq.index).ffill() * store.get("initial_capital")
    return pd.DataFrame({"equity": eq, "benchmark": benchmark})


def status_text(store: PaperStore, bars: dict[str, pd.DataFrame]) -> str:
    eq = store.frame("SELECT date, equity FROM equity ORDER BY date").set_index("date")["equity"]
    capital = store.get("initial_capital")
    start, last = store.get("start"), store.get("last_processed")
    trades = store.frame("SELECT ticker, return FROM trades")
    daily = eq.pct_change().dropna()

    lines = [
        f"Paper account  {start} -> {last}  ({len(eq) - 1} trading days)",
        "",
        f"  equity          ${eq.iloc[-1]:,.2f}   (started ${capital:,.0f})",
        f"  return          {eq.iloc[-1] / capital - 1:+.2%}",
        f"  benchmark       {benchmark_return(bars, start, last):+.2%}   "
        "(equal-weight buy & hold, same ETFs)",
        f"  max drawdown    {max_drawdown(daily):.2%}",
        f"  sharpe          {sharpe(daily):.2f}" if len(daily) > 20 else "  sharpe          -",
        f"  cash            ${store.cash:,.2f}",
        f"  closed trades   {len(trades)}"
        + (f"   win rate {(trades['return'] > 0).mean():.0%}"
           f"   avg {trades['return'].mean():+.2%}" if len(trades) else ""),
        "",
    ]

    sid, genome = store.active_strategy()
    origin = store.frame("SELECT origin, created FROM strategies WHERE id = ?", (sid,)).iloc[0]
    objective = {"excess": "beat buy & hold (information ratio)",
                 "sharpe": "own Sharpe ratio"}[store.get("objective", "sharpe")]
    lines += [f"Active strategy #{sid} ({origin['origin']}, since {origin['created']}):",
              f"  {genome}", f"  learning objective: {objective}", ""]

    positions = store.positions()
    if positions:
        lines.append("Open positions:")
        for t, p in positions.items():
            price = float(bars[t]["close"].loc[:last].iloc[-1])
            lines.append(f"  {t:4} {p.qty:10.3f} @ {p.avg_price:8.2f}  now {price:8.2f}  "
                         f"{price / p.avg_price - 1:+.2%}  since {p.entry_date}")
        lines.append("")

    pending = store.pending_orders()
    if pending:
        lines.append("Orders for next open:")
        lines += [f"  {o.side.upper():4} {o.ticker:4} {o.qty:10.3f}" for o in pending]
        lines.append("")

    log = store.frame("SELECT date, promoted, notes FROM learning_log ORDER BY id DESC LIMIT 3")
    if len(log):
        lines.append("Recent learning:")
        for _, row in log.iterrows():
            lines.append(f"  {row['date']}")
            lines += [f"    {line}" for line in row["notes"].splitlines()]
    return "\n".join(lines)


def journal_text(store: PaperStore, n: int = 10) -> str:
    trades = store.frame(f"SELECT * FROM trades ORDER BY id DESC LIMIT {int(n)}")
    if trades.empty:
        return "No closed trades yet."
    out = []
    for _, t in trades.iterrows():
        verdict = "WIN " if t["return"] > 0 else "LOSS"
        out += [
            f"{verdict} {t['ticker']:4} {t['entry_date']} -> {t['exit_date']}  "
            f"{t['return']:+.2%}  (${t['pnl']:+,.2f})",
            f"  why in : {t['entry_reason']}",
            f"  why out: {t['exit_reason']}",
            "",
        ]
    return "\n".join(out)
