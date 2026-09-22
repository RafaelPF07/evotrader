"""Streamlit dashboard for the paper account and the research results.

    evotrader dashboard      (or: streamlit run src/evotrader/dashboard.py)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from evotrader.metrics import max_drawdown, sharpe
from evotrader.paper.report import equity_frame
from evotrader.paper.store import PaperStore
from evotrader.walkforward import load_datasets

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

# Same validated categorical order as the static charts; dark steps for dark mode.
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "data" / "paper.db"))
    return parser.parse_known_args(sys.argv[1:])[0]


def _colors() -> list[str]:
    theme = getattr(getattr(st, "context", None), "theme", None)
    return PALETTE["dark" if getattr(theme, "type", "light") == "dark" else "light"]


@st.cache_data(ttl=3600)
def _bars(tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    return {ds.ticker: ds.bars for ds in load_datasets(list(tickers), use_ml=False,
                                                          log=lambda _: None)}


def line_chart(series: dict[str, pd.Series], colors: list[str], pct: bool = True,
               vlines: list[pd.Timestamp] = (), height: int = 380) -> go.Figure:
    fig = go.Figure()
    for (name, s), color in zip(series.items(), colors, strict=False):
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode="lines", line={"color": color, "width": 2},
            hovertemplate=f"{name}: %{{y:{'+.1%' if pct else '.3f'}}}<extra></extra>",
        ))
    for d in vlines:
        fig.add_vline(x=d, line_width=1, line_color="rgba(128,128,128,0.35)")
    fig.update_layout(
        height=height, hovermode="x unified", margin={"l": 10, "r": 10, "t": 10, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        yaxis={"tickformat": "+.0%" if pct else ".2f"},
    )
    return fig


def paper_tab(store: PaperStore, bars: dict[str, pd.DataFrame], colors: list[str]) -> None:
    frame = equity_frame(store, bars)
    capital = store.get("initial_capital")
    growth = frame / capital - 1
    daily = frame["equity"].pct_change().dropna()
    trades = store.frame("SELECT * FROM trades ORDER BY id DESC")
    relearns = pd.to_datetime(store.frame("SELECT date FROM learning_log")["date"]).tolist()

    c = st.columns(5)
    c[0].metric("Equity", f"${frame['equity'].iloc[-1]:,.0f}")
    c[1].metric("Return", f"{growth['equity'].iloc[-1]:+.1%}",
                f"{growth['equity'].iloc[-1] - growth['benchmark'].iloc[-1]:+.1%} vs buy & hold")
    c[2].metric("Max drawdown", f"{max_drawdown(daily):.1%}")
    c[3].metric("Sharpe", f"{sharpe(daily):.2f}")
    c[4].metric("Closed trades", len(trades),
                f"{(trades['return'] > 0).mean():.0%} winners" if len(trades) else None,
                delta_color="off")

    st.subheader("Growth vs buy & hold")
    st.caption("Grey lines mark re-learning checks. Hover for daily values.")
    st.plotly_chart(line_chart({"Paper account": growth["equity"],
                                "Buy & hold (equal-weight)": growth["benchmark"]}, colors,
                               vlines=relearns), use_container_width=True)

    st.subheader("Drawdown")
    dd = frame / frame.cummax() - 1
    st.plotly_chart(line_chart({"Paper account": dd["equity"],
                                "Buy & hold (equal-weight)": dd["benchmark"]}, colors,
                               height=220), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Open positions")
        last = pd.Timestamp(store.get("last_processed"))
        rows = [{"ticker": t, "qty": round(p.qty, 3), "entry": p.entry_date,
                 "avg price": round(p.avg_price, 2),
                 "P&L": f"{float(bars[t]['close'].loc[:last].iloc[-1]) / p.avg_price - 1:+.2%}"}
                for t, p in store.positions().items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Orders for next open")
        pending = store.frame("SELECT created, side, ticker, qty FROM orders "
                              "WHERE status = 'pending'")
        st.dataframe(pending, hide_index=True, use_container_width=True)
    st.caption(f"Last processed trading day: {store.get('last_processed')}")


def journal_tab(store: PaperStore) -> None:
    trades = store.frame("SELECT * FROM trades ORDER BY exit_date DESC")
    if trades.empty:
        st.info("No closed trades yet.")
        return
    f1, f2 = st.columns(2)
    tickers = f1.multiselect("Ticker", sorted(trades["ticker"].unique()))
    outcome = f2.radio("Outcome", ["All", "Wins", "Losses"], horizontal=True)
    if tickers:
        trades = trades[trades["ticker"].isin(tickers)]
    if outcome != "All":
        trades = trades[(trades["return"] > 0) == (outcome == "Wins")]

    table = trades[["ticker", "entry_date", "exit_date", "return", "pnl"]].copy()
    table["return"] = table["return"].map("{:+.2%}".format)
    table["pnl"] = table["pnl"].map("${:+,.2f}".format)
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.subheader("Why each trade happened")
    for _, t in trades.head(25).iterrows():
        label = "WIN" if t["return"] > 0 else "LOSS"
        with st.expander(f"{label} {t['ticker']}  {t['entry_date']} -> {t['exit_date']}  "
                         f"{t['return']:+.2%}"):
            st.markdown(f"**Why in:** `{t['entry_reason'].split(': ', 1)[-1]}`")
            st.markdown(f"**Why out:** `{t['exit_reason'].split(': ', 1)[-1]}`")


def learning_tab(store: PaperStore) -> None:
    sid, genome = store.active_strategy()
    st.subheader("Active strategy")
    st.code(str(genome), language=None)
    st.subheader("Strategy history")
    st.dataframe(store.frame("SELECT id, created, origin, active, rule FROM strategies"),
                 hide_index=True, use_container_width=True)
    st.subheader("Re-learning decisions")
    st.caption("A challenger is promoted only if it beats the champion on the most recent "
               "year, which neither was evolved on, by a margin.")
    log = store.frame("SELECT * FROM learning_log ORDER BY id DESC")
    for _, row in log.iterrows():
        verdict = "Promoted" if row["promoted"] else "Kept champion"
        with st.expander(f"{row['date']}: {verdict}  (incumbent {row['incumbent_score']:+.3f}"
                         f" vs challenger {row['challenger_score']:+.3f})"):
            st.text(row["notes"])
            st.caption(f"Challenger: {row['challenger_rule']}")


def research_tab(colors: list[str]) -> None:
    returns_path = REPORTS / "walkforward_returns.csv"
    if not returns_path.exists():
        st.info("Run `evotrader walkforward` first.")
        return
    rets = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    st.subheader("Out-of-sample growth, 2016 - today")
    st.caption("Every point is from a period the strategy had never seen. "
               "Equal-weight portfolio across the ETF universe, costs included.")
    names = {"evolved": "Evolved bot", "buy_and_hold": "Buy & hold", "sma_cross": "SMA 50/200",
             "momentum": "Momentum", "ml_signal": "ML only"}
    growth = {names[c]: (1 + rets[c]).cumprod() - 1 for c in names if c in rets}
    st.plotly_chart(line_chart(growth, colors), use_container_width=True)

    seeds_path = REPORTS / "walkforward_seeds.csv"
    if seeds_path.exists():
        st.subheader("Robustness across random seeds")
        seeds = pd.read_csv(seeds_path)
        st.dataframe(seeds.style.format({"cagr": "{:+.1%}", "sharpe": "{:.2f}",
                                         "max_drawdown": "{:.1%}", "exposure": "{:.0%}"}),
                     hide_index=True, use_container_width=True)

    hist_path = REPORTS / "walkforward_history.csv"
    if hist_path.exists():
        st.subheader("Learning curves")
        hist = pd.read_csv(hist_path)
        if "seed" in hist:
            hist = hist[hist["seed"] == hist["seed"].min()]
        curves = {f"fold {f} (test from {2016 + 2 * f})":
                  hist[hist["fold"] == f].set_index("generation")["best_fitness"]
                  for f in sorted(hist["fold"].unique())}
        st.plotly_chart(line_chart(curves, colors, pct=False, height=320),
                        use_container_width=True)

    report = REPORTS / "walkforward.md"
    if report.exists():
        with st.expander("Full walk-forward report"):
            st.markdown(report.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="EvoTrader", layout="wide")
    st.title("EvoTrader")
    st.caption("A self-improving paper trading bot. No real money involved.")
    colors = _colors()
    db = Path(_args().db)

    tabs = st.tabs(["Paper account", "Trade journal", "Learning", "Research"])
    if not db.exists():
        for tab in tabs[:3]:
            tab.info("No paper account yet: run `evotrader paper init` then `evotrader paper run`.")
    else:
        store = PaperStore(db)
        bars = _bars(tuple(store.get("tickers")))
        with tabs[0]:
            paper_tab(store, bars, colors)
        with tabs[1]:
            journal_tab(store)
        with tabs[2]:
            learning_tab(store)
    with tabs[3]:
        research_tab(colors)


main()
