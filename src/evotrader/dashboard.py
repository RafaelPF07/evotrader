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


def _is_dark() -> bool:
    theme = getattr(getattr(st, "context", None), "theme", None)
    return getattr(theme, "type", "light") == "dark"


def _colors() -> list[str]:
    return PALETTE["dark" if _is_dark() else "light"]


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


EVOLUTION = REPORTS / "evolution"
GOALS = {"excess": "Beat buy & hold (live bot)", "sharpe": "Highest Sharpe ratio (original)"}
# Sequential blue ramp for the gene-pool heatmap; in dark mode "none" recedes into the surface.
RAMP = {"light": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
        "dark": ["#1a2a40", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb"]}
GREY = "rgba(150,150,145,0.75)"
BIRTH_LABELS = {"offspring": "offspring (crossover / mutation)",
                "survivor": "survivor (elite, unchanged)",
                "newcomer": "newcomer (random / immigrant)"}


@st.cache_data
def _evolution_log(path: str) -> tuple[dict, pd.DataFrame]:
    from evotrader.evolution import history as h

    return h.load(Path(path))


def _wrap(rule: str, width: int = 60) -> str:
    import textwrap

    return "<br>".join(textwrap.wrap(rule, width))


def swarm_chart(df: pd.DataFrame, colors: list[str]) -> go.Figure:
    """Animated scatter: one frame per generation, one dot per strategy."""
    lo, hi = df["fitness"].quantile(0.05), df["fitness"].max()
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    groups = {"offspring": colors[0], "survivor": colors[1], "newcomer": colors[2]}
    gens = sorted(df["generation"].unique())

    def traces(gen: int) -> list[go.Scatter]:
        now = df[df["generation"] == gen]
        out = []
        for group, color in groups.items():
            pts = now[now["group"] == group]
            out.append(go.Scatter(
                x=pts["exposure"], y=pts["fitness"].clip(lower=lo + (hi - lo) * 0.01),
                mode="markers", name=BIRTH_LABELS[group],
                marker={"color": color, "size": 11 if group == "survivor" else 8,
                        "line": {"width": 1, "color": "rgba(255,255,255,0.7)"}},
                customdata=list(zip(pts["origin"], pts["fitness"], pts["trades_per_year"],
                                    pts["rule"].map(_wrap), strict=True)),
                hovertemplate=("<b>%{customdata[0]}</b><br>fitness %{customdata[1]:.3f}"
                               " | invested %{x:.0%} | %{customdata[2]:.1f} trades/yr"
                               "<br>%{customdata[3]}<extra></extra>"),
            ))
        return out

    play = {"frame": {"duration": 450, "redraw": True}, "transition": {"duration": 250},
            "fromcurrent": True}
    pause = {"frame": {"duration": 0}, "mode": "immediate"}
    fig = go.Figure(data=traces(gens[0]),
                    frames=[go.Frame(data=traces(g), name=str(g)) for g in gens])
    fig.update_layout(
        height=560, margin={"l": 10, "r": 10, "t": 30, "b": 130},
        xaxis={"title": "share of time invested", "range": [-0.03, 1.03], "tickformat": ".0%"},
        yaxis={"title": "training fitness", "range": [lo, hi]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        # Controls live below the axis title: buttons on their own row, slider underneath.
        updatemenus=[{"type": "buttons", "direction": "left", "x": 0, "y": -0.2,
                      "xanchor": "left", "yanchor": "top", "showactive": False,
                      "buttons": [{"label": "Play", "method": "animate", "args": [None, play]},
                                  {"label": "Pause", "method": "animate",
                                   "args": [[None], pause]}]}],
        sliders=[{"x": 0, "len": 1, "y": -0.3, "yanchor": "top", "pad": {"t": 10},
                  "currentvalue": {"prefix": "generation ", "xanchor": "right"},
                  "steps": [{"label": str(g), "method": "animate",
                             "args": [[str(g)], {"frame": {"duration": 0, "redraw": True},
                                                 "mode": "immediate"}]} for g in gens]}],
    )
    return fig


def family_tree_figure(df: pd.DataFrame, colors: list[str]) -> go.Figure:
    """Every ancestor of the champion, by generation and fitness, with its direct line."""
    from evotrader.evolution import history as h

    champ = h.champion(df)
    tree = h.ancestry(df, champ["id"]).set_index("id")
    line = h.main_line(df, champ["id"])
    steps = h.story(df, champ["id"])

    edges: dict[int, tuple[list, list]] = {0: ([], []), 1: ([], [])}
    for _, child in tree.iterrows():
        for k, pid in enumerate(child["parents"][:2]):
            parent = tree.loc[pid]
            xs, ys = edges[k]
            xs += [parent["generation"], child["generation"], None]
            ys += [parent["fitness"], child["fitness"], None]
    fig = go.Figure()
    for k, (xs, ys) in edges.items():
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", hoverinfo="skip",
            name="parent -> child" if k == 0 else "genes donated by crossover",
            line={"color": GREY, "width": 1, "dash": "solid" if k == 0 else "dot"}))
    hover = ("<b>%{customdata[0]}</b> (gen %{x})<br>fitness %{y:.3f}<br>%{customdata[1]}"
             "<extra></extra>")
    fig.add_trace(go.Scatter(
        x=tree["generation"], y=tree["fitness"], mode="markers", name="ancestor",
        marker={"color": GREY, "size": 7},
        customdata=list(zip(tree["origin"], tree["rule"].map(_wrap), strict=True)),
        hovertemplate=hover))
    fig.add_trace(go.Scatter(
        x=line["generation"], y=line["fitness"], mode="lines+markers",
        name="direct line of descent", line={"color": colors[0], "width": 3},
        marker={"size": 8},
        customdata=list(zip(line["origin"], line["rule"].map(_wrap), strict=True)),
        hovertemplate=hover))
    fig.add_trace(go.Scatter(
        x=[champ["generation"]], y=[champ["fitness"]], mode="markers", name="champion",
        marker={"color": colors[0], "size": 20, "symbol": "star"}, hoverinfo="skip"))
    best = steps[steps["fitness"] > steps["fitness"].cummax().shift(fill_value=float("-inf"))]
    for i, (_, step) in enumerate(best.iterrows()):  # alternate sides so labels don't collide
        how = step["how"].replace(" (entry)", "").replace(" (exit)", "")
        fig.add_annotation(x=step["generation"], y=step["fitness"], text=how,
                           showarrow=False, yshift=18 if i % 2 == 0 else -18, font={"size": 11})
    span = line["fitness"].max() - line["fitness"].min()
    fig.update_layout(
        height=480, margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"title": "generation", "dtick": 3},
        yaxis={"title": "training fitness", "range": [line["fitness"].min() - 0.35 * span,
                                                       line["fitness"].max() + 0.35 * span]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def gene_pool_figure(df: pd.DataFrame) -> go.Figure:
    """Heatmap: share of the population using each indicator, per generation."""
    from evotrader.evolution import history as h

    pool = h.gene_pool(df)
    ramp = RAMP["dark" if _is_dark() else "light"]
    fig = go.Figure(go.Heatmap(
        z=pool.values, x=pool.columns, y=pool.index, zmin=0, zmax=1,
        colorscale=[[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)],
        xgap=2, ygap=2, colorbar={"title": "share", "tickformat": ".0%"},
        hovertemplate="%{y} used by %{z:.0%} of generation %{x}<extra></extra>"))
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      xaxis={"title": "generation", "dtick": 3})
    return fig


def evolution_tab(colors: list[str]) -> None:
    from evotrader.evolution import history as h

    logs = {obj: EVOLUTION / f"{obj}_seed0.json" for obj in GOALS}
    logs = {k: v for k, v in logs.items() if v.exists()}
    if not logs:
        st.info("Run `evotrader evolution` to record an evolution run first.")
        return
    goal = st.radio("Evolution goal", list(logs), format_func=GOALS.get, horizontal=True)
    meta, df = _evolution_log(str(logs[goal]))
    champ = h.champion(df)

    st.caption(
        f"One genetic-algorithm run on {meta['start']} to {meta['end']} "
        f"({len(meta['tickers'])} ETFs, seed {meta['seed']}). Fitness here is **training** "
        "fitness, i.e. how the algorithm judges its population, not out-of-sample performance.")
    c = st.columns(4)
    c[0].metric("Strategies created", f"{len(df):,}")
    c[1].metric("Generations", int(df["generation"].max()) + 1)
    c[2].metric("Crossovers", int(df["origin"].str.startswith("crossover").sum()))
    c[3].metric("Champion fitness", f"{champ['fitness']:+.3f}")

    st.subheader("The population swarm")
    st.caption("Each dot is one strategy. Press Play: the random first generation converges "
               "as selection, crossover and mutation do their work. Switch goals above: the "
               "same algorithm evolves a different kind of strategy for each.")
    st.plotly_chart(swarm_chart(df, colors), use_container_width=True)

    st.subheader("How the champion was born")
    st.caption("Every ancestor of the final champion. The blue line is its direct line of "
               "descent; labels mark each new best. Hover any point for its rule.")
    st.plotly_chart(family_tree_figure(df, colors), use_container_width=True)
    steps = h.story(df, champ["id"])
    table = steps.assign(
        generations=[str(a) if a == b else f"{a}-{b}"
                     for a, b in zip(steps["generation"], steps["survived_until"], strict=True)],
        fitness=steps["fitness"].map("{:+.3f}".format),
        change=steps["change"].map(lambda v: "" if pd.isna(v) else f"{v:+.3f}"),
    )[["generations", "how", "fitness", "change", "rule"]]
    st.dataframe(table, hide_index=True, use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("The gene pool")
        st.caption("Share of the population using each indicator, generation by generation. "
                   "Useful building blocks spread; useless ones die out.")
        st.plotly_chart(gene_pool_figure(df), use_container_width=True)
    with right:
        st.subheader("The champion's rule")
        st.code(h.rule_tree(champ["genome"]), language=None)


def main() -> None:
    st.set_page_config(page_title="EvoTrader", layout="wide")
    st.title("EvoTrader")
    st.caption("A self-improving paper trading bot. No real money involved.")
    colors = _colors()
    db = Path(_args().db)

    tabs = st.tabs(["Paper account", "Trade journal", "Learning", "Evolution", "Research"])
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
        evolution_tab(colors)
    with tabs[4]:
        research_tab(colors)


main()
