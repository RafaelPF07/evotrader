"""Static PNG charts for the README, regenerated from saved results.

    evotrader charts        -> docs/img/*.png

Colours follow a colour-blind-validated categorical palette, used in a fixed
order: the bot is always blue, buy & hold always orange. Every chart with two
or more series has a legend *and* direct labels, so identity never relies on
colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed order, never cycled
BOT, BENCHMARK = SERIES[0], SERIES[1]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titlesize": 13, "axes.titleweight": "bold",
    "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
})

PCT = FuncFormatter(lambda v, _: f"{v:+.0%}" if v else "0%")


def _label_ends(ax, items: list[tuple[pd.Series, str, str]], min_gap: float = 0.045) -> None:
    """Direct labels just right of each line's last point, outside the plot area.

    Labels are sorted by their line's end value and pushed apart so they never
    overlap or swap order: each label stays level with (or nearest to) its own line.
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap
    ends = sorted(((s.iloc[-1], s, text, color) for s, text, color in items), key=lambda t: t[0])
    placed: list[float] = []
    for y, s, text, color in ends:
        target = max(y, placed[-1] + gap) if placed else y
        placed.append(target)
        ax.plot([s.index[-1]], [y], "o", color=color, markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=5, clip_on=False)
        ax.annotate(text, (s.index[-1], target), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT, annotation_clip=False)


def _drawdown(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def walkforward_chart(reports: Path, out: Path) -> Path:
    rets = pd.read_csv(reports / "walkforward_returns.csv", index_col=0, parse_dates=True)
    seeds_path = reports / "walkforward_evolved_by_seed.csv"
    seeds = (pd.read_csv(seeds_path, index_col=0, parse_dates=True)
             if seeds_path.exists() else None)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    if seeds is not None:
        for col in seeds.columns[1:]:
            top.plot((1 + seeds[col]).cumprod() - 1, color=BOT, linewidth=1, alpha=0.3)
    bot = (1 + rets["evolved"]).cumprod() - 1
    bh = (1 + rets["buy_and_hold"]).cumprod() - 1
    top.plot(bh, color=BENCHMARK, label="Buy & hold (equal-weight)")
    top.plot(bot, color=BOT, label="Evolved bot" + (" (thin: other seeds)" if seeds is not None
                                                     else ""))
    top.set_title("Out-of-sample growth, 2016 - today (never seen during learning)")
    top.yaxis.set_major_formatter(PCT)
    top.legend(loc="upper left")

    dd_bot, dd_bh = _drawdown(rets["evolved"]), _drawdown(rets["buy_and_hold"])
    bottom.plot(dd_bh, color=BENCHMARK, linewidth=1.5)
    bottom.plot(dd_bot, color=BOT, linewidth=1.5)
    bottom.set_title("Drawdown from peak", fontsize=11)
    bottom.yaxis.set_major_formatter(PCT)
    bottom.annotate(f"worst: bot (bold line) {dd_bot.min():.0%}, buy & hold {dd_bh.min():.0%}",
                    (0.01, 0.08), xycoords="axes fraction", fontsize=9, color=TEXT_2)
    for ax in (top, bottom):
        ax.margins(x=0.0)
    _label_ends(top, [(bh, f"Buy & hold {bh.iloc[-1]:+.0%}", BENCHMARK),
                      (bot, f"Bot {bot.iloc[-1]:+.0%}", BOT)])
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    path = out / "walkforward_equity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def learning_curve_chart(reports: Path, out: Path) -> Path:
    hist = pd.read_csv(reports / "walkforward_history.csv")
    if "seed" in hist:
        hist = hist[hist["seed"] == hist["seed"].min()]
    folds = sorted(hist["fold"].unique())
    first_test_year = 2016

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    for i, fold in enumerate(folds):
        h = hist[hist["fold"] == fold].set_index("generation")["best_fitness"]
        year = first_test_year + 2 * fold
        ax.plot(h, color=SERIES[i], label=f"before {year} test", marker="o", markersize=3.5)
        labels.append((h, f"before {year}", SERIES[i]))
    ax.set_title("The bot learning: best fitness per generation, one line per walk-forward fold",
                 pad=24)
    ax.annotate("Later folds start high: each is seeded with the previous fold's best rules.",
                (0, 1.02), xycoords="axes fraction", fontsize=9, color=TEXT_2)
    ax.set_xlabel("generation")
    ax.set_ylabel("training fitness (Sharpe-based)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=0.01)
    ax.legend(loc="lower right", ncols=len(folds), fontsize=8.5)
    _label_ends(ax, labels)
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    path = out / "learning_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def paper_chart(frame: pd.DataFrame, relearn_dates: list[pd.Timestamp], out: Path) -> Path:
    capital = frame["equity"].iloc[0]
    growth = frame / capital - 1

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for d in relearn_dates:
        ax.axvline(d, color=GRID, linewidth=1.2, zorder=0)
    ax.plot(growth["benchmark"], color=BENCHMARK, label="Buy & hold (equal-weight)")
    ax.plot(growth["equity"], color=BOT, label="Paper account")
    if relearn_dates:
        ax.annotate("grey lines = re-learning checks", (0.01, 0.04), xycoords="axes fraction",
                    fontsize=9, color=TEXT_2)
    ax.set_title(f"Paper account since {frame.index[0].date()} (daily replay, then live)")
    ax.yaxis.set_major_formatter(PCT)
    ax.margins(x=0)
    ax.legend(loc="upper left")
    _label_ends(ax, [(growth["benchmark"], f"Buy & hold {growth['benchmark'].iloc[-1]:+.0%}",
                      BENCHMARK),
                     (growth["equity"], f"Bot {growth['equity'].iloc[-1]:+.0%}", BOT)])
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    path = out / "paper_equity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --- evolution -----------------------------------------------------------------------

GROUP_COLORS = {"offspring": SERIES[0], "survivor": SERIES[1], "newcomer": SERIES[2]}
GROUP_LABELS = {"offspring": "offspring (crossover / mutation)",
                "survivor": "survivor (elite, unchanged)",
                "newcomer": "newcomer (random / immigrant)"}
OBJECTIVE_TITLES = {"excess": "Goal: beat buy & hold", "sharpe": "Goal: highest Sharpe ratio"}


def _fitness_range(df: pd.DataFrame) -> tuple[float, float]:
    lo, hi = df["fitness"].quantile(0.05), df["fitness"].max()
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def evolution_swarm_gif(evo_dir: Path, out: Path, seed: int = 0) -> Path | None:
    """Animated side-by-side swarm: each dot is a strategy, one frame per generation."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    from evotrader.evolution import history as h

    runs = [(obj, h.load(evo_dir / f"{obj}_seed{seed}.json")[1]) for obj in ("excess", "sharpe")
            if (evo_dir / f"{obj}_seed{seed}.json").exists()]
    if not runs:
        return None
    last_gen = int(runs[0][1]["generation"].max())
    fig, axes = plt.subplots(1, len(runs), figsize=(11, 4.8), squeeze=False)
    axes = axes[0]

    def draw(gen: int) -> None:
        for ax, (obj, df) in zip(axes, runs, strict=True):
            ax.clear()
            lo, hi = _fitness_range(df)
            now = df[df["generation"] == gen]
            for group in ("offspring", "newcomer", "survivor"):  # survivors drawn on top
                pts = now[now["group"] == group]
                ax.scatter(pts["exposure"], pts["fitness"].clip(lower=lo + (hi - lo) * 0.01),
                           s=46 if group == "survivor" else 30, color=GROUP_COLORS[group],
                           edgecolors=SURFACE, linewidths=1.2, alpha=0.9, zorder=3,
                           label=GROUP_LABELS[group])
            median = now["exposure"].median()
            ax.axvline(median, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(f"median invested {median:.0%}", (median, hi), xytext=(4, -12),
                        textcoords="offset points", fontsize=8.5, color=TEXT_2,
                        ha="right" if median > 0.6 else "left")
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(lo, hi)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_xlabel("share of time invested")
            ax.set_title(OBJECTIVE_TITLES.get(obj, obj), fontsize=11.5)
        axes[0].set_ylabel("training fitness")
        fig.suptitle(f"Evolution in action: generation {gen} of {last_gen}", x=0.01, ha="left",
                     fontsize=13, fontweight="bold")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legends.clear()
        fig.legend(handles, labels, loc="lower center", ncols=3, fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, 0.97))

    frames = list(range(last_gen + 1)) + [last_gen] * 5  # linger on the final generation
    anim = FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames))
    path = out / "evolution_swarm.gif"
    anim.save(path, writer=PillowWriter(fps=2), dpi=80)
    plt.close(fig)
    return path


def family_tree_chart(evo_dir: Path, out: Path, objective: str = "excess",
                      seed: int = 0) -> Path | None:
    """Every ancestor of the final champion, by generation and fitness."""
    from evotrader.evolution import history as h

    path_in = evo_dir / f"{objective}_seed{seed}.json"
    if not path_in.exists():
        return None
    _, df = h.load(path_in)
    champ = h.champion(df)
    tree = h.ancestry(df, champ["id"]).set_index("id")
    line = h.main_line(df, champ["id"])
    steps = h.story(df, champ["id"])

    fig, ax = plt.subplots(figsize=(10, 5.4))
    for _, child in tree.iterrows():
        for k, parent_id in enumerate(child["parents"]):
            parent = tree.loc[parent_id]
            ax.plot([parent["generation"], child["generation"]],
                    [parent["fitness"], child["fitness"]], color="#b5b3ad", linewidth=0.8,
                    linestyle="-" if k == 0 else (0, (3, 2)), zorder=1)
    ax.scatter(tree["generation"], tree["fitness"], s=14, color="#9c9a93", zorder=2,
               label="ancestor")
    ax.plot(line["generation"], line["fitness"], color=BOT, linewidth=2.2, zorder=3,
            label="direct line of descent")
    ax.scatter(line["generation"], line["fitness"], s=22, color=BOT, zorder=4)
    ax.scatter([champ["generation"]], [champ["fitness"]], s=140, marker="*", color=BOT,
               edgecolors=SURFACE, linewidths=1.2, zorder=5, label="champion")

    # Label only breakthroughs: the founder and each step that set a new best in the line.
    breakthroughs = steps[steps["fitness"] > steps["fitness"].cummax().shift(fill_value=-np.inf)]
    for _, step in breakthroughs.iterrows():
        label = step["how"].replace(" (entry)", "").replace(" (exit)", "")
        ax.annotate(f"{label}\n{step['fitness']:+.3f}", (step["generation"], step["fitness"]),
                    xytext=(0, 14), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=TEXT, linespacing=1.1)
    span = line["fitness"].max() - line["fitness"].min()
    ax.set_ylim(line["fitness"].min() - 0.35 * span, line["fitness"].max() + 0.35 * span)
    ax.set_title(f"How the champion was born ({OBJECTIVE_TITLES[objective].lower()})", pad=24)
    ax.annotate(f"{len(tree)} ancestors. Labels mark each new best in the direct line; "
                "dips are mutations that hurt before a later change recovered.",
                (0, 1.02), xycoords="axes fraction", fontsize=8.5, color=TEXT_2)
    ax.annotate(f"champion: {champ['rule']}", (0.01, 0.03), xycoords="axes fraction",
                fontsize=8.5, color=TEXT_2)
    ax.set_xlabel("generation")
    ax.set_ylabel("training fitness")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    path = out / "family_tree.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
