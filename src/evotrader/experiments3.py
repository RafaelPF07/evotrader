"""Pre-registered experiment 3: leveraged trend following vs buy & hold.

Fixed here and committed *before* any results exist (see docs/EXPERIMENTS_3.md).
Nothing is tuned: the 200-day average and the leverage come from the literature
and from the user's 1.5x limit, so each arm is exactly one trial.

Stages
    dev              the 10 US ETFs used throughout the project, data cut at 2021-12-31,
                     evaluated 2007-2021 (development and sanity checks only)
    final-sectors    5 US sector ETFs this project has never used, 2000-01-01 to today
    final-countries  13 country ETFs this project has never used, 2001-06-01 to today
    post-publication (reported, not judged) the 10 US ETFs since the paper appeared,
                     2016-12-01 to today; this data was already seen in experiment 1
Each final stage runs once; the code refuses to run it again.

Arms
    L1  trend filter, no leverage (diagnostic: the classic 200-day rule)
    L2  trend filter, 1.5x leverage above the average (the paper's idea)
    L3  trend filter plus experiment 2's volatility targeting, up to 1.5x

Scoring: CAGR on raw returns; Sharpe on returns in excess of the T-bill rate, for
every arm and for buy & hold. Pass rule: an arm beats buy & hold only if its CAGR
AND Sharpe are both higher than equal-weight buy & hold's in dev AND both finals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from evotrader.backtest import DEFAULT_COST_BPS
from evotrader.data import DEFAULT_UNIVERSE, load, load_rates
from evotrader.evolution.fitness import Dataset, truncate
from evotrader.evolution.rotation import Panel
from evotrader.metrics import TRADING_DAYS, cagr, max_drawdown, sharpe
from evotrader.sizing import VolTarget
from evotrader.trend import TrendLeverage, simulate

RESULTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "experiments3"

SECTORS = ["XLI", "XLP", "XLU", "XLY", "XLB"]
COUNTRIES = ["EWW", "EWT", "EWY", "EWS", "EWH", "EWP", "EWQ", "EWI", "EWL", "EWN", "EWD",
             "EWK", "EWO"]


@dataclass(frozen=True)
class Stage:
    name: str
    tickers: tuple[str, ...]
    data_end: str | None
    start: str
    end: str | None
    judged: bool = True


STAGES = {
    s.name: s
    for s in (
        Stage("dev", tuple(DEFAULT_UNIVERSE), "2021-12-31", "2007-01-01", "2021-12-31"),
        Stage("final-sectors", tuple(SECTORS), None, "2000-01-01", None),
        Stage("final-countries", tuple(COUNTRIES), None, "2001-06-01", None),
        Stage("post-publication", tuple(DEFAULT_UNIVERSE), None, "2016-12-01", None,
              judged=False),
    )
}
FINALS = ("final-sectors", "final-countries")

ARMS = {
    "L1": TrendLeverage(sma=200, leverage=1.0),
    "L2": TrendLeverage(sma=200, leverage=1.5),
    "L3": TrendLeverage(sma=200, leverage=1.5,
                        vol_target=VolTarget(lookback=20, target_scale=1.0, band=0.10, cap=1.5)),
}


def result_path(stage: str) -> Path:
    return RESULTS_DIR / f"{stage}.csv"


def check_allowed(stage: str) -> None:
    if stage in FINALS or stage == "post-publication":
        if not result_path("dev").exists():
            raise RuntimeError("Run the dev stage first")
    if stage in FINALS and result_path(stage).exists():
        raise RuntimeError(f"{stage} already ran; each final test is run exactly once")


def load_panel(stage: Stage) -> Panel:
    datasets = [Dataset(t, load(t, start="1990-01-01")) for t in stage.tickers]
    if stage.data_end:
        datasets = truncate(datasets, stage.data_end)
    return Panel(datasets)


def score(returns: pd.Series, held: pd.Series, rates: pd.Series) -> dict[str, float]:
    rf = rates.reindex(returns.index).ffill().fillna(0.0).clip(lower=0.0) / TRADING_DAYS
    return {
        "cagr": cagr(returns),
        "sharpe": sharpe(returns - rf),  # excess over T-bills, the textbook definition
        "volatility": float(returns.std() * TRADING_DAYS ** 0.5),
        "max_drawdown": max_drawdown(returns),
        "avg_exposure": float(held.mean()),
        "days_levered": float((held > 1.0 + 1e-9).mean()),
        "days_in_cash": float((held < 1e-9).mean()),
        "switches_per_year": float((held.diff().abs() > 1e-9).sum()
                                   / (len(held) / TRADING_DAYS)),
    }


def run_stage(stage_name: str, rates: pd.Series | None = None,
              cost_bps: float = DEFAULT_COST_BPS) -> pd.DataFrame:
    stage = STAGES[stage_name]
    rates = load_rates() if rates is None else rates
    panel = load_panel(stage)
    end = stage.end or panel.index[-1]

    bh = panel.buy_and_hold(stage.start, end, cost_bps)
    bh_score = score(bh, pd.Series(1.0, index=bh.index), rates)
    rows = [{"stage": stage_name, "arm": "buy_and_hold", "settings": "-", **bh_score}]
    for name, tl in ARMS.items():
        returns, held = simulate(panel, tl, cost_bps, rates)
        s = score(returns.loc[stage.start:end], held.loc[stage.start:end], rates)
        rows.append({"stage": stage_name, "arm": name, "settings": str(tl), **s,
                     "beats_bh": s["cagr"] > bh_score["cagr"] and s["sharpe"] > bh_score["sharpe"]})
    return pd.DataFrame(rows)


def verdict() -> pd.DataFrame:
    judged = [s for s, st in STAGES.items() if st.judged]
    frames = [pd.read_csv(result_path(s)) for s in judged if result_path(s).exists()]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    arms = df[df["arm"] != "buy_and_hold"]
    wins = arms.pivot(index="arm", columns="stage", values="beats_bh")
    wins["all_stages_run"] = len(frames) == len(judged)
    wins["passes"] = wins.drop(columns="all_stages_run").all(axis=1) & wins["all_stages_run"]
    return wins
