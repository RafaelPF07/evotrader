"""Pre-registered experiment 2: can volatility targeting beat buy & hold?

Fixed here and committed *before* any results exist (see docs/EXPERIMENTS_2.md):

Stages
    dev           US universe (10 ETFs), data cut at 2021-12-31, evaluated 2007-2021.
                  V2's settings are chosen here, from GRID, by highest Sharpe.
    final-us2000  The 8 US ETFs that existed in 2000, data cut at 2006-12-31, evaluated
                  2001-06-01..2006-12-31: a period no part of this project has used.
    final-intl    8 international ETFs never used by this project, 2004-06-01..today.
Each final stage runs once; the code refuses to run it again.

Arms
    V0  textbook settings, no leverage (diagnostic: how much comes from leverage?)
    V1  textbook settings, up to 1.5x leverage
    V2  settings tuned on dev from GRID (27 combinations), up to 1.5x leverage

Pass rule: an arm "beats buy & hold" only if its CAGR AND Sharpe are both higher than
equal-weight buy & hold's, in dev AND in both final stages.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from evotrader.backtest import DEFAULT_COST_BPS
from evotrader.data import DEFAULT_UNIVERSE, load, load_rates
from evotrader.evolution.fitness import Dataset, truncate
from evotrader.evolution.rotation import Panel
from evotrader.metrics import compute_metrics
from evotrader.sizing import VolTarget, simulate

RESULTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "experiments2"

US_2000 = ["SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV"]
INTERNATIONAL = ["EFA", "EEM", "EWJ", "EWG", "EWU", "EWC", "EWA", "EWZ"]


@dataclass(frozen=True)
class Stage:
    name: str
    tickers: tuple[str, ...]
    data_end: str | None  # bars after this date are physically removed
    start: str  # evaluation window
    end: str | None


STAGES = {
    s.name: s
    for s in (
        Stage("dev", tuple(DEFAULT_UNIVERSE), "2021-12-31", "2007-01-01", "2021-12-31"),
        Stage("final-us2000", tuple(US_2000), "2006-12-31", "2001-06-01", "2006-12-31"),
        Stage("final-intl", tuple(INTERNATIONAL), None, "2004-06-01", None),
    )
}
FINALS = ("final-us2000", "final-intl")

TEXTBOOK = VolTarget(lookback=20, target_scale=1.0, band=0.10, cap=1.5)
ARMS = {
    "V0": "textbook volatility targeting, no leverage (cap 1.0)",
    "V1": "textbook volatility targeting, up to 1.5x leverage",
    "V2": "volatility targeting tuned on dev (27 settings), up to 1.5x leverage",
}
GRID = [
    VolTarget(lookback=lb, target_scale=sc, band=bd, cap=1.5)
    for lb, sc, bd in itertools.product((10, 20, 63), (0.8, 1.0, 1.2), (0.0, 0.10, 0.25))
]


def result_path(stage: str) -> Path:
    return RESULTS_DIR / f"{stage}.csv"


def tuned_path() -> Path:
    return RESULTS_DIR / "v2_selected.json"


def check_allowed(stage: str) -> None:
    if stage in FINALS:
        if not result_path("dev").exists() or not tuned_path().exists():
            raise RuntimeError("Run the dev stage first")
        if result_path(stage).exists():
            raise RuntimeError(f"{stage} already ran; each final test is run exactly once")


def load_panel(stage: Stage) -> Panel:
    datasets = [Dataset(t, load(t, start="1990-01-01")) for t in stage.tickers]
    if stage.data_end:
        datasets = truncate(datasets, stage.data_end)
    return Panel(datasets)


def evaluate(panel: Panel, vt: VolTarget, stage: Stage, rates: pd.Series,
             cost_bps: float = DEFAULT_COST_BPS) -> dict[str, float]:
    returns, held = simulate(panel, vt, cost_bps, rates)
    window = slice(stage.start, stage.end)
    returns, held = returns.loc[window], held.loc[window]
    m = compute_metrics(returns, held)
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "volatility": m["volatility"],
            "max_drawdown": m["max_drawdown"], "avg_exposure": float(held.mean()),
            "days_levered": float((held > 1.0 + 1e-9).mean()), "turnover": m["turnover"]}


def buy_and_hold(panel: Panel, stage: Stage, cost_bps: float = DEFAULT_COST_BPS) -> dict:
    returns = panel.buy_and_hold(stage.start, stage.end or panel.index[-1], cost_bps)
    held = pd.Series(1.0, index=returns.index)
    m = compute_metrics(returns, held)
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "volatility": m["volatility"],
            "max_drawdown": m["max_drawdown"], "avg_exposure": 1.0, "days_levered": 0.0,
            "turnover": 0.0}


def select_v2(panel: Panel, stage: Stage, rates: pd.Series) -> tuple[VolTarget, pd.DataFrame]:
    """Pick V2's settings on the dev stage only: highest Sharpe across GRID."""
    rows = [{**asdict(vt), **evaluate(panel, vt, stage, rates)} for vt in GRID]
    table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    best = table.iloc[0]
    chosen = VolTarget(lookback=int(best["lookback"]), target_scale=float(best["target_scale"]),
                       band=float(best["band"]), cap=float(best["cap"]))
    return chosen, table


def run_stage(stage_name: str, rates: pd.Series | None = None) -> pd.DataFrame:
    stage = STAGES[stage_name]
    rates = load_rates() if rates is None else rates
    panel = load_panel(stage)
    if stage_name == "dev":
        v2, grid = select_v2(panel, stage, rates)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        grid.to_csv(RESULTS_DIR / "dev_grid.csv", index=False)
        tuned_path().write_text(json.dumps(asdict(v2), indent=2), encoding="utf-8")
    else:
        v2 = VolTarget(**json.loads(tuned_path().read_text(encoding="utf-8")))

    arms = {"V0": VolTarget(**{**asdict(TEXTBOOK), "cap": 1.0}), "V1": TEXTBOOK, "V2": v2}
    bh = buy_and_hold(panel, stage)
    rows = [{"stage": stage_name, "arm": "buy_and_hold", "settings": "-", **bh}]
    for name, vt in arms.items():
        m = evaluate(panel, vt, stage, rates)
        rows.append({"stage": stage_name, "arm": name, "settings": str(vt), **m,
                     "beats_bh": m["cagr"] > bh["cagr"] and m["sharpe"] > bh["sharpe"]})
    return pd.DataFrame(rows)


def verdict() -> pd.DataFrame:
    """Apply the pass rule across every stage that has run."""
    frames = [pd.read_csv(result_path(s)) for s in STAGES if result_path(s).exists()]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    arms = df[df["arm"] != "buy_and_hold"]
    wins = arms.pivot(index="arm", columns="stage", values="beats_bh")
    wins["all_stages_run"] = len(frames) == len(STAGES)
    wins["passes"] = wins.drop(columns="all_stages_run").all(axis=1) & wins["all_stages_run"]
    return wins
