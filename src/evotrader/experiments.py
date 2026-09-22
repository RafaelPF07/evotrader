"""Pre-registered experiments: can a change make the bot beat buy & hold?

The protocol is fixed in this file and committed *before* any results exist:

1. **Development stage.** Data is physically cut at DEV_END. Walk-forward test
   windows are 2012-2021 (2 years each). Every arm is run with the same 5 seeds.
2. **Pass rule (decided in advance).** An arm passes development if its Sharpe beats
   equal-weight buy & hold's on average across the seeds AND in at least 4 of 5.
3. **Holdout stage.** Every arm is then trained on data up to DEV_END and tested
   once on HOLDOUT_START..today, a period no experiment has been developed on.
   Results for all arms are reported, pass or fail. The holdout refuses to run
   twice, so it can't be re-rolled until it looks good.

Nothing in ARMS, SEEDS or the dates may change after results exist; a new idea
means a new arm, and it counts as one more trial.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from evotrader.evolution.engine import EvolutionConfig
from evotrader.evolution.fitness import Dataset, FitnessConfig, truncate
from evotrader.walkforward import make_folds, run_walkforward

DEV_END = "2021-12-31"
HOLDOUT_START_YEAR = 2022
TRAIN_START = "2007-01-01"
DEV_FIRST_TEST_YEAR = 2012
SEEDS = (0, 1, 2, 3, 4)
PASS_MIN_SEEDS = 4
RESULTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "experiments"

Log = Callable[[str], None]


@dataclass(frozen=True)
class Arm:
    name: str
    description: str
    space: str  # "timing" | "rotation"
    objective: str  # "sharpe" | "excess"


ARMS = {
    arm.name: arm
    for arm in (
        Arm("A", "baseline: in/out timing rules, Sharpe fitness", "timing", "sharpe"),
        Arm("B", "idea 1: timing rules, fitness = information ratio vs buy & hold",
            "timing", "excess"),
        Arm("C", "idea 2: ETF rotation (always invested), Sharpe fitness", "rotation", "sharpe"),
        Arm("D", "ideas 1+2: ETF rotation, information-ratio fitness", "rotation", "excess"),
    )
}


def result_path(stage: str, arm: str) -> Path:
    return RESULTS_DIR / f"{stage}_{arm}.csv"


def stage_setup(stage: str, datasets: list[Dataset]) -> tuple[list[Dataset], list]:
    if stage == "dev":
        data = truncate(datasets, DEV_END)
        last = min(ds.bars.index[-1] for ds in data)
        return data, make_folds(TRAIN_START, DEV_FIRST_TEST_YEAR, last, test_years=2)
    if stage == "holdout":
        last = min(ds.bars.index[-1] for ds in datasets)
        span = last.year - HOLDOUT_START_YEAR + 1
        return datasets, make_folds(TRAIN_START, HOLDOUT_START_YEAR, last, test_years=span)
    raise ValueError("stage must be 'dev' or 'holdout'")


def check_allowed(stage: str, arm: str) -> None:
    if stage == "holdout":
        missing = [a for a in ARMS if not result_path("dev", a).exists()]
        if missing:
            raise RuntimeError(f"Run the development stage for arms {missing} first")
        if result_path("holdout", arm).exists():
            raise RuntimeError(f"Holdout for arm {arm} already ran; it is judged exactly once")


def run_arm(
    arm: Arm, stage: str, datasets: list[Dataset], evo: EvolutionConfig | None = None,
    seeds: tuple[int, ...] = SEEDS, log: Log = print,
) -> pd.DataFrame:
    data, folds = stage_setup(stage, datasets)
    evo = evo or EvolutionConfig()
    rows = []
    for seed in seeds:
        log(f"[{stage} {arm.name}] seed {seed}")
        report = run_walkforward(
            data, folds, EvolutionConfig(**{**vars(evo), "seed": seed}),
            FitnessConfig(objective=arm.objective), log=lambda _: None, space=arm.space,
        )
        m, bh = report.oos_metrics.loc["evolved"], report.oos_metrics.loc["buy_and_hold"]
        rows.append({
            "arm": arm.name, "stage": stage, "seed": seed,
            "sharpe": m["sharpe"], "cagr": m["cagr"], "max_drawdown": m["max_drawdown"],
            "exposure": m["exposure"], "bh_sharpe": bh["sharpe"], "bh_cagr": bh["cagr"],
            "bh_max_drawdown": bh["max_drawdown"],
            "excess_sharpe": m["sharpe"] - bh["sharpe"], "excess_cagr": m["cagr"] - bh["cagr"],
            "periods_beating_bh": sum(
                r.test_metrics["evolved"]["sharpe"] > r.test_metrics["buy_and_hold"]["sharpe"]
                for r in report.folds),
            "periods": len(report.folds),
            "last_champion": str(report.folds[-1].champion),
        })
        log(f"    Sharpe {m['sharpe']:.2f} vs buy & hold {bh['sharpe']:.2f}")
    return pd.DataFrame(rows)


def verdict(results: pd.DataFrame) -> dict[str, float | bool]:
    """Apply the pre-registered pass rule to one arm's per-seed results."""
    wins = int((results["excess_sharpe"] > 0).sum())
    mean = float(results["excess_sharpe"].mean())
    return {"mean_excess_sharpe": mean, "seeds_beating": wins,
            "passed": mean > 0 and wins >= PASS_MIN_SEEDS}


def summary_table(stage: str) -> pd.DataFrame:
    rows = []
    for name, arm in ARMS.items():
        path = result_path(stage, name)
        if not path.exists():
            continue
        res = pd.read_csv(path)
        rows.append({
            "arm": name, "description": arm.description,
            "sharpe": res["sharpe"].mean(), "bh_sharpe": res["bh_sharpe"].mean(),
            "cagr": res["cagr"].mean(), "bh_cagr": res["bh_cagr"].mean(),
            "max_drawdown": res["max_drawdown"].mean(),
            "bh_max_drawdown": res["bh_max_drawdown"].mean(),
            **verdict(res),
        })
    return pd.DataFrame(rows)
