"""Deflated Sharpe ratios for every result in this project that looked like a win.

Each candidate is judged on its *active* returns (strategy minus equal-weight buy &
hold, day by day) over the one-time test period it was judged on. The question is
whether that edge survives a correction for how many configurations were tried.

Trial count (see the experiment docs): 1 original bot + 3 (experiment 1: B, C, D)
+ 2 + 27 (experiment 2: V0, V1, and the 27 settings behind V2) + 3 (experiment 3)
= 36. Trials were correlated, so the effective number is lower; N = 10 is shown too.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd

from evotrader import experiments as ex1
from evotrader import experiments2 as ex2
from evotrader import experiments3 as ex3
from evotrader.data import DEFAULT_UNIVERSE, load_rates
from evotrader.evolution.engine import EvolutionConfig
from evotrader.evolution.fitness import FitnessConfig
from evotrader.sizing import simulate as simulate_vol
from evotrader.stats import deflated_sharpe
from evotrader.trend import simulate as simulate_trend
from evotrader.walkforward import load_datasets, run_walkforward

TRIALS = 36
TRIAL_COUNTS = (1, 10, TRIALS)

Log = Callable[[str], None]


def experiment1_arm_b(log: Log = print) -> list[tuple[str, pd.Series, float]]:
    """Reproduce arm B's one-time holdout (5 seeds) and return its active returns.

    Deterministic: same seeds, same code. The recorded Sharpe is returned alongside so
    the caller can confirm the reproduction matches what was judged.
    """
    recorded = pd.read_csv(ex1.result_path("holdout", "B")).set_index("seed")["sharpe"]
    datasets = load_datasets(DEFAULT_UNIVERSE, use_ml=False, log=lambda _: None)
    data, folds = ex1.stage_setup("holdout", datasets)
    out = []
    for seed in ex1.SEEDS:
        log(f"  reproducing experiment 1 arm B, seed {seed}")
        report = run_walkforward(data, folds, EvolutionConfig(seed=seed),
                                 FitnessConfig(objective="excess"), log=lambda _: None)
        active = report.oos_returns["evolved"] - report.oos_returns["buy_and_hold"]
        reproduced = report.oos_metrics.loc["evolved", "sharpe"]
        out.append((f"exp1 B seed {seed}", active, reproduced - recorded[seed]))
    return out


def rule_based(log: Log = print) -> list[tuple[str, str, pd.Series]]:
    """Active returns of experiment 2 and 3 arms on each judged stage (deterministic)."""
    rates = load_rates()
    out = []
    v2 = ex2.VolTarget(**json.loads(ex2.tuned_path().read_text("utf-8")))
    for stage_name in ("dev", *ex2.FINALS):
        stage = ex2.STAGES[stage_name]
        panel = ex2.load_panel(stage)
        end = stage.end or panel.index[-1]
        bh = panel.buy_and_hold(stage.start, end, ex2.DEFAULT_COST_BPS)
        for arm, vt in (("V1", ex2.TEXTBOOK), ("V2", v2)):
            r, _ = simulate_vol(panel, vt, ex2.DEFAULT_COST_BPS, rates)
            out.append((f"exp2 {arm}", stage_name, r.loc[stage.start:end] - bh))
    for stage_name in ("dev", *ex3.FINALS):
        stage = ex3.STAGES[stage_name]
        panel = ex3.load_panel(stage)
        end = stage.end or panel.index[-1]
        bh = panel.buy_and_hold(stage.start, end, ex3.DEFAULT_COST_BPS)
        r, _ = simulate_trend(panel, ex3.ARMS["L2"], ex3.DEFAULT_COST_BPS, rates)
        out.append(("exp3 L2", stage_name, r.loc[stage.start:end] - bh))
    log("  computed experiment 2 and 3 arms")
    return out


def row(candidate: str, stage: str, active: pd.Series) -> dict:
    sig = {n: deflated_sharpe(active, n) for n in TRIAL_COUNTS}
    s = sig[TRIALS]
    return {
        "candidate": candidate, "stage": stage, "days": s.n_obs,
        "active_sharpe": s.sharpe_annual, "skew": s.skew, "kurtosis": s.kurtosis,
        "luck_threshold_n36": s.luck_threshold_annual,
        "psr": sig[1].psr, "dsr_n10": sig[10].dsr, "dsr_n36": sig[TRIALS].dsr,
    }


def build(log: Log = print) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, checks = [], []
    for name, active, diff in experiment1_arm_b(log):
        rows.append(row(name, "holdout 2022-26", active))
        checks.append({"candidate": name, "sharpe_diff_vs_recorded": diff})
    for name, stage, active in rule_based(log):
        rows.append(row(name, stage, active))
    return pd.DataFrame(rows), pd.DataFrame(checks)


def to_markdown(table: pd.DataFrame, checks: pd.DataFrame | None = None) -> str:
    notes = []
    if checks is not None and len(checks):
        notes = [
            "",
            "## Reproduction check (experiment 1)",
            "",
            "Experiment 1's evolved strategies were re-run with the same seeds and code. "
            "Seeds whose Sharpe differs from the recorded one were affected by a re-download "
            "of the price data that changed prices by about one part in a million; the "
            "original code gives the same differences on today's data, so the code is not "
            "the cause (see docs/EXPERIMENTS.md, 'Reproducibility'). The rows above use the "
            "reproduced series.",
            "",
            "| seed | reproduced minus recorded Sharpe |",
            "|---|---|",
            *[f"| {r['candidate']} | {r['sharpe_diff_vs_recorded']:+.4f} |"
              for _, r in checks.iterrows()],
        ]
    lines = [
        "# Significance: do any of the wins survive a correction for trying 36 things?",
        "",
        "Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) of each candidate's *active* "
        "returns (strategy minus equal-weight buy & hold) on the period it was judged on.",
        "- **active Sharpe**: annualised Sharpe of beating buy & hold",
        "- **PSR**: probability the edge is real, ignoring how many things were tried",
        "- **DSR**: the same, after accounting for N trials. Above 0.95 means significant.",
        "",
        "| candidate | stage | days | active Sharpe | luck threshold (N=36) | PSR | DSR N=10 "
        "| DSR N=36 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['candidate']} | {r['stage']} | {r['days']} | {r['active_sharpe']:+.2f} | "
            f"{r['luck_threshold_n36']:.2f} | {r['psr']:.2f} | {r['dsr_n10']:.2f} | "
            f"{r['dsr_n36']:.2f} |")
    return "\n".join(lines + notes) + "\n"
