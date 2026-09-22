"""Exploration rounds: iterate freely on historical data, honestly.

Each round evolves strategies with a walk-forward over 2007 to today (test windows
2012-2026, each unseen by the evolution that produced its rules), for several random
seeds in parallel. The seeds' strategies are combined equally into one "ensemble",
which is what gets reported and compared with equal-weight buy & hold.

Honesty rules:
- Every round is appended to the trial ledger (reports/trials.csv), and its deflated
  Sharpe ratio is computed against the running trial count.
- Results are *exploration*: this history has been used many times. A promising
  round earns a place in the live forward test; it does not prove anything by itself.
- With `max_exposure`, rules invested more than that share of the time are heavily
  penalised, so the search cannot win by becoming buy & hold.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from evotrader import ledger
from evotrader.data import DEFAULT_UNIVERSE
from evotrader.evolution.engine import EvolutionConfig, genome_kinds
from evotrader.evolution.fitness import Dataset, FitnessConfig
from evotrader.metrics import compute_metrics
from evotrader.stats import deflated_sharpe
from evotrader.walkforward import Fold, load_datasets, make_folds, run_walkforward

RESULTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "exploration"


@dataclass(frozen=True)
class ExploreConfig:
    name: str
    description: str = ""
    objective: str = "excess"
    max_exposure: float | None = 0.8
    macro: bool = True
    seeds: tuple[int, ...] = (0, 1, 2)
    population: int = 60
    generations: int = 25
    train_start: str = "2007-01-01"
    first_test_year: int = 2012
    test_years: int = 2
    snapshot: str | None = None  # frozen data snapshot (data/snapshots/<name>)
    space: str = "timing"  # "timing" (in/out per ETF) or "switch" (risk-on/off baskets)


def _run_seed(job: tuple[list[Dataset], list[Fold], ExploreConfig, int]) -> dict:
    datasets, folds, cfg, seed = job
    report = run_walkforward(
        datasets, folds,
        EvolutionConfig(population=cfg.population, generations=cfg.generations, seed=seed),
        FitnessConfig(objective=cfg.objective, max_exposure=cfg.max_exposure),
        log=lambda _: None, space=cfg.space,
    )
    exposure = pd.concat([r.test_exposure["evolved"] for r in report.folds])
    return {
        "seed": seed,
        "returns": report.oos_returns["evolved"],
        "buy_and_hold": report.oos_returns["buy_and_hold"],
        "exposure": exposure,
        "champions": [str(r.champion) for r in report.folds],
        "kinds": [k for r in report.folds for k in genome_kinds(r.champion)],
    }


def load(cfg: ExploreConfig) -> tuple[list[Dataset], list[Fold]]:
    if cfg.snapshot:
        from evotrader import data

        data.use_snapshot(cfg.snapshot)
    datasets = load_datasets(DEFAULT_UNIVERSE, use_ml=False, log=lambda _: None)
    if cfg.macro:
        from evotrader.macro import attach_macro

        datasets = attach_macro(datasets)
    last = min(ds.bars.index[-1] for ds in datasets)
    return datasets, make_folds(cfg.train_start, cfg.first_test_year, last, cfg.test_years)


def summarise(returns: pd.Series, bh: pd.Series, exposure: pd.Series, n_trials: int) -> dict:
    m = compute_metrics(returns, exposure)
    b = compute_metrics(bh, pd.Series(1.0, index=bh.index))
    sig = deflated_sharpe(returns - bh, n_trials)
    return {
        "sharpe": m["sharpe"], "bh_sharpe": b["sharpe"], "cagr": m["cagr"], "bh_cagr": b["cagr"],
        "max_drawdown": m["max_drawdown"], "bh_max_drawdown": b["max_drawdown"],
        "exposure": float(exposure.mean()), "corr_with_bh": float(returns.corr(bh)),
        "active_sharpe": sig.sharpe_annual, "psr": sig.psr, "dsr": sig.dsr,
        "beats_bh": m["cagr"] > b["cagr"] and m["sharpe"] > b["sharpe"],
    }


def run(cfg: ExploreConfig, log=print) -> dict:
    datasets, folds = load(cfg)
    log(f"[{cfg.name}] {len(folds)} walk-forward folds, seeds {list(cfg.seeds)}, "
        f"macro={'on' if cfg.macro else 'off'}, max exposure={cfg.max_exposure}, "
        f"space={cfg.space}, snapshot={cfg.snapshot}")
    jobs = [(datasets, folds, cfg, s) for s in cfg.seeds]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        seeds = list(pool.map(_run_seed, jobs))

    n_trials = ledger.next_trial()
    bh = seeds[0]["buy_and_hold"]
    ensemble = pd.concat([s["returns"] for s in seeds], axis=1).mean(axis=1)
    ensemble_exposure = pd.concat([s["exposure"] for s in seeds], axis=1).mean(axis=1)
    fingerprint = None
    if cfg.snapshot:
        from evotrader import data

        fingerprint = data.fingerprint(data.SNAPSHOT_DIR / cfg.snapshot)
    result = {
        "config": asdict(cfg), "trial": n_trials, "data_fingerprint": fingerprint,
        "period": f"{bh.index[0].date()}..{bh.index[-1].date()}",
        "ensemble": summarise(ensemble, bh, ensemble_exposure, n_trials),
        "seeds": [{"seed": s["seed"], **summarise(s["returns"], bh, s["exposure"], n_trials),
                   "champions": s["champions"]} for s in seeds],
        "champion_building_blocks": Counter(k for s in seeds for k in s["kinds"]).most_common(),
    }
    ledger.record(cfg.name, cfg.description, {**asdict(cfg), "data_fingerprint": fingerprint},
                  result["ensemble"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{cfg.name}.json").write_text(json.dumps(result, indent=2, default=str),
                                                  encoding="utf-8")
    pd.DataFrame({"ensemble": ensemble, "buy_and_hold": bh,
                  **{f"seed_{s['seed']}": s["returns"] for s in seeds}}).to_csv(
        RESULTS_DIR / f"{cfg.name}_returns.csv")
    return result
