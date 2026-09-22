"""Train the fixed rules for forward-test account 3 (exploration round 2's recipe).

Round 2 (trial 38): in/out rules per ETF, price + macro building blocks, Sharpe
objective, out of stocks at least 20% of the time, seeds 0-2 combined equally.

For the live account, each seed is evolved one final time on all data up to the
snapshot date, the same way every walk-forward fold did it: evolve on the first 75%
of 2007..today, pick the champion from the hall of fame on the last 25%. The three
champions are then frozen for the forward test.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from evotrader import data
from evotrader.evolution.engine import EvolutionConfig, Evolver
from evotrader.evolution.fitness import Dataset, FitnessConfig
from evotrader.evolution.genome import Genome
from evotrader.walkforward import load_datasets

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "forward_copper_gold.json"
RECIPE = {"objective": "sharpe", "max_exposure": 0.8, "macro": True, "seeds": [0, 1, 2],
          "population": 60, "generations": 25, "train_start": "2007-01-01",
          "val_fraction": 0.25}


def _train_seed(job: tuple[list[Dataset], str, str, str, int]) -> dict:
    datasets, start, val_start, end, seed = job
    fit = FitnessConfig(objective=RECIPE["objective"], max_exposure=RECIPE["max_exposure"])
    last_train = str((pd.Timestamp(val_start) - pd.Timedelta(days=1)).date())
    evolver = Evolver(datasets, start, last_train,
                      EvolutionConfig(population=RECIPE["population"],
                                      generations=RECIPE["generations"], seed=seed), fit)
    hall = evolver.run().hall_of_fame
    scored = [(g, e, evolver.evaluate_window(g, val_start, end)) for g, e in hall]
    champion, train, val = max(scored, key=lambda t: t[2].fitness)
    return {"seed": seed, "rule": str(champion), "genome": champion.to_dict(),
            "train_fitness": train.fitness, "val_fitness": val.fitness,
            "val_exposure": val.exposure}


def train(snapshot: str) -> dict:
    from evotrader.macro import attach_macro

    fingerprint = data.use_snapshot(snapshot)
    datasets = attach_macro(load_datasets(data.DEFAULT_UNIVERSE, use_ml=False,
                                          log=lambda _: None))
    end_ts = min(ds.bars.index[-1] for ds in datasets)
    start = pd.Timestamp(RECIPE["train_start"])
    val_start = (start + (end_ts - start) * (1 - RECIPE["val_fraction"])).normalize()
    end = str(end_ts.date())
    jobs = [(datasets, RECIPE["train_start"], str(val_start.date()), end, s)
            for s in RECIPE["seeds"]]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        rules = list(pool.map(_train_seed, jobs))
    model = {"name": "copper-gold ensemble (exploration round 2 recipe)", "recipe": RECIPE,
             "snapshot": snapshot, "data_fingerprint": fingerprint,
             "trained": f"{RECIPE['train_start']}..{val_start.date()}",
             "validated": f"{val_start.date()}..{end}",
             "created": pd.Timestamp.now().isoformat(timespec="seconds"), "rules": rules}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def load_genomes(model: dict) -> list[Genome]:
    return [Genome.from_dict(r["genome"]) for r in model["rules"]]
