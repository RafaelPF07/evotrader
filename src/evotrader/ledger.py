"""The trial ledger: every configuration ever tried, so "wins" can be deflated honestly.

Iterating on the same history is fine *if the project keeps count*. Every exploration
run appends a row here, and significance is always computed against the running
total: the 36 trials from the pre-registered experiments plus every row below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LEDGER = Path(__file__).resolve().parents[2] / "reports" / "trials.csv"
HISTORICAL_TRIALS = 36  # original bot + experiments 1-3 (see reports/significance.md)
COLUMNS = ["trial", "timestamp", "name", "description", "config", "sharpe", "bh_sharpe",
           "cagr", "bh_cagr", "max_drawdown", "exposure", "dsr"]


def read(path: Path = LEDGER) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)


def total_trials(path: Path = LEDGER) -> int:
    return HISTORICAL_TRIALS + len(read(path))


def next_trial(path: Path = LEDGER) -> int:
    """The number the next recorded trial will get (use it for the DSR before recording)."""
    return total_trials(path) + 1


def record(name: str, description: str, config: dict, metrics: dict,
           path: Path = LEDGER) -> int:
    ledger = read(path)
    trial = HISTORICAL_TRIALS + len(ledger) + 1
    row = {"trial": trial, "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
           "name": name, "description": description,
           "config": json.dumps(config, sort_keys=True, default=str),
           **{k: metrics.get(k) for k in COLUMNS[5:]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)[COLUMNS].to_csv(path, index=False)
    return trial
