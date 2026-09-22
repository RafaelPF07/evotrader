"""Adapter so an evolved genome can be used anywhere a hand-written Strategy can."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from evotrader.evolution.genome import Genome
from evotrader.features import FeatureStore
from evotrader.strategies.base import Strategy


class GeneticStrategy(Strategy):
    name = "evolved"

    def __init__(self, genome: Genome) -> None:
        super().__init__(rule=str(genome))
        self.genome = genome

    def _signal(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(self.genome.target_position(FeatureStore(bars)), index=bars.index)

    def describe(self) -> str:
        return f"evolved[{self.genome}]"

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rule": str(self.genome), "genome": self.genome.to_dict(), **(metadata or {})}
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> GeneticStrategy:
        return cls(Genome.from_dict(json.loads(path.read_text(encoding="utf-8"))["genome"]))
