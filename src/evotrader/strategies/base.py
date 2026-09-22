"""Strategy interface.

A strategy looks at bars up to the close of day t and outputs a target position
for day t: 0 = flat (cash), 1 = fully invested. The backtester, not the strategy,
decides when that order can actually be filled, so a strategy cannot cheat.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    name: str = "strategy"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def _signal(self, bars: pd.DataFrame) -> pd.Series:
        """Raw target position per date. NaN means 'not enough history yet'."""

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        pos = self._signal(bars).reindex(bars.index)
        return pos.fillna(0.0).clip(0.0, 1.0).astype(float).rename("target")

    def describe(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({args})"

    def __repr__(self) -> str:
        return self.describe()
