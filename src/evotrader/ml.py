"""Machine-learning signal: P(price is higher in `horizon` days), made walk-forward.

The model is refitted every `retrain_every` bars on an expanding window of the past.
To predict at the close of day t it may only train on rows whose label was already
*known* at t. Label j depends on open[j + 1 + horizon], so training rows stop at
j = t - 1 - horizon ("purging"). Every probability produced here is therefore
genuinely out-of-sample, and safe for the GA to use as a building block.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from evotrader import indicators as ind
from evotrader.data import CACHE_DIR
from evotrader.features import ML_COLUMN
from evotrader.strategies.base import Strategy

MODEL_VERSION = 1


@dataclass(frozen=True)
class MlConfig:
    horizon: int = 5
    min_train: int = 504  # ~2 years before the first prediction
    retrain_every: int = 63  # refit quarterly
    max_iter: int = 150
    learning_rate: float = 0.05
    max_depth: int = 3

    def key(self) -> str:
        blob = repr((MODEL_VERSION, sorted(asdict(self).items())))
        return hashlib.sha1(blob.encode()).hexdigest()[:10]


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    vol20 = ind.volatility(close, 20)
    return pd.DataFrame(
        {
            "ret_1": close.pct_change(1),
            "ret_5": close.pct_change(5),
            "ret_20": close.pct_change(20),
            "ret_60": close.pct_change(60),
            "rsi_14": ind.rsi(close, 14),
            "rsi_2": ind.rsi(close, 2),
            "zscore_20": ind.zscore(close, 20),
            "trend_50": close / ind.sma(close, 50) - 1,
            "trend_200": close / ind.sma(close, 200) - 1,
            "vol_20": vol20,
            "vol_ratio": vol20 / ind.volatility(close, 63),
            "gap": bars["open"] / close.shift(1) - 1,
        },
        index=bars.index,
    )


def build_labels(bars: pd.DataFrame, horizon: int) -> pd.Series:
    """1 if a position entered at the next open would be up `horizon` days later."""
    entry = bars["open"].shift(-1)
    exit_ = bars["open"].shift(-1 - horizon)
    return (exit_ / entry - 1 > 0).astype(float).where(exit_.notna())


def walk_forward_proba(bars: pd.DataFrame, config: MlConfig | None = None) -> pd.Series:
    cfg = config or MlConfig()
    X = build_features(bars).to_numpy()
    y = build_labels(bars, cfg.horizon).to_numpy()
    proba = np.full(len(bars), np.nan)

    for t in range(cfg.min_train, len(bars), cfg.retrain_every):
        known = slice(0, t - cfg.horizon)  # rows j <= t - 1 - horizon
        Xt, yt = X[known], y[known]
        ok = ~np.isnan(yt) & ~np.isnan(Xt).any(axis=1)
        if ok.sum() < cfg.min_train // 2 or len(np.unique(yt[ok])) < 2:
            continue
        model = HistGradientBoostingClassifier(
            max_iter=cfg.max_iter,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            early_stopping=False,
            random_state=0,
        ).fit(Xt[ok], yt[ok])
        block = slice(t, min(t + cfg.retrain_every, len(bars)))
        proba[block] = model.predict_proba(X[block])[:, 1]

    return pd.Series(proba, index=bars.index, name=ML_COLUMN)


def attach_ml_prob(
    ticker: str, bars: pd.DataFrame, config: MlConfig | None = None, use_cache: bool = True
) -> pd.DataFrame:
    """Return `bars` with an `ml_prob` column, cached on disk per ticker and config."""
    cfg = config or MlConfig()
    path = CACHE_DIR / f"ml_{ticker.upper()}_{cfg.key()}.csv"
    proba = None
    if use_cache and path.exists():
        cached = pd.read_csv(path, index_col="date", parse_dates=True)[ML_COLUMN]
        if cached.index.equals(bars.index):
            proba = cached
    if proba is None:
        proba = walk_forward_proba(bars, cfg)
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            proba.to_frame().to_csv(path)
    return bars.assign(**{ML_COLUMN: proba})


def diagnostics(bars: pd.DataFrame, config: MlConfig | None = None) -> dict[str, float]:
    """Out-of-sample quality of the ML signal. Expect accuracy barely above the base rate."""
    cfg = config or MlConfig()
    labels = build_labels(bars, cfg.horizon)
    df = pd.DataFrame({"p": bars[ML_COLUMN], "y": labels}).dropna()
    if df.empty:
        return {}
    return {
        "samples": float(len(df)),
        "base_rate": float(df["y"].mean()),
        "accuracy": float(((df["p"] > 0.5) == (df["y"] == 1)).mean()),
        "auc": float(roc_auc_score(df["y"], df["p"])) if df["y"].nunique() == 2 else float("nan"),
        "mean_prob": float(df["p"].mean()),
    }


class MlSignal(Strategy):
    """Baseline: invested whenever the ML probability exceeds a threshold."""

    name = "ml_signal"

    def __init__(self, threshold: float = 0.55) -> None:
        super().__init__(threshold=threshold)

    def _signal(self, bars: pd.DataFrame) -> pd.Series:
        prob = bars[ML_COLUMN] if ML_COLUMN in bars else pd.Series(np.nan, index=bars.index)
        return (prob > self.params["threshold"]).astype(float).where(prob.notna())
