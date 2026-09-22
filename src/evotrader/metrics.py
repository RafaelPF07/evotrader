"""Performance metrics. All annualisation assumes 252 trading days per year."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall in equity, as a negative fraction."""
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0


def sharpe(returns: pd.Series) -> float:
    std = returns.std()
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0


def information_ratio(active: pd.Series) -> float:
    """Annualised mean / volatility of returns *relative to a benchmark*."""
    std = active.std()
    return float(active.mean() / std * np.sqrt(TRADING_DAYS)) if std > 1e-12 else 0.0


def sortino(returns: pd.Series) -> float:
    downside = np.sqrt((returns.clip(upper=0) ** 2).mean())
    return float(returns.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else 0.0


def cagr(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    total = float((1 + returns).prod())
    years = len(returns) / TRADING_DAYS
    return total ** (1 / years) - 1 if total > 0 else -1.0


def compute_metrics(
    returns: pd.Series, position: pd.Series, trades: pd.DataFrame | None = None
) -> dict[str, float]:
    mdd = max_drawdown(returns)
    growth = cagr(returns)
    closed = trades[~trades["open"]] if trades is not None and len(trades) else pd.DataFrame()
    return {
        "total_return": float((1 + returns).prod() - 1),
        "cagr": growth,
        "volatility": float(returns.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": mdd,
        "calmar": growth / abs(mdd) if mdd < 0 else 0.0,
        "exposure": float(position.mean()),
        "turnover": float(position.diff().abs().sum() / max(len(position) / TRADING_DAYS, 1e-9)),
        "trades": float(len(trades)) if trades is not None else 0.0,
        "win_rate": float((closed["return"] > 0).mean()) if len(closed) else float("nan"),
    }
