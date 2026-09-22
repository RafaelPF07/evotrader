import numpy as np
import pandas as pd
import pytest


def make_bars(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic random-walk OHLCV bars, so tests never need the network."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    index = pd.bdate_range("2015-01-01", periods=n, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000},
        index=index,
    )


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()
