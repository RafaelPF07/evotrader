import numpy as np
import pandas as pd
import pytest

from evotrader.ml import MlConfig, MlSignal, build_labels, walk_forward_proba
from tests.conftest import make_bars

FAST = MlConfig(min_train=200, retrain_every=100, max_iter=20)


def test_labels_measure_next_open_to_future_open(bars):
    h = 5
    labels = build_labels(bars, h)
    t = 100
    expected = float(bars["open"].iloc[t + 1 + h] > bars["open"].iloc[t + 1])
    assert labels.iloc[t] == expected
    assert labels.iloc[-(h + 1):].isna().all()


def test_proba_is_causal():
    """Scrambling prices after the cutoff must not change any earlier prediction."""
    bars = make_bars(n=700, seed=4)
    cutoff = 450
    future = bars.copy()
    rng = np.random.default_rng(0)
    future.iloc[cutoff + 1:, :4] *= rng.uniform(0.7, 1.3, size=(len(bars) - cutoff - 1, 1))

    original = walk_forward_proba(bars, FAST)
    scrambled = walk_forward_proba(future, FAST)
    pd.testing.assert_series_equal(original.iloc[: cutoff + 1], scrambled.iloc[: cutoff + 1])
    assert original.iloc[: FAST.min_train].isna().all()  # nothing before enough history
    predicted = original.dropna()
    assert len(predicted) > 200 and predicted.between(0, 1).all()


def test_ml_signal_needs_the_column(bars):
    assert MlSignal().target_position(bars).sum() == 0
    with_prob = bars.assign(ml_prob=0.9)
    assert MlSignal(threshold=0.55).target_position(with_prob).eq(1).all()


@pytest.mark.parametrize("threshold", [0.4, 0.6])
def test_ml_signal_threshold(bars, threshold):
    prob = pd.Series(np.linspace(0, 1, len(bars)), index=bars.index)
    pos = MlSignal(threshold).target_position(bars.assign(ml_prob=prob))
    assert (pos == (prob > threshold)).all()
