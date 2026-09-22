# Walk-forward results

Every number below is **out-of-sample**: the rules were evolved and selected
without ever seeing the test period. Equal-weight portfolio across the universe,
costs included.

## Stitched out-of-sample record

| strategy | CAGR | Sharpe | max drawdown | exposure |
|---|---|---|---|---|
| evolved | +8.2% | 0.81 | -15.5% | 78% |
| buy_and_hold | +14.0% | 0.96 | -30.1% | 100% |
| sma_cross | +8.9% | 0.84 | -23.7% | 74% |
| momentum | +8.1% | 0.91 | -14.4% | 72% |
| ml_signal | +5.3% | 0.58 | -24.7% | 55% |

## Per fold (test Sharpe)

| fold | test period | evolved | buy_and_hold | sma_cross | momentum | ml_signal |
|---|---|---|---|---|---|---|
| 0 | 2016-01-01 .. 2017-12-31 | 1.26 | 1.81 | 1.55 | 2.06 | 1.27 |
| 1 | 2018-01-01 .. 2019-12-31 | 0.59 | 0.84 | 0.54 | 0.40 | 0.07 |
| 2 | 2020-01-01 .. 2021-12-31 | 0.98 | 0.94 | 0.85 | 1.07 | 0.59 |
| 3 | 2022-01-01 .. 2023-12-31 | 0.29 | 0.30 | 0.19 | 0.00 | 0.24 |
| 4 | 2024-01-01 .. 2026-09-22 | 1.11 | 1.44 | 1.20 | 1.19 | 1.01 |

## Champion rules

**Fold 0** (train fitness +0.572, validation -0.096)

`ENTRY zscore(50) < 1.24 | EXIT (zscore(100) < 0.30 AND zscore(10) > -0.57)`

**Fold 1** (train fitness +0.653, validation +0.403)

`ENTRY zscore(10) < -0.54 | EXIT (vol(10) > 0.28 AND zscore(10) > -1.35)`

**Fold 2** (train fitness +0.582, validation +0.442)

`ENTRY zscore(10) < -0.58 | EXIT (vol(10) > 0.28 AND zscore(10) > -1.35)`

**Fold 3** (train fitness +0.600, validation +0.346)

`ENTRY zscore(10) < -0.52 | EXIT (vol(10) > 0.29 AND zscore(10) > -1.35)`

**Fold 4** (train fitness +0.568, validation +0.334)

`ENTRY trend(20) < -0.030 | EXIT (vol(10) > 0.30 AND zscore(10) > -1.80)`
