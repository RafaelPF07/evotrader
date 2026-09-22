# Walk-forward results

Every number below is **out-of-sample**: the rules were evolved and selected
without ever seeing the test period. Equal-weight portfolio across the universe,
costs included.

## Stitched out-of-sample record

| strategy | CAGR | Sharpe | max drawdown | exposure |
|---|---|---|---|---|
| evolved | +8.1% | 0.80 | -21.2% | 73% |
| buy_and_hold | +14.0% | 0.96 | -30.1% | 100% |
| sma_cross | +8.9% | 0.84 | -23.7% | 74% |
| momentum | +8.1% | 0.91 | -14.4% | 72% |
| ml_signal | +5.5% | 0.60 | -23.3% | 54% |

## Per fold (test Sharpe)

| fold | test period | evolved | buy_and_hold | sma_cross | momentum | ml_signal |
|---|---|---|---|---|---|---|
| 0 | 2016-01-01 .. 2017-12-31 | 1.91 | 1.81 | 1.55 | 2.06 | 1.33 |
| 1 | 2018-01-01 .. 2019-12-31 | 0.56 | 0.84 | 0.54 | 0.40 | 0.06 |
| 2 | 2020-01-01 .. 2021-12-31 | 0.26 | 0.94 | 0.85 | 1.07 | 0.67 |
| 3 | 2022-01-01 .. 2023-12-31 | 0.62 | 0.30 | 0.19 | 0.00 | 0.14 |
| 4 | 2024-01-01 .. 2026-09-21 | 1.08 | 1.44 | 1.21 | 1.20 | 1.08 |

## Champion rules

**Fold 0** (train fitness +0.573, validation +0.371)

`ENTRY zscore(20) < -1.12 | EXIT (vol(10) > 0.30 AND rsi(5) > 33.0)`

**Fold 1** (train fitness +0.661, validation +0.507)

`ENTRY zscore(10) < -1.73 | EXIT (vol(10) > 0.30 AND rsi(5) > 32.8)`

**Fold 2** (train fitness +0.616, validation +0.473)

`ENTRY zscore(50) < -1.14 | EXIT (vol(20) > 0.26 AND rsi(5) > 32.8)`

**Fold 3** (train fitness +0.636, validation +0.061)

`ENTRY zscore(50) < -1.15 | EXIT (vol(20) > 0.26 AND rsi(5) > 32.2)`

**Fold 4** (train fitness +0.591, validation +0.169)

`ENTRY zscore(50) < -1.19 | EXIT (vol(20) > 0.31 AND rsi(5) > 32.2)`

## Robustness across random seeds

The whole walk-forward was repeated with 5 seeds. The tables above show the first seed, not the best one.

| seed | CAGR | Sharpe | max drawdown | periods beating buy & hold |
|---|---|---|---|---|
| 0 | +8.1% | 0.80 | -21.2% | 2 |
| 1 | +8.2% | 0.74 | -30.0% | 1 |
| 2 | +10.2% | 0.84 | -27.1% | 0 |
| 3 | +9.8% | 0.92 | -24.6% | 3 |
| 4 | +7.6% | 0.74 | -24.9% | 0 |
| **mean** | +8.8% | 0.81 | -25.6% | 1.2 |
| buy & hold | +14.0% | 0.96 | -30.1% | - |
