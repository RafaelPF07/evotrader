# Significance: do any of the wins survive a correction for trying 36 things?

Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) of each candidate's *active* returns (strategy minus equal-weight buy & hold) on the period it was judged on.
- **active Sharpe**: annualised Sharpe of beating buy & hold
- **PSR**: probability the edge is real, ignoring how many things were tried
- **DSR**: the same, after accounting for N trials. Above 0.95 means significant.

| candidate | stage | days | active Sharpe | luck threshold (N=36) | PSR | DSR N=10 | DSR N=36 |
|---|---|---|---|---|---|---|---|
| exp1 B seed 0 | holdout 2022-26 | 1183 | +0.63 | 0.99 | 0.96 | 0.39 | 0.16 |
| exp1 B seed 1 | holdout 2022-26 | 1183 | -0.66 | 0.99 | 0.07 | 0.00 | 0.00 |
| exp1 B seed 2 | holdout 2022-26 | 1183 | +0.02 | 0.99 | 0.52 | 0.06 | 0.02 |
| exp1 B seed 3 | holdout 2022-26 | 1183 | +0.33 | 0.99 | 0.79 | 0.17 | 0.05 |
| exp1 B seed 4 | holdout 2022-26 | 1183 | -0.88 | 0.99 | 0.03 | 0.00 | 0.00 |
| exp2 V1 | dev | 3777 | +0.22 | 0.55 | 0.80 | 0.23 | 0.10 |
| exp2 V2 | dev | 3777 | +0.35 | 0.55 | 0.91 | 0.41 | 0.21 |
| exp2 V1 | final-us2000 | 1403 | +0.12 | 0.91 | 0.61 | 0.10 | 0.03 |
| exp2 V2 | final-us2000 | 1403 | +0.10 | 0.91 | 0.60 | 0.09 | 0.03 |
| exp2 V1 | final-intl | 5612 | -0.09 | 0.46 | 0.33 | 0.02 | 0.00 |
| exp2 V2 | final-intl | 5612 | -0.02 | 0.46 | 0.47 | 0.05 | 0.01 |
| exp3 L2 | dev | 3777 | -0.04 | 0.55 | 0.43 | 0.04 | 0.01 |
| exp3 L2 | final-sectors | 6719 | -0.29 | 0.42 | 0.07 | 0.00 | 0.00 |
| exp3 L2 | final-countries | 6363 | +0.02 | 0.43 | 0.54 | 0.07 | 0.02 |

## Reproduction check (experiment 1)

Experiment 1's evolved strategies were re-run with the same seeds and code. Seeds whose Sharpe differs from the recorded one were affected by a re-download of the price data that changed prices by about one part in a million; the original code gives the same differences on today's data, so the code is not the cause (see docs/EXPERIMENTS.md, 'Reproducibility'). The rows above use the reproduced series.

| seed | reproduced minus recorded Sharpe |
|---|---|
| exp1 B seed 0 | +0.0000 |
| exp1 B seed 1 | +0.0000 |
| exp1 B seed 2 | +0.0204 |
| exp1 B seed 3 | +0.0006 |
| exp1 B seed 4 | -0.1728 |
