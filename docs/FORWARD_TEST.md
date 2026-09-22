# Forward test: live paper trading from 22 September 2026

This plan was **committed before the trend account made its first decision**. Every trading day from here is data that no strategy, experiment or person in this project has seen. That makes the forward test the only fully clean test left.

## Why

Three pre-registered experiments ([1](EXPERIMENTS.md), [2](EXPERIMENTS_2.md), [3](EXPERIMENTS_3.md)) tested 36 configurations on historical data. Every further idea tested on the same history makes a lucky "win" more likely. The [significance report](../reports/significance.md) quantifies this. Instead of trying more ideas on the past, two strategies now run live, and the future judges them.

## The two accounts

| Account | File | Strategy | Changes during the test? |
|---|---|---|---|
| **Evolved bot** | `data/paper.db` | GA-evolved rules scored against buy & hold (experiment 1's only pass). Re-learns quarterly and promotes a challenger only on held-out evidence. | Yes, but only through its own pre-defined re-learning process |
| **Trend + leverage** | `data/paper_trend.db` | Experiment 3's L2. Each of the 10 US ETFs is held at **1.5×** while it closes above its **200-day average**, cash otherwise. Cash earns the T-bill rate; borrowing costs T-bill + 1%. | **No.** A fixed rule. |

Both trade the same 10 US ETFs with $100,000 of paper money:
- trading costs are 5 bps per side;
- orders are decided at the close and filled at the next open;
- they run automatically every weekday at 22:30 UK time (`scripts/daily.ps1`).

The evolved bot's account was replayed from January 2025. Only its days **after 22 September 2026** count for the forward test.

## Benchmark and checkpoints

- **Benchmark:** equal-weight buy & hold of the same 10 ETFs over the same days.
- **Checkpoints:** **6 months** (late March 2027), **12 months** (late September 2027) and **24 months** (late September 2028).
- **At each checkpoint, for each account, report:**
  - CAGR, Sharpe over T-bills, max drawdown;
  - the same for buy & hold;
  - the probabilistic and deflated Sharpe ratio of the account's active returns (strategy minus buy & hold), with **N = 38** trials (36 historical + these 2).

## What counts as beating buy & hold

At the **24-month** checkpoint, an account beats buy & hold only if:
1. its CAGR **and** Sharpe (over T-bills) are both higher than buy & hold's, **and**
2. the deflated Sharpe ratio of its active returns is **above 0.95**.

The 6- and 12-month checkpoints are **progress reports only**. No conclusion is drawn from them.

## Rules for the test itself

- **No changes during the test.** Changing either strategy's code or settings **ends that account's forward test and restarts the clock**. Bug fixes that don't change decisions (e.g. reporting) are allowed and must be noted here.
- **Every checkpoint gets reported,** good or bad.
- **What to expect.** Two years is short. Roughly how long a genuine edge over buy & hold takes to reach 95% confidence:

  | Edge (annual Sharpe of active returns) | Ignoring the trials | Corrected for 38 trials |
  |---|---|---|
  | 0.3 (a good real-world edge) | ~30 years | ~160 years |
  | 0.5 | ~11 years | ~58 years |
  | 1.0 (exceptional) | ~3 years | ~15 years |

  Unless an edge is enormous, the likeliest 24-month outcome is "not enough evidence either way". That is an honest result, and far better than a false positive. It is also why this project cannot, and does not, claim to beat the market.

## Checkpoint log

| Date | Evolved bot vs B&H | Trend vs B&H | Notes |
|---|---|---|---|
| 2026-09-22 | start | start | plan committed; trend account created |
