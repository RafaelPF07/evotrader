# Forward test: live paper trading from 22 September 2026

This plan was **committed before the trend account made its first decision**. Every trading day from here is data that no strategy, experiment or person in this project has seen. That makes the forward test the only fully clean test left.

## Why

Three pre-registered experiments ([1](EXPERIMENTS.md), [2](EXPERIMENTS_2.md), [3](EXPERIMENTS_3.md)) tested 36 configurations on historical data. Every further idea tested on the same history makes a lucky "win" more likely. The [significance report](../reports/significance.md) quantifies this. Instead of trying more ideas on the past, two strategies now run live, and the future judges them.

## The accounts

| Account | File | Strategy | Changes during the test? |
|---|---|---|---|
| **Evolved bot** | `data/paper.db` | GA-evolved rules scored against buy & hold (experiment 1's only pass). Re-learns quarterly and promotes a challenger only on held-out evidence. | Yes, but only through its own pre-defined re-learning process |
| **Trend + leverage** | `data/paper_trend.db` | Experiment 3's L2. Each of the 10 US ETFs is held at **1.5×** while it closes above its **200-day average**, cash otherwise. Cash earns the T-bill rate; borrowing costs T-bill + 1%. | **No.** A fixed rule. |

Both trade the same 10 US ETFs with $100,000 of paper money:
- trading costs are 5 bps per side;
- orders are decided at the close and filled at the next open;
- they run automatically every weekday at 22:30 UK time (`scripts/daily.ps1`).

The evolved bot's account was replayed from January 2025. Only its days **after 22 September 2026** count for the forward test.

## Account 3: copper/gold ensemble (added 22 September 2026)

This account was added after the first two, **with its own plan committed before it was created**. It is forward-tested on exactly the same days, rules and benchmark as the others.

- **Strategy.** Exploration round 2's recipe (trial 38, [EXPLORATION.md](EXPLORATION.md)):
  - three evolved in/out rules using price and macro building blocks;
  - each out of stocks at least 20% of the time;
  - trained once on 2007-01-01..2021-10-17 and chosen on 2021-10-17..2026-09-22, using the frozen data snapshot `2026-09-22` (fingerprint `a5c9f001…`).

  The rules are frozen in [`models/forward_copper_gold.json`](../models/forward_copper_gold.json):

  | Rule | Entry | Exit |
  |---|---|---|
  | 0 | `credit_mom(20) < -0.010` | `copper_gold_mom(63) < -0.128` |
  | 1 | `copper_gold_mom(63) > 0.086` | `copper_gold_mom(63) < -0.034` |
  | 2 | `tom > 0.6` | `(curve_chg(126) > -0.25 AND ma_spread(20,200) < 0.013)` |

- **Sizing.** Each ETF holds *k*/3 of its equal slot when *k* of the 3 rules say "in". No leverage; cash earns nothing.
- **File:** `data/paper_ensemble.db`. The rules never change during the test.

**Its claim is risk reduction, not beating the market.** It is judged at 24 months on two pre-registered questions:
1. **Risk reduction.** Its max drawdown is **at most 60%** of buy & hold's, **and** its Sharpe (over T-bills) is **no more than 0.10 below** buy & hold's. This is the claim the exploration supports.
2. **Beating buy & hold.** Same test as the other accounts: higher CAGR and Sharpe, and a deflated Sharpe above 0.95. Here N = **43**: the 40 trials in the ledger when it was created, plus the 3 forward-test accounts. The exploration expects this to **fail**.

If markets never fall much during the 24 months, question 1 can't really be tested (drawdowns will be small for everyone). That will be reported as inconclusive, not as a pass.

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

| Date | Evolved bot vs B&H | Trend vs B&H | Copper/gold vs B&H | Notes |
|---|---|---|---|---|
| 2026-09-22 | start | start | - | Plan committed (`086bbc6`), then the trend account was created after the US close. First decisions at the 22 Sep close: 8 of 10 ETFs above their 200-day average (all but TLT and GLD), so the account is 120% invested from the 23 Sep open. |
