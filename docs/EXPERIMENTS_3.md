# Experiment 3: leveraged trend following vs buy & hold

This protocol was **committed and pushed before any experiment was run**. The code that enforces it is [`src/evotrader/experiments3.py`](../src/evotrader/experiments3.py) and [`src/evotrader/trend.py`](../src/evotrader/trend.py).

## Where the idea comes from

**Research first, not trial and error.** Two findings shaped the choice of what to test:
- Published market anomalies earn [26% less out-of-sample and 58% less after publication](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623) (McLean & Pontiff).
- Popular rules such as [dual momentum lagged even plain SPY after publication](https://quant4free.com/analysis/dual-momentum/).

So this experiment tests a strategy that has both published evidence and a mechanism that addresses the failure we diagnosed ourselves.

**The strategy:** [*Leverage for the Long Run*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2741701) (Gayed & Bilello, 2016, Charles H. Dow Award). Hold stocks *with leverage* when they are above their 200-day moving average, and T-bills when they are below. Over 1928–2015 on the S&P 500, a 2× version reached a Sharpe of 0.51 vs 0.30 for buy & hold. An [independent replication](https://www.cxoadvisory.com/volatility-effects/leveraging-the-u-s-stock-market-based-on-sma-rules/) confirmed the pattern, but warned that:
- switching costs were ignored;
- the sample starts just before the Great Depression, which heavily favours trend timing.

**Why it might succeed where experiment 2 failed.** Volatility targeting stayed levered through *slow, grinding* declines (2001–02, 2011, 2015, 2018, 2022), because volatility never spiked. A trend filter exits exactly those: a grinding decline pushes the price below its 200-day average.

**Honest caveat.** That diagnosis came from experiment 2's final-test data. So this experiment is judged only on markets **no part of this project has ever used**.

## Rules

- Each ETF gets an equal slot.
  - **Above its 200-day average** at the close: the slot is held at the arm's leverage.
  - **Below:** the slot sits in T-bills, earning the 3-month T-bill rate (as in the paper).
- Borrowed money costs the T-bill rate + 1% a year.
- Trading costs are 5 bps per side on every change.
- Decide at the close, trade at the next open.
- **Nothing is tuned.** The 200-day average is the paper's headline setting, and 1.5× is your leverage limit. Each arm is exactly one trial.

| Arm | What | Purpose |
|---|---|---|
| **L1** | 200-day trend filter, no leverage | diagnostic: the classic trend rule |
| **L2** | 200-day trend filter, 1.5× above the average | the paper's idea, within our leverage limit |
| **L3** | trend filter plus experiment 2's volatility targeting (textbook settings), up to 1.5× | does combining the two help? |

## Stages

| Stage | Universe | Evaluated | Status |
|---|---|---|---|
| **dev** | the 10 US ETFs used throughout, data cut at 2021-12-31 | 2007–2021 | development / sanity checks |
| **final-sectors** | XLI, XLP, XLU, XLY, XLB: **never used** by this project | 2000-01-01 to today | judged, runs **once** |
| **final-countries** | EWW, EWT, EWY, EWS, EWH, EWP, EWQ, EWI, EWL, EWN, EWD, EWK, EWO: **never used** | 2001-06-01 to today | judged, runs **once** |
| post-publication | the 10 US ETFs since the paper was published | 2016-12-01 to today | **reported, not judged**: this data was already seen in experiment 1 |

**Limitation.** The finals overlap the development period in time, and markets are correlated. They are new *assets*, not new *history*. Live paper trading remains the only fully clean test.

## Scoring and pass rule (fixed in advance)

- **CAGR** is measured on raw returns.
- **Sharpe** is measured on returns *in excess of the T-bill rate*, for every arm and for buy & hold. Otherwise, sitting in interest-earning T-bills would flatter the strategy.
- An arm **beats buy & hold** only if its **CAGR and Sharpe are both higher** than equal-weight buy & hold's in **dev and both final stages**.
- All results are reported, pass or fail.

## Trial count

- **Before this experiment:** the original bot; experiment 1's 3 arms; experiment 2's 3 arms (27 settings behind V2).
- **This experiment:** 3 arms, no tuning.
- **Pre-registered passes so far:** one, experiment 1's arm B, by a negligible margin.

## Results

*Filled in after each stage runs. The numbers must come from `reports/experiments3/`.*

### Development
_pending_

### Final: US sectors (never used)
_pending_

### Final: countries (never used)
_pending_

### Post-publication (not judged)
_pending_
