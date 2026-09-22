# Experiments: can the bot beat buy & hold?

This protocol was **committed before any experiment was run**; the git history shows the order. The code that enforces it is [`src/evotrader/experiments.py`](../src/evotrader/experiments.py).

## Why a protocol

The 2016–2026 results in the README have already been seen. If we tried ideas until one beat buy & hold on those years, the "winner" would most likely be luck: the more ideas you try, the more likely one looks good by chance. So:

- **Development** uses only data up to **2021-12-31**. Later bars are physically removed before any code runs.
- The **pass rule** is fixed here, in advance.
- The **holdout**, 2022-01-01 to today, is run **once** for every arm. All results are reported, pass or fail. The code refuses to run the holdout twice.

## Hypotheses

The original bot (arm A) loses to buy & hold mainly because:
1. **Its fitness never measured beating buy & hold.** It maximised its own Sharpe ratio.
2. **It sits in cash ~25% of the time.** In a rising market, every day out costs return.

| Arm | Change | Search space | Fitness |
|---|---|---|---|
| **A** | none (reference) | in/out timing rules per ETF | Sharpe (original) |
| **B** | idea 1 | in/out timing rules per ETF | information ratio of returns vs buying & holding that ETF |
| **C** | idea 2 | rotation: every *k* days rank the ETFs by an evolved weighted score and hold the top *N*, always fully invested | portfolio Sharpe |
| **D** | ideas 1 + 2 | rotation | information ratio vs equal-weight buy & hold |

**Fixed for all arms:**
- The 10 ETFs of the default universe.
- 5 bps costs per side.
- No ML signal. It showed no predictive power, and dropping it makes runs ~3× faster.
- Population 60, 25 generations.
- Seeds 0–4.

## Development stage (data ≤ 2021)

- **Folds:** expanding training window from 2007, with 2-year test windows 2012–13, 2014–15, 2016–17, 2018–19 and 2020–21.
- **Champion selection:** each fold picks its champion on the last 25% of its training window.
- **Metric:** Sharpe ratio of the stitched out-of-sample equal-weight portfolio minus the Sharpe of equal-weight buy & hold over the same days.

**Pass rule:** an arm passes if its **mean excess Sharpe across the 5 seeds is > 0** and **at least 4 of 5 seeds** have excess Sharpe > 0.

## Holdout stage (run once)

- **Training:** 2007 to 2021, with the champion chosen on 2018–2021.
- **Test:** 2022-01-01 to the latest completed trading day, 5 seeds.
- **What we claim:** an arm "beats buy & hold" only if it passed development **and** its holdout mean excess Sharpe is > 0 with at least 4 of 5 seeds positive. Everything else is reported as not beating buy & hold.

## Trial count

This matters for interpreting any success.
- **Trials before this protocol:** the original walk-forward (1 configuration, run at 1 then 5 seeds).
- **Trials in this protocol:** 3 new arms (B, C, D). Arm A is the reference.

Any future idea is a new arm and adds a trial.

## Results

*Filled in after each stage runs. The numbers must come from `reports/experiments/`.*

### Development

_pending_

### Holdout

_pending_
