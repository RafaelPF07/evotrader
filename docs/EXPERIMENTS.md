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

All numbers come from `reports/experiments/{dev,holdout}_{A,B,C,D}.csv`: means over seeds 0 to 4, equal-weight portfolio across the 10 ETFs, 5 bps costs. Buy & hold is equal-weight over the same days.

### Development (test windows 2012–2021)

| Arm | Sharpe | B&H Sharpe | Mean excess Sharpe | Seeds beating B&H | CAGR vs B&H | Max drawdown vs B&H | Rule |
|---|---|---|---|---|---|---|---|
| A | 0.80 | 1.01 | -0.203 | 1/5 | 8.8% vs 13.7% | -24.9% vs -30.1% | fail |
| **B** | **1.04** | 1.01 | **+0.027** | **5/5** | 13.7% vs 13.7% | -29.9% vs -30.1% | **pass** |
| C | 0.91 | 1.01 | -0.097 | 2/5 | 10.4% vs 13.7% | -22.7% vs -30.1% | fail |
| D | 0.85 | 1.01 | -0.156 | 1/5 | 12.5% vs 13.7% | -30.2% vs -30.1% | fail |

### Holdout (2022-01-01 to 2026-09-21, run once)

| Arm | Sharpe | B&H Sharpe | Mean excess Sharpe | Seeds beating B&H | CAGR vs B&H | Max drawdown vs B&H | Rule |
|---|---|---|---|---|---|---|---|
| A | 0.79 | 0.89 | -0.098 | 0/5 | 8.3% vs 12.2% | -15.5% vs -18.5% | fail |
| **B** | **0.90** | 0.89 | **+0.014** | **4/5** | 12.3% vs 12.2% | -18.3% vs -18.5% | **pass** |
| C | 0.52 | 0.89 | -0.370 | 0/5 | 5.7% vs 12.2% | -20.2% vs -18.5% | fail |
| D | 0.53 | 0.89 | -0.353 | 1/5 | 7.7% vs 12.2% | -24.7% vs -18.5% | fail |

### Verdict

**Arm B (idea 1) formally beats buy & hold under the pre-registered rule, in both stages. The margin is economically negligible.**

- **What B learned.** It is invested 97–99.8% of the time. Its rules amount to "hold, but step out briefly after an unusually sharp short-term spike", e.g. `ENTRY trend(20) < 0.086 | EXIT trend(10) > 0.065`. Scoring against buy & hold taught the search that the best way not to lose to buy & hold is to *be* buy & hold, plus a small, consistent tweak.
- **The edge is tiny.** Holdout: +0.014 Sharpe and +0.1 percentage points of CAGR per year. Over under five years that is far below what the data can distinguish from luck. It is consistent, but not a meaningful improvement.
- **Idea 1 fixed what it was aimed at.** Arm A under-performed by -0.10 to -0.20 Sharpe because its fitness rewarded sitting in cash. Aligning the fitness with the goal removed that entire loss.

**Idea 2 (rotation) failed, informatively.**
- **Arm C** converged on *"hold the lowest-volatility ETFs"*, e.g. `ROTATE top 2 every 63d by -0.45*vol(63) +0.12*mom(252)`. Checked year by year on the holdout (seed 0), it did its job in the 2022 sell-off, holding mainly gold, healthcare and the Dow ETF: -7.7% vs -10.3% for buy & hold. It then lagged badly in the 2023–2026 rally (e.g. 2024: +4.2% vs +15.8%) because it kept holding calm, defensive ETFs while tech surged. This is the same trade-off that sank arm A: defence pays in crashes and costs in rising markets.
- **Arm D** was unstable across seeds: holdout Sharpe ranged from -0.10 to 0.99. That is the signature of fitting noise.

**What this means for the live bot.** Arm B is a defensible replacement for the current rule. It closes the gap to buy & hold, which arm A never did, without claiming to beat it by a meaningful amount.

**Trial count after this experiment: 4 configurations** (original, plus arms B, C, D). One passed.

## Reproducibility (added after the experiment, 2026-09-22)

The results above are exactly what the pre-registered runs produced. Nothing here re-judges them. However, a later reproduction check (`evotrader significance`) found that **arm B's pass is fragile**.

**What happened.** The price data was re-downloaded from Yahoo after the holdout ran. The re-download changed prices by about **one part in a million**; buy & hold's holdout Sharpe moved from 0.886731 to 0.886732. Re-running arm B with the same seeds and code on the new data:

| Seed | Recorded holdout Sharpe | Reproduced on re-downloaded data | Beats B&H (0.887)? |
|---|---|---|---|
| 0 | 0.921 | 0.921 | yes |
| 1 | 0.865 | 0.865 | no |
| 2 | 0.894 | 0.914 | yes |
| 3 | 0.903 | 0.903 | yes |
| 4 | **0.922** | **0.750** | **no** |

**The code is not the cause.** Running the exact code that produced the recorded results (commit `6f9a42b`) on the new data gives the same reproduced numbers.

**Why the data change matters so much.** The genetic algorithm is chaotic: a one-in-a-million change can flip a near-tie between two rules early on, and the search then evolves a different champion.

**What it means.** On the re-downloaded data, arm B would have beaten buy & hold in **3 of 5 seeds** with a slightly **negative** mean excess Sharpe, i.e. it would **fail** the pass rule. Its recorded pass should be read as luck of the data snapshot, not as evidence of an edge. The deflated Sharpe ratios in [reports/significance.md](../reports/significance.md) agree: none of its seeds comes close to significance.

**Lesson for future experiments:** freeze and fingerprint the exact data used by any one-time test, so that it can be reproduced bit for bit.
