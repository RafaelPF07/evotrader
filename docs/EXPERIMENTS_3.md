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

All numbers come from `reports/experiments3/`. CAGR is raw; Sharpe is over T-bills.

| Stage | Buy & hold | L1 trend | **L2 trend + 1.5×** | L3 trend + vol target |
|---|---|---|---|---|
| **dev** (US, 2007–21) | 10.6% / 0.64 / -43% | 7.6% / 0.75 / -15% | 10.5% / 0.74 / -21% ❌ | 9.3% / 0.75 / -17% |
| **final-sectors** (never used) | 8.9% / 0.47 / -49% | 4.7% / 0.32 / -27% | 5.4% / 0.30 / -40% ❌ | 5.4% / 0.33 / -27% |
| **final-countries** (never used) | 9.3% / 0.44 / -64% | 8.1% / 0.57 / -22% | **10.6% / 0.56 / -31%** ✅ | 8.6% / 0.51 / -31% |
| post-publication (US, not judged) | 14.1% / 0.79 / -30% | 9.7% / 0.80 / -13% | 12.8% / 0.78 / -20% | 11.3% / 0.78 / -14% |

*Each cell: CAGR / Sharpe / max drawdown. ✅/❌ = beats buy & hold on both CAGR and Sharpe.*

### Verdict

**No arm beats buy & hold under the pre-registered rule.**
- **L2 is closest.** It won on the never-used country ETFs (higher return and Sharpe, with half the drawdown). It missed dev by 0.1 point of CAGR. It lost clearly on the never-used US sectors.
- **One win out of three stages is not evidence.** Picking the stage where it worked would be cherry-picking.

**What the per-ETF breakdown shows** (diagnosis only; nothing was changed or re-run):

- **The trend filter cut the maximum drawdown on all 18 never-used ETFs,** often dramatically. Korea (EWY) went from -74% to -41%, Denmark (EWK) from -74% to -26%, industrials (XLI) from -62% to -28%.
- **It lowered return on 13 of 18,** by missing rebounds and whipsawing around the average. The worst case was consumer staples (XLP): 7.4% → 2.2% a year.
- **Leverage converts lower risk into higher return only where crashes were very deep.** The country basket fell 64% under buy & hold, and sidestepping that more than paid for the missed rebounds. The US sectors, especially defensive ones like staples and utilities, had shallower falls to avoid, so leverage could not recover the lost return.
- **After the paper was published** (US 2016–26), L2 earned less than buy & hold (12.8% vs 14.1%) at about the same Sharpe. This is consistent with the post-publication decay in the literature.

### What three experiments add up to

Across every approach tested (evolved timing rules, rotation, volatility targeting, trend filters, with and without leverage), one pattern repeats:

1. **Risk reduction is real and robust.** Every defensive rule cut maximum drawdowns substantially on data it had never seen.
2. **Return enhancement is not.** Turning that lower risk into higher return than buy & hold, through leverage, worked in some markets and periods and failed in others. It never passed a pre-registered test.

That is what the academic literature predicts for publicly known, price-based strategies on liquid markets.

**Trial count after this experiment:** original bot + 3 + 3 + 3 arms (plus V2's 27 settings). One pre-registered pass: experiment 1's arm B, by a negligible margin.
