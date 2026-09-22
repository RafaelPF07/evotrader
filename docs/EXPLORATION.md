# Exploration log: iterating on historical data, honestly

**Goal:** find a strategy that beats equal-weight buy & hold on **return and Sharpe** without being buy & hold in disguise. It must be **out of stocks at least 20% of the time**. Evolution gets every free building block available:
- price indicators;
- fear (VIX, VIX term structure, VVIX, SKEW);
- interest rates and the yield curve;
- credit (HYG/IEF) and inflation (TIP/IEF);
- the dollar, oil, the copper/gold ratio;
- market internals (breadth, small vs large, cyclicals vs defensives, emerging markets vs US);
- each ETF's strength vs SPY;
- calendar effects.

## How this stays honest

- **This is exploration, not proof.** The 2007–2026 history has been used many times. Every result below is labelled exploration.
- **Every round is a trial.** Rounds are logged in [`reports/trials.csv`](../reports/trials.csv), and each deflated Sharpe ratio is computed against the running total, which started at 36 from the pre-registered experiments.
- **Each round is a proper walk-forward:**
  - test windows 2012–2026, each unseen by the evolution that produced its rules;
  - several random seeds, combined equally into one "ensemble".
- **Frozen data.** From round 4, rounds run on a frozen, fingerprinted snapshot (`evotrader snapshot`). An automatic re-download of prices between rounds 2 and 3 showed again that the genetic algorithm is chaotic in tiny data revisions: seeds 0 and 1 gave different results with identical settings.
- **The real judge** of anything promising is the live forward test ([FORWARD_TEST.md](FORWARD_TEST.md)).

## Rounds

All rounds: 10 US ETFs, walk-forward 2012-01 to 2026-09, costs 5 bps per side, at least 20% of the time out of stocks. Buy & hold over the same days: **CAGR 13.2%, Sharpe 0.97, max drawdown -30.1%**.

| Trial | Round | Idea | CAGR | Sharpe | Max DD | In stocks | Corr. w/ B&H | DSR |
|---|---|---|---|---|---|---|---|---|
| 37 | r1 | in/out per ETF, all building blocks, "beat B&H" objective | 8.6% | 0.94 | -22.7% | 59% | 0.94 | 0.00 |
| 38 | r2 | same, **Sharpe objective** | 7.9% | **1.02** | **-12.9%** | 58% | 0.79 | 0.00 |
| 39 | r3 | r2 with 9 seeds combined | 6.2% | **1.02** | **-10.2%** | 46% | 0.91 | 0.00 |
| 40 | r4 | risk-on stocks / risk-off bonds+gold switch (no idle cash) | 6.7% | 0.62 | -27.2% | 44% | 0.73 | 0.00 |

## What evolution found

- **The copper/gold ratio, repeatedly.** Copper tracks industrial demand and gold tracks fear, so a rising ratio signals "risk-on". Independently evolved champions used it 32 times in r2, 76 times in r3 and 20 times in r4, across 15 separate evolutions. It is a well-known macro indicator, rediscovered from scratch every time: the most robust *pattern* this project has found.
- **Other recurring signals:** yield-curve changes, cyclicals vs defensives, market breadth, and relative strength vs SPY.
- **"Buy only in extreme panic" (r1, seed 0).** Enter when the VIX is above 60 and RSI is low. The VIX passed 60 only twice in this data (2008, 2020), so this is fitting two events: a textbook overfitting warning, not a discovery.
- **Switching into bonds and gold made things worse (r4).** The "safe" basket had its own crash (TLT -31% in 2022) and lagged stocks over the period. Rules also leaned more on weak calendar effects.

## Why nothing beats buy & hold here

With at least 20% of the time out of stocks, beating buy & hold's return requires the time out to dodge *more* losses than it misses gains. 2012–2026 was a long bull market, and no combination of these signals picked the bad periods reliably enough.

The best rounds (r2, r3) reliably **cut risk**:
- they matched or slightly beat buy & hold's Sharpe (1.02 vs 0.97);
- their worst drawdown was a third to half of buy & hold's.

That is the same pattern as all three pre-registered experiments. Leverage cannot close the return gap within the 1.5× limit. A rough estimate for r2's ensemble:

| r2 ensemble | CAGR | Sharpe | Max DD |
|---|---|---|---|
| levered 1.5× | 11.0% | 0.96 | -19% |
| levered 2.0× | 14.0% | 0.92 | -25% |

Borrowing costs eat the Sharpe advantage.

Every round's deflated Sharpe is ~0.00: none of the differences from buy & hold are distinguishable from luck after 40 trials.
