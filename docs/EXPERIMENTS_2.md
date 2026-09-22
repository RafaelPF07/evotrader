# Experiment 2: can volatility targeting beat buy & hold?

This protocol was **committed and pushed before any experiment was run**; the git history and GitHub's timestamps show the order. The code that enforces it is [`src/evotrader/experiments2.py`](../src/evotrader/experiments2.py) and [`src/evotrader/sizing.py`](../src/evotrader/sizing.py).

## Why this experiment

[Experiment 1](EXPERIMENTS.md) showed that price-based *timing* rules can't beat buy & hold here. The best one learned to *be* buy & hold, invested ~99% of the time. Forbidding that would only force bets with no evidence behind them.

So this experiment tests a different, well-documented route: **volatility targeting**.
- Hold *more* when markets are calm and *less* when they are turbulent. Crashes tend to arrive in turbulent periods, so this should cut risk more than it cuts return.
- Then use the spare risk budget for **modest leverage** (at most 1.5×), aiming for a higher return than buy & hold at similar risk.

## The strategy

Each day, the equal-weight basket of ETFs is held at:

    exposure = target_scale × long-run volatility ÷ recent volatility,   capped at [0, cap]

| Setting | Meaning |
|---|---|
| *recent volatility* | standard deviation of daily basket returns over the last `lookback` days |
| *long-run volatility* | the same, over all history up to that day (at least 252 days) |
| no-trade band | exposure changes only when the target moves by more than `band`, which limits trading costs |
| costs | 5 bps per side on every change in exposure |
| borrowing | levered money costs the 3-month T-bill rate + 1% a year, charged daily |
| cash | earns nothing. This is deliberately conservative, since real brokers pay interest on cash |
| timing | decide at the close, trade at the next open (as everywhere in this project) |

## Arms

| Arm | Settings | Purpose |
|---|---|---|
| **V0** | textbook: lookback 20, scale 1.0, band 10%, **cap 1.0** | diagnostic: how much comes from leverage? |
| **V1** | textbook: lookback 20, scale 1.0, band 10%, **cap 1.5** | the main hypothesis |
| **V2** | tuned on the development stage only: highest Sharpe among 27 settings (lookback 10/20/63 × scale 0.8/1.0/1.2 × band 0/10/25%), cap 1.5 | does tuning add anything? |

## Stages

| Stage | Universe | Data | Evaluated | Why |
|---|---|---|---|---|
| **dev** | the 10 US ETFs used everywhere else | cut at 2021-12-31 | 2007–2021 | development; V2's settings are chosen here |
| **final-us2000** | the 8 of those ETFs that existed in 2000 (no TLT, no GLD) | cut at 2006-12-31 | 2001-06-01 to 2006-12-31 | a period **no part of this project has ever used**, including the 2001–02 bear market |
| **final-intl** | 8 international ETFs never used by this project: EFA, EEM, EWJ, EWG, EWU, EWC, EWA, EWZ | full | 2004-06-01 to today | different markets entirely |

**Honest limitations of the final tests:**
- **final-us2000 is earlier in time** than the development data. It is untouched, but not "the future".
- **final-intl overlaps the development period in time.** Global markets are correlated, so it is independent in assets but not in history.

Passing both is still far stronger evidence than either alone. The truly clean test remains live paper trading.

## Pass rule (fixed in advance)

An arm **beats buy & hold** only if its **CAGR and Sharpe ratio are both higher** than equal-weight buy & hold's, in **dev and in both final stages**. That means more return *and* more return per unit of risk, everywhere.

Each final stage runs **once**. The code refuses to run it again. All results are reported.

## Trial count

- **Before:** the original bot, plus 3 arms in experiment 1.
- **This experiment:** V0 and V1 are one setting each. V2 is chosen from **27 settings**, so its dev results are optimistic by construction. That is exactly why it must also pass the final tests, which it never saw.

## Results

*Filled in after each stage runs. The numbers must come from `reports/experiments2/`.*

### Development
_pending_

### Final: US 2000–2006
_pending_

### Final: international
_pending_
