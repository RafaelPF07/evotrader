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

All numbers come from `reports/experiments2/{dev,final-us2000,final-intl}.csv`. CAGR and Sharpe are the pass criteria; the other columns are for context.

### Development (US, 2007–2021)

| Arm | CAGR | Sharpe | Volatility | Max drawdown | Avg exposure | Beats B&H? |
|---|---|---|---|---|---|---|
| buy & hold | 10.6% | 0.69 | 16.8% | -43.4% | 100% | |
| V0 (no leverage) | 10.1% | 0.85 | 12.2% | -29.1% | 90% | no (lower CAGR) |
| **V1** (textbook, ≤1.5×) | **12.9%** | **0.88** | 15.2% | -29.4% | 121% | **yes** |
| **V2** (tuned, ≤1.5×) | **14.1%** | **0.93** | 15.4% | -29.8% | 123% | **yes** |

V2's tuning selected lookback 20, scale 1.0 and **band 25%**. It is the textbook setting with a wider no-trade band.

### Final: US 2000–2006 (run once)

| Arm | CAGR | Sharpe | Volatility | Max drawdown | Avg exposure | Beats B&H? |
|---|---|---|---|---|---|---|
| buy & hold | 5.6% | 0.39 | 17.8% | -39.3% | 100% | |
| V0 | 4.9% | 0.37 | 16.4% | -39.4% | 97% | no |
| V1 | 5.7% | 0.37 | 20.7% | **-46.5%** | 130% | no (lower Sharpe) |
| V2 | 5.6% | 0.37 | 20.6% | -46.5% | 128% | no |

### Final: international, 2004 to today (run once)

| Arm | CAGR | Sharpe | Volatility | Max drawdown | Avg exposure | Beats B&H? |
|---|---|---|---|---|---|---|
| buy & hold | 8.3% | 0.47 | 22.3% | -59.9% | 100% | |
| **V0** | 8.3% | **0.55** | 17.3% | **-40.2%** | 93% | yes (by a hair on CAGR) |
| V1 | 7.4% | 0.44 | 21.6% | -44.4% | 123% | no |
| V2 | 8.2% | 0.48 | 21.6% | -42.6% | 121% | no |

### Verdict

**No arm beats buy & hold under the pre-registered rule.** The leveraged versions V1 and V2 passed development convincingly, then failed both final tests.

**Why the development result didn't hold.** A year-by-year breakdown (diagnosis only; nothing was changed or re-run) shows a clear pattern:

- **It works in volatile crashes.** When a crash comes with a volatility spike, exposure is cut hard and early. In 2008: 0.3–0.4× in the US, and international losses of -25% vs -44% for buy & hold. In March 2020: 0.21×.
- **It fails in slow, grinding declines.** When prices fall *without* volatility spiking, the strategy stays levered all the way down:
  - US: 2001 (-16.9% vs -9.2%) and 2002 (-25.3% vs -20.0%);
  - international: 2011, 2014, 2015, 2018 and 2022.

  Leverage amplifies every one of those losses, and borrowing costs add to them.
- **It is most exposed at the start of a crash that begins from calm.** On 2020-02-14, just before COVID, exposure was at the maximum 1.5×. It reacts to turbulence; it cannot predict it.
- **The long-run volatility estimate can mislead.** At the start of the US 2000–06 test it came from the turbulent dot-com crash (25% vs 18% realised afterwards). The strategy therefore judged conditions as calm and stayed levered ~90% of the time.
- **Development flattered the idea.** 2007–2021 happened to be dominated by exactly the kind of crash volatility targeting handles well (2008, 2020). A single train/test split would have declared victory. The two untouched final tests are what caught it.

**What did hold up: volatility targeting as risk control, not return.** Without leverage (V0), it cut the worst drawdown sharply in development (-29% vs -43%) and internationally (-40% vs -60%), at about the same return. In US 2000–06 it roughly matched buy & hold. So it is a defensible way to take *less* risk for similar return, but not a way to *beat* buy & hold.

**Trial count after this experiment:**
- **Configurations tested:** 3 (original bot + experiment 1) + 3 arms here, with 27 settings behind V2.
- **Configurations that beat buy & hold under a pre-registered rule:** one, experiment 1's arm B, by a negligible margin.
