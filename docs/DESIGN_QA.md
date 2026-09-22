# Design Q&A

The questions this project most often raises, with answers that point to the code.

---

## "Walk me through the project in 60 seconds."

EvoTrader is a paper trading bot for 10 US ETFs that invents its own strategies. A strategy is a pair of boolean rule trees (entry and exit) over normalised indicators. A genetic algorithm evolves a population of them, scored on all ten ETFs at once, so it rewards rules that generalise. I evaluated the learning process with walk-forward testing over 2016 to 2026, and I report the result honestly. Across 5 random seeds it did not beat buy & hold (Sharpe 0.81 vs 0.96). It matched classic baselines like a 50/200 moving-average crossover, with slightly smaller drawdowns. Then I built a daily paper-trading loop around it. The loop keeps a SQLite journal explaining every trade, analyses losing trades for patterns, and re-evolves quarterly. A new strategy is promoted only if it wins on a held-out year.

## "Did it make money?"

It grew, but the honest benchmark is buy & hold, and it **did not beat that** on return or Sharpe in any of 5 seeds. Its drawdowns were slightly smaller on average (-26% vs -30%), with about a quarter less market exposure. I didn't tune anything after seeing test results, because that turns the test set into a training set. I also re-ran the whole evaluation with 5 random seeds and report the spread, because one GA run is one sample of a random process.

## How do you know there's no look-ahead bias? → `backtest.py`, `tests/test_backtest.py`

- **Execution timing.** A strategy decides at the close of day *t*, and the order fills at the open of *t+1*. The overnight gap into *t+1* still belongs to the old position.
- **Tests, not promises.** `test_no_lookahead` multiplies all prices after a cutoff by random noise, re-runs every strategy, and asserts that positions and returns up to the cutoff are *identical*. The same test style covers the ML model (`test_proba_is_causal`) and the whole paper account (`test_no_future_information_in_replay`, which compares orders, fills, equity and learning decisions).
- **A live-only leak I found and fixed.** During market hours Yahoo returns a partial bar for today, whose "close" is just the latest price. `data.drop_incomplete` removes it until the session ends.

## What is walk-forward validation and why not a normal train/test split? → `walkforward.py`

Markets change over time, and a single split tests only one regime. With walk-forward I train on 2007 up to year *N*, test on *N* to *N+2*, then expand the window and repeat: 2016, 2018, 2020, 2022, 2024. Inside each training window, the last 25% is a **validation** slice used only to pick the champion from the hall of fame. The test years are touched exactly once. Stitching the test windows gives one continuous out-of-sample track record.

## What does "purging" mean in your ML model? → `ml.py`

The label for day *j* is "is the price up 5 days after the next open?", so it's only known at day *j+6*. When the model predicts at day *t*, it may only train on rows up to *t−6*. Otherwise training labels overlap the period being predicted. It's easy to get wrong and silently inflates accuracy.

## Why is the ML signal so weak? Isn't that a failure?

Out-of-sample AUC is 0.50–0.52, essentially a coin flip, and accuracy is below the base rate. That is the normal result for predicting daily direction from price-derived features; if it were much higher, I'd suspect a leak. The GA was free to use it as a building block and never chose it. The negative result is a finding, and I report it.

## How does the genetic algorithm work? → `evolution/genome.py`, `evolution/engine.py`

- **Representation:** trees of `Compare(feature, <|>, threshold)` leaves joined by AND/OR/NOT, with separate entry and exit trees.
- **Selection:** tournament of 3, plus elitism (the top 4 survive unchanged, so the best fitness never decreases; there's a test).
- **Crossover:** graft a random subtree from parent B into a random spot in parent A.
- **Mutation:** nudge a threshold, change a window, flip a comparison, grow or shrink the tree, or replace a subtree.
- **Diversity:** 10% random immigrants each generation.
- **Bloat control:** a max depth, plus a complexity penalty in fitness.
- **Speed:** each feature is computed once per ticker (`FeatureStore`) and fitness is cached by rule string, so scoring a rule across 10 ETFs takes about 25 ms.

## Why is fitness "mean Sharpe minus 0.5 × std across tickers"? → `evolution/fitness.py`

A rule that works brilliantly on one ETF and badly on nine is probably luck. Penalising the spread across tickers rewards consistency. Three other penalties also apply: rule size (Occam's razor against overfitting), fewer than ~1 trade a year (Sharpe on 3 trades is meaningless), and almost never being invested.

## Why were results different when you re-ran it?

The rule *family* is stable: buy when price is well below its average, exit when volatility spikes. The exact thresholds and per-fold results vary between runs, though. Yahoo re-adjusts historical prices for dividends, and a GA's search path is chaotic, so tiny input changes lead to different final rules. That's why I added a multi-seed robustness run and report its spread rather than one number. A single earlier run had shown the drawdown cut in half, and the seed study showed that was luck, so I replaced the claim. It's also why the live bot demands a margin before swapping strategies.

## How does the bot "learn from its mistakes"? → `paper/learner.py`

Every 63 trading days:
1. It replays the champion's trades and searches for a simple entry-time condition that separated losers from winners, e.g. "losses clustered when `vol(20) > 0.31`". Each such lesson becomes a candidate rule: the champion AND that filter.
2. It runs the GA seeded with the champion, those candidates and the hall of fame, on data stopping one year before today.
3. The best challenger is promoted **only if** it beats the champion on that held-out year by a margin.

In the 2025–26 replay, every lesson search came back empty. Across ~900 historical trades, no simple filter removed more losing than winning return, and no challenger cleared the margin. That's the gate working: it stopped the bot from churning strategies on noise.

## What would you do next?

- **Better risk model.** Position sizing by volatility instead of equal weight, and a portfolio-level drawdown limit.
- **More honest data.** Include delisted securities if I move from ETFs to single stocks.
- **Statistical tests.** A deflated Sharpe ratio to correct for how many rules the GA tried.
- **Broker integration.** Swap the simulated broker for Alpaca's paper API behind the same interface. The `PaperStore` / `PaperTrader` split was designed for that.

## Engineering choices worth mentioning

- **Tooling.** `uv` for reproducible environments with a lockfile, `ruff` for linting, and GitHub Actions running both on Python 3.12 and 3.13.
- **Offline tests.** The suite runs on synthetic data, so CI never touches the network.
- **One inspectable file.** SQLite for the account. It needs no server, and every decision is auditable with any SQL tool.
- **Reproducibility.** Everything is seeded, so the same seed gives the same evolved rules (tested).
