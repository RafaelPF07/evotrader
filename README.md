# EvoTrader

A self-improving **paper trading** bot for US ETFs. It evolves its own trading strategies with a genetic algorithm, uses machine learning signals as building blocks, and only promotes a strategy after it passes out-of-sample, walk-forward validation. No real money is involved.

> Status: **Phase 2 of 4**. The learning engine and walk-forward evaluation are done.

## Roadmap

- [x] **Phase 1: Foundation.** Data cache, look-ahead-proof backtester, costs, metrics, baseline strategies, tests.
- [x] **Phase 2: Learning engine.** Genetic programming over indicator rules, walk-forward ML signal, walk-forward validation.
- [ ] **Phase 3: Paper trading loop.** Daily simulated broker, trade journal, feedback into learning.
- [ ] **Phase 4: Showcase.** Dashboard, results write-up, CI.

## How the bot learns

A strategy is a **genome**: an entry rule and an exit rule, each a boolean expression tree over normalised features:

```
ENTRY (rsi(2) < 15.0 AND trend(200) > 0.010) | EXIT rsi(2) > 70.0
```

- **Building blocks.** RSI, z-score, momentum, distance from a moving average, moving-average spread, volatility, and `ml_prob`, the output of a gradient-boosting model. Every feature is scale-free, so one rule means the same thing on every ticker.
- **Evolution.** Tournament selection, subtree crossover, mutation (nudge a threshold, change a window, flip a comparison, grow or shrink the tree), elitism and random immigrants.
- **Fitness.** Scored across the whole universe at once: mean Sharpe, minus a penalty for inconsistency between tickers, rule complexity, and barely trading. This rewards rules that generalise rather than ones that fit a single lucky chart.
- **Selection.** The GA trains on the first 75% of the training window. The champion is picked from the hall of fame on the held-out last 25%.
- **Walk-forward.** An expanding training window, then 2-year test windows from 2016 to today that the process never sees. Each fold's hall of fame seeds the next fold, so the bot keeps building on what it learned.

### The ML signal

A `HistGradientBoostingClassifier` predicts whether a position opened at the next open is up 5 days later. It is refit every quarter on an expanding window. It is also **purged**: a label is known only 5 days after its date, so the model at day *t* trains only on rows up to *t − 6*. A test scrambles future prices and asserts that past probabilities don't change.

## Results (honest, out-of-sample)

Walk-forward over 2016 to Sep 2026: an equal-weight portfolio across 10 ETFs, costs included. Every test period was unseen by both evolution and selection. See [reports/walkforward.md](reports/walkforward.md) for the full breakdown.

| strategy | CAGR | Sharpe | max drawdown | exposure |
|---|---|---|---|---|
| **evolved** | 8.2% | 0.81 | **-15.5%** | 78% |
| buy & hold | 14.0% | 0.96 | -30.1% | 100% |
| SMA 50/200 | 8.9% | 0.84 | -23.7% | 74% |
| momentum | 8.1% | 0.91 | -14.4% | 72% |
| ML only | 5.3% | 0.58 | -24.7% | 55% |

**What this shows:**
- **No alpha.** The evolved strategies did **not** beat buy & hold on return or Sharpe. They won 1 of 5 test periods (2020–21).
- **Half the drawdown.** They did cut the maximum drawdown roughly in half (-15.5% vs -30.1%).
- **The GA found a stable rule.** From fold 1 onward it independently converged on the same family of rules: *buy short-term dips (`zscore(10) < -0.5`), exit when volatility spikes*. That is short-term mean reversion with a volatility filter, a well-documented effect in equity indices. Rediscovering it from scratch, and keeping it stable across folds, is evidence that the search works rather than random curve-fitting.
- **The ML signal is essentially a coin flip.** It has an out-of-sample AUC of 0.50–0.52 and was never picked into a champion rule. This is expected for daily-horizon direction prediction.

The parameters were **not** tuned after seeing these results. Doing that would turn the test set into a training set.

## Design principles

1. **No look-ahead.** Strategies decide at the close of day *t*, and orders fill at the open of day *t+1*. Tests scramble future prices and assert that no past decision, return or ML prediction changes.
2. **Costs are always on.** The default is 5 bps per side for commission, spread and slippage.
3. **Buy & hold is the bar to beat.** Every result is reported next to it.
4. **Survivorship-safe universe.** Broad ETFs rather than hand-picked winning stocks.
5. **Reproducible.** Every run is seeded, so the same seed gives the same evolved rules.

## Quick start

```bash
uv sync
uv run pytest
uv run evotrader backtest --ticker SPY           # baselines on one ticker
uv run evotrader compare                         # baselines across the universe
uv run evotrader ml                              # out-of-sample quality of the ML signal
uv run evotrader walkforward                     # the honest test of the learning process
uv run evotrader evolve                          # evolve on all history, save models/champion.json
```

## Project layout

```
src/evotrader/
  data.py          # Yahoo Finance download + CSV cache + validation
  indicators.py    # SMA, EMA, RSI, momentum, z-score, volatility
  features.py      # Normalised feature catalogue + per-dataset feature cache
  ml.py            # Walk-forward, purged gradient-boosting signal
  strategies/      # Strategy interface and baseline strategies
  evolution/
    genome.py      # Rule trees, random generation, mutation, crossover
    fitness.py     # Multi-ticker fitness function
    engine.py      # The genetic algorithm
    strategy.py    # Genome -> Strategy adapter, save/load
  walkforward.py   # Fold construction, evaluation, markdown report
  backtest.py      # Execution simulation, costs, trade log
  metrics.py       # CAGR, Sharpe, Sortino, max drawdown, Calmar, ...
  cli.py           # `evotrader` command
tests/             # Offline tests on synthetic data
reports/           # Generated walk-forward results
```
