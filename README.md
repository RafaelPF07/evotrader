# EvoTrader

A self-improving **paper trading** bot for US ETFs. It evolves its own trading strategies with a genetic algorithm, uses machine learning signals as building blocks, and only promotes a strategy after it passes out-of-sample, walk-forward validation. No real money is involved.

> Status: **Phase 1 of 4**. The backtesting foundation and baseline strategies are done.

## Roadmap

- [x] **Phase 1: Foundation.** Data cache, look-ahead-proof backtester, costs, metrics, baseline strategies, tests.
- [ ] **Phase 2: Learning engine.** Genetic programming over indicator rules, ML signal, walk-forward validation.
- [ ] **Phase 3: Paper trading loop.** Daily simulated broker, trade journal, feedback into learning.
- [ ] **Phase 4: Showcase.** Dashboard, results write-up, CI.

## Design principles

1. **No look-ahead.** Strategies decide at the close of day *t*, and orders fill at the open of day *t+1*. A test scrambles future prices and asserts that no past decision changes.
2. **Costs are always on.** The default is 5 bps per side for commission, spread and slippage.
3. **Buy & hold is the bar to beat.** Every result is reported next to it.
4. **Survivorship-safe universe.** Broad ETFs rather than hand-picked winning stocks.

## Quick start

```bash
uv sync
uv run pytest
uv run evotrader backtest --ticker SPY
uv run evotrader backtest --ticker QQQ --strategy sma_cross --param fast=20 --param slow=100
uv run evotrader compare
```

## Project layout

```
src/evotrader/
  data.py          # Yahoo Finance download + CSV cache + validation
  indicators.py    # SMA, EMA, RSI, momentum, z-score, volatility
  strategies/      # Strategy interface and baseline strategies
  backtest.py      # Execution simulation, costs, trade log
  metrics.py       # CAGR, Sharpe, Sortino, max drawdown, Calmar, ...
  cli.py           # `evotrader` command
tests/             # Offline tests on synthetic data
```
