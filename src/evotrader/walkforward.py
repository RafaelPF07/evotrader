"""Walk-forward evaluation: the honest answer to "does the learning actually work?"

For each fold:
    [ train ................ | validation ] [ test ]
      evolve rules here        pick champion   never seen during evolution or selection

The champion is then compared on the test years with baselines, on an equal-weight
portfolio across the universe. Test windows are stitched together into one
continuous out-of-sample track record.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from evotrader.backtest import simulate
from evotrader.data import load
from evotrader.evolution.engine import EvolutionConfig, Evolver, GenerationStats
from evotrader.evolution.fitness import Dataset, FitnessConfig, evaluate
from evotrader.evolution.genome import Genome
from evotrader.evolution.strategy import GeneticStrategy
from evotrader.metrics import compute_metrics
from evotrader.ml import MlSignal, attach_ml_prob
from evotrader.strategies import BuyAndHold, Momentum, SmaCrossover
from evotrader.strategies.base import Strategy

Log = Callable[[str], None]


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    val_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __str__(self) -> str:
        d = lambda ts: ts.date().isoformat()  # noqa: E731
        return (f"fold {self.index}: train {d(self.train_start)}..{d(self.val_start)} "
                f"| val ..{d(self.train_end)} | test {d(self.test_start)}..{d(self.test_end)}")


def make_folds(
    train_start: str,
    first_test_year: int,
    last_date: pd.Timestamp,
    test_years: int = 2,
    val_fraction: float = 0.25,
) -> list[Fold]:
    """Expanding-window folds. A final remainder shorter than a year is absorbed
    into the last test window rather than becoming a tiny fold of its own."""
    start = pd.Timestamp(train_start)
    folds, year = [], first_test_year
    while (test_start := pd.Timestamp(f"{year}-01-01")) <= last_date:
        test_end = pd.Timestamp(f"{year + test_years - 1}-12-31")
        if test_end + pd.DateOffset(years=1) > last_date:
            test_end = last_date
        train_end = test_start - pd.Timedelta(days=1)
        val_start = (start + (train_end - start) * (1 - val_fraction)).normalize()
        folds.append(Fold(len(folds), start, val_start, train_end, test_start, test_end))
        if test_end >= last_date:
            break
        year += test_years
    return folds


def load_datasets(
    tickers: list[str], use_ml: bool = True, log: Log = print, refresh: bool = False
) -> list[Dataset]:
    datasets = []
    for ticker in tickers:
        bars = load(ticker, start="1990-01-01", refresh=refresh)
        if use_ml:
            log(f"  {ticker}: ML signal (walk-forward, cached after first run)")
            bars = attach_ml_prob(ticker, bars)
        datasets.append(Dataset(ticker, bars))
    return datasets


def portfolio_returns(
    strategy: Strategy, datasets: list[Dataset], start: pd.Timestamp, end: pd.Timestamp,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series]:
    """Equal-weight, daily-rebalanced portfolio of `strategy` run on every ticker."""
    rets, held = {}, {}
    for ds in datasets:
        r, h = simulate(ds.bars, strategy.target_position(ds.bars), cost_bps)
        rets[ds.ticker], held[ds.ticker] = r.loc[start:end], h.loc[start:end]
    returns = pd.DataFrame(rets).mean(axis=1).fillna(0.0)
    exposure = pd.DataFrame(held).mean(axis=1).fillna(0.0)
    return returns, exposure


@dataclass
class FoldResult:
    fold: Fold
    champion: Genome
    train_fitness: float
    val_fitness: float
    hall_of_fame: list[Genome]
    history: list[GenerationStats]
    test_metrics: dict[str, dict[str, float]]  # strategy -> metrics
    test_returns: dict[str, pd.Series] = field(repr=False)
    test_exposure: dict[str, pd.Series] = field(repr=False)


@dataclass
class WalkForwardReport:
    folds: list[FoldResult]
    oos_returns: pd.DataFrame  # stitched test returns, one column per strategy
    oos_metrics: pd.DataFrame


def baselines() -> dict[str, Strategy]:
    return {
        "buy_and_hold": BuyAndHold(),
        "sma_cross": SmaCrossover(),
        "momentum": Momentum(),
        "ml_signal": MlSignal(),
    }


def run_fold(
    fold: Fold,
    datasets: list[Dataset],
    evo_config: EvolutionConfig,
    fit_config: FitnessConfig,
    seeds: list[Genome],
    log: Log = print,
) -> FoldResult:
    log(f"\n{fold}")
    last_train_day = fold.val_start - pd.Timedelta(days=1)
    evolver = Evolver(datasets, str(fold.train_start.date()), str(last_train_day.date()),
                      evo_config, fit_config, seeds=seeds)
    result = evolver.run(
        lambda s: log(f"  gen {s.generation:>2}  best {s.best_fitness:+.3f}  "
                      f"mean {s.mean_fitness:+.3f}  unique {s.unique}")
    )

    # Choose the champion on validation data the GA never trained on.
    val = [
        (g, e, evaluate(g, datasets, str(fold.val_start.date()), str(fold.train_end.date()),
                        fit_config))
        for g, e in result.hall_of_fame
    ]
    champion, train_eval, val_eval = max(val, key=lambda t: t[2].fitness)
    log(f"  champion (train {train_eval.fitness:+.3f}, val {val_eval.fitness:+.3f}): {champion}")

    strategies: dict[str, Strategy] = {"evolved": GeneticStrategy(champion), **baselines()}
    metrics, returns, exposures = {}, {}, {}
    for name, strat in strategies.items():
        r, exposure = portfolio_returns(strat, datasets, fold.test_start, fold.test_end,
                                        fit_config.cost_bps)
        returns[name], exposures[name] = r, exposure
        metrics[name] = compute_metrics(r, exposure)
    log("  test Sharpe: " + "  ".join(f"{k} {v['sharpe']:+.2f}" for k, v in metrics.items()))
    return FoldResult(fold, champion, train_eval.fitness, val_eval.fitness,
                      [g for g, _ in result.hall_of_fame], result.history, metrics, returns,
                      exposures)


def run_walkforward(
    datasets: list[Dataset],
    folds: list[Fold],
    evo_config: EvolutionConfig | None = None,
    fit_config: FitnessConfig | None = None,
    carry_over: bool = True,
    log: Log = print,
) -> WalkForwardReport:
    """Run every fold. With `carry_over`, each fold's hall of fame seeds the next
    fold's population, so the bot keeps building on what it learned before."""
    evo_config = evo_config or EvolutionConfig()
    fit_config = fit_config or FitnessConfig()
    results: list[FoldResult] = []
    seeds: list[Genome] = []
    for fold in folds:
        res = run_fold(fold, datasets, evo_config, fit_config, seeds, log)
        results.append(res)
        if carry_over:
            seeds = [res.champion, *(g for g in res.hall_of_fame if g != res.champion)]

    names = list(results[0].test_returns)
    stitch = lambda attr, n: pd.concat([getattr(r, attr)[n] for r in results])  # noqa: E731
    oos = pd.DataFrame({n: stitch("test_returns", n) for n in names})
    oos_metrics = pd.DataFrame(
        {n: compute_metrics(oos[n], stitch("test_exposure", n)) for n in names}
    ).T
    return WalkForwardReport(results, oos, oos_metrics)


def to_markdown(report: WalkForwardReport, title: str = "Walk-forward results") -> str:
    """Human-readable report: per-fold champions and test results, then the stitched record."""
    pct = lambda v: f"{v:+.1%}"  # noqa: E731
    names = list(report.oos_metrics.index)
    lines = [f"# {title}", "",
             "Every number below is **out-of-sample**: the rules were evolved and selected",
             "without ever seeing the test period. Equal-weight portfolio across the universe,",
             "costs included.", "", "## Stitched out-of-sample record", "",
             "| strategy | CAGR | Sharpe | max drawdown | exposure |", "|---|---|---|---|---|"]
    for n in names:
        m = report.oos_metrics.loc[n]
        lines.append(f"| {n} | {pct(m['cagr'])} | {m['sharpe']:.2f} | {pct(m['max_drawdown'])} "
                     f"| {m['exposure']:.0%} |")

    lines += ["", "## Per fold (test Sharpe)", "",
              "| fold | test period | " + " | ".join(names) + " |",
              "|---|---|" + "---|" * len(names)]
    for r in report.folds:
        f = r.fold
        period = f"{f.test_start.date()} .. {f.test_end.date()}"
        cells = " | ".join(f"{r.test_metrics[n]['sharpe']:.2f}" for n in names)
        lines.append(f"| {f.index} | {period} | {cells} |")

    lines += ["", "## Champion rules", ""]
    for r in report.folds:
        lines += [f"**Fold {r.fold.index}** (train fitness {r.train_fitness:+.3f}, "
                  f"validation {r.val_fitness:+.3f})", "", f"`{r.champion}`", ""]
    return "\n".join(lines)
