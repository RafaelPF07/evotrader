"""Command line interface.

    evotrader backtest --ticker SPY
    evotrader backtest --ticker QQQ --strategy sma_cross --param fast=20 --param slow=100
    evotrader compare
    evotrader walkforward
    evotrader paper init --start 2025-01-02
    evotrader paper run
    evotrader paper status
    evotrader paper journal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evotrader import data
from evotrader.backtest import DEFAULT_COST_BPS, run_backtest
from evotrader.strategies import REGISTRY, BuyAndHold

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
EVOLUTION_DIR = REPORTS_DIR / "evolution"
PCT = ["total_return", "cagr", "volatility", "max_drawdown", "exposure", "win_rate"]


def _parse_params(pairs: list[str]) -> dict[str, float | int]:
    params: dict[str, float | int] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        params[key] = float(value) if "." in value else int(value)
    return params


def format_metrics(table: pd.DataFrame) -> str:
    out = table.copy().astype(object)
    for col in out.columns:
        fmt = "{:.1%}" if col in PCT else "{:.2f}"
        out[col] = [fmt.format(v) if pd.notna(v) else "-" for v in table[col]]
    return out.to_string()


def cmd_backtest(args: argparse.Namespace) -> None:
    bars = data.load(args.ticker, start=args.start, end=args.end)
    names = [args.strategy] if args.strategy else list(REGISTRY)
    params = _parse_params(args.param)
    results = [
        run_backtest(REGISTRY[n](**(params if args.strategy else {})), bars, args.cost_bps)
        for n in names
    ]
    if args.strategy and args.strategy != BuyAndHold.name:
        results.append(run_backtest(BuyAndHold(), bars, args.cost_bps))

    print(f"\n{args.ticker}  {bars.index[0].date()} -> {bars.index[-1].date()}  "
          f"({len(bars)} bars, costs {args.cost_bps} bps/side)\n")
    table = pd.DataFrame({r.strategy: r.metrics for r in results}).T
    print(format_metrics(table[["cagr", "sharpe", "max_drawdown", "exposure", "trades",
                                "win_rate"]]))


def cmd_compare(args: argparse.Namespace) -> None:
    """Sharpe ratio of every baseline strategy across the whole universe."""
    rows = {}
    for ticker in args.tickers:
        bars = data.load(ticker, start=args.start, end=args.end)
        rows[ticker] = {
            name: run_backtest(cls(), bars, args.cost_bps).metrics["sharpe"]
            for name, cls in REGISTRY.items()
        }
    table = pd.DataFrame(rows).T
    print(f"\nSharpe ratio by strategy ({args.start} -> {args.end or 'today'})\n")
    print(table.round(2).to_string())
    beats = (table.drop(columns=BuyAndHold.name).gt(table[BuyAndHold.name], axis=0)).sum()
    print(f"\nTickers where strategy beat buy & hold:\n{beats.to_string()}")


def _evo_config(args: argparse.Namespace):
    from evotrader.evolution import EvolutionConfig

    return EvolutionConfig(population=args.population, generations=args.generations,
                           seed=args.seed)


def cmd_evolve(args: argparse.Namespace) -> None:
    """Evolve on all history up to --end and save the champion for paper trading."""
    from evotrader.evolution import Evolver, GeneticStrategy, evaluate
    from evotrader.walkforward import load_datasets

    datasets = load_datasets(args.tickers, use_ml=not args.no_ml)
    end = pd.Timestamp(args.end) if args.end else min(ds.bars.index[-1] for ds in datasets)
    val_start = pd.Timestamp(args.start) + (end - pd.Timestamp(args.start)) * 0.75
    val_start, train_end = str(val_start.date()), str((val_start - pd.Timedelta(days=1)).date())

    print(f"\nEvolving on {args.start}..{train_end}, validating on {val_start}..{end.date()}")
    evolver = Evolver(datasets, args.start, train_end, _evo_config(args))
    result = evolver.run(lambda s: print(f"  gen {s.generation:>2}  best {s.best_fitness:+.3f}"
                                         f"  mean {s.mean_fitness:+.3f}"))
    scored = [(g, e, evaluate(g, datasets, val_start, str(end.date()), evolver.fitness_config))
              for g, e in result.hall_of_fame]
    scored.sort(key=lambda t: -t[2].fitness)

    print("\nHall of fame (sorted by validation fitness):")
    for g, e, v in scored[:5]:
        print(f"  train {e.fitness:+.3f}  val {v.fitness:+.3f}  {g}")

    champion, train_eval, val_eval = scored[0]
    path = MODELS_DIR / "champion.json"
    GeneticStrategy(champion).save(path, {
        "trained": f"{args.start}..{train_end}", "validated": f"{val_start}..{end.date()}",
        "train_fitness": train_eval.fitness, "val_fitness": val_eval.fitness,
        "tickers": args.tickers, "created": pd.Timestamp.now().isoformat(timespec="seconds"),
    })
    print(f"\nChampion saved to {path}")


def cmd_walkforward(args: argparse.Namespace) -> None:
    from dataclasses import replace

    from evotrader.walkforward import load_datasets, make_folds, run_walkforward, to_markdown

    datasets = load_datasets(args.tickers, use_ml=not args.no_ml)
    last = min(ds.bars.index[-1] for ds in datasets)
    folds = make_folds(args.start, args.first_test_year, last, test_years=args.test_years)
    REPORTS_DIR.mkdir(exist_ok=True)
    cols = ["cagr", "sharpe", "max_drawdown", "exposure"]

    # A GA run is one sample of a random process, so repeat it with several seeds and
    # report the spread. The main report is always the first seed, never the best one.
    per_seed, evolved_returns, history = [], {}, []
    for seed in range(args.seed, args.seed + args.seeds):
        print(f"\n######## seed {seed} ########")
        report = run_walkforward(datasets, folds, replace(_evo_config(args), seed=seed))
        if seed == args.seed:
            first = report
        evolved_returns[f"seed_{seed}"] = report.oos_returns["evolved"]
        wins = sum(r.test_metrics["evolved"]["sharpe"] > r.test_metrics["buy_and_hold"]["sharpe"]
                   for r in report.folds)
        per_seed.append({"seed": seed, **report.oos_metrics.loc["evolved", cols].to_dict(),
                         "folds_beating_buy_and_hold": wins})
        history += [{"seed": seed, "fold": r.fold.index, **vars(s)}
                    for r in report.folds for s in r.history]

    seeds = pd.DataFrame(per_seed).set_index("seed")
    print("\nStitched out-of-sample results (first seed):\n")
    print(format_metrics(first.oos_metrics[cols]))
    markdown = to_markdown(first)
    if len(seeds) > 1:
        print(f"\nEvolved strategy across {len(seeds)} seeds:\n")
        print(format_metrics(seeds[cols].describe().loc[["mean", "std", "min", "max"]]))
        markdown += "\n" + seed_markdown(seeds, first.oos_metrics.loc["buy_and_hold"])

    (REPORTS_DIR / "walkforward.md").write_text(markdown, encoding="utf-8")
    first.oos_returns.to_csv(REPORTS_DIR / "walkforward_returns.csv")
    pd.DataFrame(evolved_returns).to_csv(REPORTS_DIR / "walkforward_evolved_by_seed.csv")
    seeds.to_csv(REPORTS_DIR / "walkforward_seeds.csv")
    pd.DataFrame(history).to_csv(REPORTS_DIR / "walkforward_history.csv", index=False)
    print(f"\nReport written to {REPORTS_DIR / 'walkforward.md'}")


def seed_markdown(seeds: pd.DataFrame, buy_and_hold: pd.Series) -> str:
    lines = ["## Robustness across random seeds", "",
             f"The whole walk-forward was repeated with {len(seeds)} seeds. The tables above "
             "show the first seed, not the best one.", "",
             "| seed | CAGR | Sharpe | max drawdown | periods beating buy & hold |",
             "|---|---|---|---|---|"]
    for seed, row in seeds.iterrows():
        lines.append(f"| {seed} | {row['cagr']:+.1%} | {row['sharpe']:.2f} | "
                     f"{row['max_drawdown']:+.1%} | {int(row['folds_beating_buy_and_hold'])} |")
    lines += [f"| **mean** | {seeds['cagr'].mean():+.1%} | {seeds['sharpe'].mean():.2f} | "
              f"{seeds['max_drawdown'].mean():+.1%} | "
              f"{seeds['folds_beating_buy_and_hold'].mean():.1f} |",
              f"| buy & hold | {buy_and_hold['cagr']:+.1%} | {buy_and_hold['sharpe']:.2f} | "
              f"{buy_and_hold['max_drawdown']:+.1%} | - |", ""]
    return "\n".join(lines)


def cmd_ml(args: argparse.Namespace) -> None:
    from evotrader.ml import attach_ml_prob, diagnostics

    rows = {}
    for ticker in args.tickers:
        bars = attach_ml_prob(ticker, data.load(ticker, start="1990-01-01"))
        rows[ticker] = diagnostics(bars.loc[args.start:args.end])
    table = pd.DataFrame(rows).T
    print("\nOut-of-sample ML signal quality (5-day direction)\n")
    print(table.round(3).to_string())
    print("\nAUC 0.50 = coin flip. Anything reliably above ~0.52 on daily data is notable.")


def _paper(args: argparse.Namespace, refresh: bool):
    """Open the account database and load the universe it trades."""
    from evotrader.evolution.fitness import FitnessConfig
    from evotrader.paper import LearnerConfig, PaperStore, PaperTrader
    from evotrader.walkforward import load_datasets

    store = PaperStore(args.db)
    tickers = store.get("tickers") or getattr(args, "tickers", data.DEFAULT_UNIVERSE)
    use_ml = store.get("use_ml", getattr(args, "ml", False))
    # An account always re-learns with the objective it was created with. Accounts
    # created before objectives existed used "sharpe".
    default = getattr(args, "objective", "sharpe") if not store.initialised else "sharpe"
    objective = store.get("objective", default)
    learner = LearnerConfig(fitness=FitnessConfig(objective=objective))
    datasets = load_datasets(tickers, use_ml=use_ml, refresh=refresh, log=lambda _: None)
    return store, PaperTrader(store, datasets, learner)


def cmd_paper_init(args: argparse.Namespace) -> None:
    store, trader = _paper(args, refresh=True)
    trader.init(args.capital, args.start)
    store.set("use_ml", args.ml)
    store.commit()
    print(f"\nAccount created in {args.db}. Run `evotrader paper run` to trade up to today.")


def cmd_paper_run(args: argparse.Namespace) -> None:
    from evotrader.paper.report import status_text

    store, trader = _paper(args, refresh=not args.no_refresh)
    until = pd.Timestamp(args.until) if args.until else None
    days = trader.run(until)
    print(f"\nProcessed {days} trading day(s).\n")
    print(status_text(store, trader.bars))


def cmd_paper_status(args: argparse.Namespace) -> None:
    from evotrader.paper.report import status_text

    store, trader = _paper(args, refresh=False)
    print(status_text(store, trader.bars))


def cmd_paper_journal(args: argparse.Namespace) -> None:
    from evotrader.paper import PaperStore
    from evotrader.paper.report import journal_text

    print(journal_text(PaperStore(args.db), args.n))


def cmd_charts(args: argparse.Namespace) -> None:
    from evotrader import charts

    out = ROOT / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)
    made = [charts.walkforward_chart(REPORTS_DIR, out), charts.learning_curve_chart(REPORTS_DIR,
                                                                                    out)]
    db = Path(args.db)
    if db.exists():
        from evotrader.paper.report import equity_frame

        store, trader = _paper(args, refresh=False)
        log = store.frame("SELECT date FROM learning_log")
        made.append(charts.paper_chart(equity_frame(store, trader.bars),
                                       list(pd.to_datetime(log["date"])), out))
    made += [p for p in (charts.evolution_swarm_gif(EVOLUTION_DIR, out),
                         charts.family_tree_chart(EVOLUTION_DIR, out)) if p is not None]
    for path in made:
        print(f"wrote {path.relative_to(ROOT)}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    import subprocess
    import sys

    app = Path(__file__).with_name("dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--", "--db", args.db],
                   check=False)


def cmd_evolution(args: argparse.Namespace) -> None:
    """Record full evolution runs (every individual and its parents) for visualisation."""
    from evotrader.evolution import EvolutionConfig
    from evotrader.evolution import history as h
    from evotrader.walkforward import load_datasets

    datasets = load_datasets(data.DEFAULT_UNIVERSE, use_ml=False, log=lambda _: None)
    objectives = ["excess", "sharpe"] if args.objective == "both" else [args.objective]
    for objective in objectives:
        print(f"Recording {objective} evolution ({args.start}..{args.end}, seed {args.seed})")
        log = h.record_run(datasets, args.start, args.end, objective,
                           EvolutionConfig(population=args.population,
                                           generations=args.generations, seed=args.seed))
        path = EVOLUTION_DIR / f"{objective}_seed{args.seed}.json"
        h.save(log, path)
        _, df = h.load(path)
        champ = h.champion(df)
        print(f"  {len(df)} individuals -> {path.relative_to(ROOT)}\n  champion: {champ['rule']}")


def cmd_experiment3(args: argparse.Namespace) -> None:
    from evotrader import experiments3 as ex3

    if args.summary:
        v = ex3.verdict()
        print(v.to_string() if len(v) else "No results yet.")
        return
    if args.stage is None:
        raise SystemExit("--stage is required unless --summary is given")
    ex3.check_allowed(args.stage)
    results = ex3.run_stage(args.stage)
    ex3.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(ex3.result_path(args.stage), index=False)
    cols = ["arm", "cagr", "sharpe", "volatility", "max_drawdown", "avg_exposure",
            "days_in_cash", "switches_per_year", "beats_bh"]
    print(f"\n{args.stage}\n")
    print(results[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def cmd_experiment2(args: argparse.Namespace) -> None:
    from evotrader import experiments2 as ex2

    if args.summary:
        v = ex2.verdict()
        print(v.to_string() if len(v) else "No results yet.")
        return
    if args.stage is None:
        raise SystemExit("--stage is required unless --summary is given")
    ex2.check_allowed(args.stage)
    results = ex2.run_stage(args.stage)
    ex2.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(ex2.result_path(args.stage), index=False)
    cols = ["arm", "cagr", "sharpe", "volatility", "max_drawdown", "avg_exposure",
            "days_levered", "beats_bh"]
    print(f"\n{args.stage}\n")
    print(results[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def cmd_experiment(args: argparse.Namespace) -> None:
    from evotrader import experiments as ex
    from evotrader.walkforward import load_datasets

    if args.summary:
        table = ex.summary_table(args.stage)
        print(table.to_string(index=False) if len(table) else "No results yet.")
        return
    if args.arm is None:
        raise SystemExit("--arm is required unless --summary is given")
    ex.check_allowed(args.stage, args.arm)
    datasets = load_datasets(data.DEFAULT_UNIVERSE, use_ml=False, log=lambda _: None)
    results = ex.run_arm(ex.ARMS[args.arm], args.stage, datasets)
    ex.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(ex.result_path(args.stage, args.arm), index=False)
    v = ex.verdict(results)
    print(f"\nArm {args.arm} ({args.stage}): mean excess Sharpe {v['mean_excess_sharpe']:+.3f}, "
          f"{v['seeds_beating']}/{len(results)} seeds beat buy & hold -> "
          f"{'PASS' if v['passed'] else 'FAIL'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evotrader")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start", default="2010-01-01")
    common.add_argument("--end", default=None)
    common.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)

    bt = sub.add_parser("backtest", parents=[common], help="Backtest strategies on one ticker")
    bt.add_argument("--ticker", default="SPY")
    bt.add_argument("--strategy", choices=list(REGISTRY), help="default: run all baselines")
    bt.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    bt.set_defaults(func=cmd_backtest)

    cmp_ = sub.add_parser("compare", parents=[common], help="All baselines across the universe")
    cmp_.add_argument("--tickers", nargs="+", default=data.DEFAULT_UNIVERSE)
    cmp_.set_defaults(func=cmd_compare)

    universe = argparse.ArgumentParser(add_help=False)
    universe.add_argument("--tickers", nargs="+", default=data.DEFAULT_UNIVERSE)
    universe.add_argument("--no-ml", action="store_true", help="don't use the ML signal")

    evo = argparse.ArgumentParser(add_help=False)
    evo.add_argument("--population", type=int, default=60)
    evo.add_argument("--generations", type=int, default=25)
    evo.add_argument("--seed", type=int, default=0)

    ev = sub.add_parser("evolve", parents=[universe, evo], help="Evolve and save a champion")
    ev.add_argument("--start", default="2007-01-01")
    ev.add_argument("--end", default=None)
    ev.set_defaults(func=cmd_evolve)

    wf = sub.add_parser("walkforward", parents=[universe, evo],
                        help="Out-of-sample evaluation of the learning process")
    wf.add_argument("--start", default="2007-01-01", help="start of every training window")
    wf.add_argument("--first-test-year", type=int, default=2016)
    wf.add_argument("--test-years", type=int, default=2)
    wf.add_argument("--seeds", type=int, default=1, help="repeat with this many seeds")
    wf.set_defaults(func=cmd_walkforward)

    ml = sub.add_parser("ml", parents=[common, universe], help="ML signal diagnostics")
    ml.set_defaults(func=cmd_ml)

    paper = sub.add_parser("paper", help="Self-learning paper trading account")
    paper_sub = paper.add_subparsers(dest="paper_command", required=True)
    db = argparse.ArgumentParser(add_help=False)
    db.add_argument("--db", default=str(ROOT / "data" / "paper.db"))

    p_init = paper_sub.add_parser("init", parents=[db], help="Create the account")
    p_init.add_argument("--capital", type=float, default=100_000.0)
    p_init.add_argument("--start", default=str(pd.Timestamp.today().date()),
                        help="first trading day; a past date replays history honestly")
    p_init.add_argument("--tickers", nargs="+", default=data.DEFAULT_UNIVERSE)
    p_init.add_argument("--ml", action="store_true", help="let rules use the ML signal (slower)")
    p_init.add_argument("--objective", choices=["excess", "sharpe"], default="excess",
                        help="what re-learning optimises: excess = beat buy & hold "
                             "(passed the pre-registered experiment); sharpe = original")
    p_init.set_defaults(func=cmd_paper_init)

    p_run = paper_sub.add_parser("run", parents=[db], help="Trade every day since the last run")
    p_run.add_argument("--until", default=None)
    p_run.add_argument("--no-refresh", action="store_true", help="don't download new prices")
    p_run.set_defaults(func=cmd_paper_run)

    p_status = paper_sub.add_parser("status", parents=[db], help="Account summary")
    p_status.set_defaults(func=cmd_paper_status)

    p_journal = paper_sub.add_parser("journal", parents=[db], help="Closed trades and why")
    p_journal.add_argument("-n", type=int, default=10)
    p_journal.set_defaults(func=cmd_paper_journal)

    exp3 = sub.add_parser("experiment3",
                          help="Pre-registered leveraged trend test (docs/EXPERIMENTS_3.md)")
    exp3.add_argument("--stage", choices=list(("dev", "final-sectors", "final-countries",
                                               "post-publication")))
    exp3.add_argument("--summary", action="store_true", help="apply the pass rule")
    exp3.set_defaults(func=cmd_experiment3)

    exp2 = sub.add_parser("experiment2",
                          help="Pre-registered volatility-targeting test (docs/EXPERIMENTS_2.md)")
    exp2.add_argument("--stage", choices=["dev", "final-us2000", "final-intl"])
    exp2.add_argument("--summary", action="store_true", help="apply the pass rule")
    exp2.set_defaults(func=cmd_experiment2)

    evo_rec = sub.add_parser("evolution", help="Record evolution runs for the visualiser")
    evo_rec.add_argument("--objective", choices=["both", "excess", "sharpe"], default="both")
    evo_rec.add_argument("--start", default="2007-01-01")
    evo_rec.add_argument("--end", default="2021-12-31",
                         help="default stays out of the experiment's one-time holdout")
    evo_rec.add_argument("--population", type=int, default=60)
    evo_rec.add_argument("--generations", type=int, default=25)
    evo_rec.add_argument("--seed", type=int, default=0)
    evo_rec.set_defaults(func=cmd_evolution)

    exp = sub.add_parser("experiment", help="Pre-registered experiments (docs/EXPERIMENTS.md)")
    exp.add_argument("--stage", choices=["dev", "holdout"], required=True)
    exp.add_argument("--arm", choices=["A", "B", "C", "D"])
    exp.add_argument("--summary", action="store_true", help="show results for the stage")
    exp.set_defaults(func=cmd_experiment)

    ch = sub.add_parser("charts", parents=[db], help="Regenerate README charts in docs/img")
    ch.set_defaults(func=cmd_charts)
    dash = sub.add_parser("dashboard", parents=[db], help="Open the Streamlit dashboard")
    dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
