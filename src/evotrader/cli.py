"""Command line interface.

    evotrader backtest --ticker SPY
    evotrader backtest --ticker QQQ --strategy sma_cross --param fast=20 --param slow=100
    evotrader compare
"""

from __future__ import annotations

import argparse

import pandas as pd

from evotrader import data
from evotrader.backtest import DEFAULT_COST_BPS, run_backtest
from evotrader.strategies import REGISTRY, BuyAndHold

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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
