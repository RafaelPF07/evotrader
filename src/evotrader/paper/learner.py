"""How the live bot keeps learning.

Every `relearn_every` trading days:

1. **Study mistakes.** Replay the current champion's trades and look for a
   condition that separated its losing trades from its winners, e.g. "most losses
   were entered when vol(20) > 0.31". Each such lesson becomes a candidate rule:
   the champion plus that extra filter.
2. **Evolve.** Run the GA seeded with the champion, the lesson-based candidates and
   the hall of fame, on data that stops *before* a held-out recent window.
3. **Promote only on evidence.** The best challenger replaces the champion only
   if it beats it on the held-out window by a margin. Otherwise the bot keeps
   what works; a lesson that doesn't hold up on unseen data is discarded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from evotrader.backtest import extract_trades, simulate
from evotrader.evolution.engine import EvolutionConfig, Evolver
from evotrader.evolution.fitness import Dataset, Evaluation, FitnessConfig, evaluate
from evotrader.evolution.genome import And, Compare, Genome
from evotrader.features import KINDS, FeatureSpec

Log = Callable[[str], None]

DIAGNOSTIC_FEATURES = (
    FeatureSpec("rsi", (14,)),
    FeatureSpec("zscore", (20,)),
    FeatureSpec("mom", (63,)),
    FeatureSpec("trend", (50,)),
    FeatureSpec("trend", (200,)),
    FeatureSpec("vol", (20,)),
    FeatureSpec("ma_spread", (20, 100)),
)


@dataclass(frozen=True)
class LearnerConfig:
    train_start: str = "2007-01-01"
    holdout_days: int = 252
    relearn_every: int = 63
    promote_margin: float = 0.10
    evolution: EvolutionConfig = field(
        default_factory=lambda: EvolutionConfig(population=40, generations=12)
    )
    fitness: FitnessConfig = field(default_factory=FitnessConfig)


@dataclass(frozen=True)
class Lesson:
    condition: Compare
    trades: int
    losers_avoided: int
    winners_lost: int
    return_gain: float  # sum of per-trade returns that the filter would have avoided losing

    def __str__(self) -> str:
        return (f"only enter when {self.condition}: skips {self.losers_avoided} losers and "
                f"{self.winners_lost} winners of {self.trades} trades ({self.return_gain:+.1%})")


@dataclass(frozen=True)
class Decision:
    challenger: Genome
    incumbent_score: Evaluation
    challenger_score: Evaluation
    promoted: bool
    hall_of_fame: list[Genome]
    lessons: list[Lesson]

    @property
    def notes(self) -> str:
        verdict = "PROMOTED" if self.promoted else "kept incumbent"
        lines = [f"{verdict}: holdout fitness incumbent {self.incumbent_score.fitness:+.3f} vs "
                 f"challenger {self.challenger_score.fitness:+.3f}"]
        lines += [f"lesson: {lesson}" for lesson in self.lessons] or [
            "lesson: none - no entry filter would have removed more losing than winning return"
        ]
        return "\n".join(lines)


def truncate(datasets: list[Dataset], end: pd.Timestamp) -> list[Dataset]:
    """Datasets that physically contain no bars after `end`, so no code path can peek."""
    return [Dataset(ds.ticker, ds.bars.loc[:end]) for ds in datasets]


def _windows(datasets: list[Dataset], end: pd.Timestamp, holdout: int) -> tuple[str, str, str]:
    calendar = datasets[0].bars.index[datasets[0].bars.index <= end]
    if len(calendar) <= holdout + 252:
        raise ValueError("Not enough history before the end date to learn")
    fit_end, holdout_start = calendar[-holdout - 1], calendar[-holdout]
    return str(fit_end.date()), str(holdout_start.date()), str(calendar[-1].date())


def train_initial(
    datasets: list[Dataset], end: pd.Timestamp, config: LearnerConfig | None = None,
    log: Log = print,
) -> tuple[Genome, list[Genome]]:
    """First champion: evolve up to the holdout, choose the best on the holdout."""
    cfg = config or LearnerConfig()
    data = truncate(datasets, end)
    fit_end, hold_start, hold_end = _windows(data, end, cfg.holdout_days)
    log(f"  evolving on {cfg.train_start}..{fit_end}, selecting on {hold_start}..{hold_end}")
    result = Evolver(data, cfg.train_start, fit_end, cfg.evolution, cfg.fitness).run()
    hall = [g for g, _ in result.hall_of_fame]
    champion = max(hall, key=lambda g: evaluate(g, data, hold_start, hold_end, cfg.fitness).fitness)
    return champion, hall


def analyse_mistakes(
    genome: Genome, datasets: list[Dataset], start: str, end: str, cost_bps: float, top: int = 3
) -> list[Lesson]:
    """Find simple conditions at entry time that separated losing from winning trades."""
    frames = []
    for ds in datasets:
        target = pd.Series(genome.target_position(ds.store), index=ds.bars.index)
        _, held = simulate(ds.bars, target, cost_bps)
        trades = extract_trades(ds.bars, held, cost_bps)
        in_window = trades["entry_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        trades = trades[~trades["open"].astype(bool) & in_window]
        if trades.empty:
            continue
        decided = ds.bars.index.get_indexer(trades["entry_date"]) - 1  # the bar the rule fired on
        frames.append(pd.DataFrame(
            {"return": trades["return"].to_numpy()}
            | {str(s): ds.store.get(s)[decided] for s in DIAGNOSTIC_FEATURES}
        ))
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    if len(df) < 20:
        return []

    best: dict[str, Lesson] = {}
    for spec in DIAGNOSTIC_FEATURES:
        col = df[str(spec)]
        kind = KINDS[spec.kind]
        for q in np.arange(0.1, 0.91, 0.1):
            thr = round(float(np.clip(col.quantile(q), kind.lo, kind.hi)), kind.decimals)
            for op in ("<", ">"):
                keep = (col < thr if op == "<" else col > thr) | col.isna()
                if keep.mean() < 0.5:  # a filter that removes most trades isn't a lesson
                    continue
                skipped = df.loc[~keep, "return"]
                gain = -float(skipped.sum())
                lesson = Lesson(Compare(spec, op, thr), len(df), int((skipped < 0).sum()),
                                int((skipped > 0).sum()), gain)
                if gain > 0 and gain > getattr(best.get(str(spec)), "return_gain", 0.0):
                    best[str(spec)] = lesson
    return sorted(best.values(), key=lambda lesson: -lesson.return_gain)[:top]


def relearn(
    datasets: list[Dataset], end: pd.Timestamp, incumbent: Genome, hall: list[Genome],
    config: LearnerConfig | None = None, seed: int = 0, log: Log = print,
) -> Decision:
    cfg = config or LearnerConfig()
    data = truncate(datasets, end)
    fit_end, hold_start, hold_end = _windows(data, end, cfg.holdout_days)

    lessons = analyse_mistakes(incumbent, data, cfg.train_start, fit_end, cfg.fitness.cost_bps)
    for lesson in lessons:
        log(f"  lesson: {lesson}")
    lesson_genomes = [Genome(And(incumbent.entry, ls.condition), incumbent.exit) for ls in lessons]

    seeds = [incumbent, *lesson_genomes, *hall]
    evo = replace(cfg.evolution, seed=seed)
    result = Evolver(data, cfg.train_start, fit_end, evo, cfg.fitness, seeds=seeds).run()
    new_hall = [g for g, _ in result.hall_of_fame]
    challenger = next((g for g in new_hall if str(g) != str(incumbent)), new_hall[0])

    inc = evaluate(incumbent, data, hold_start, hold_end, cfg.fitness)
    cha = evaluate(challenger, data, hold_start, hold_end, cfg.fitness)
    promoted = cha.fitness > inc.fitness + cfg.promote_margin
    return Decision(challenger, inc, cha, promoted, new_hall, lessons)
