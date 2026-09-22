"""The genetic algorithm.

Each generation: score everyone, keep the elite unchanged, then fill the rest of the
population with children made by tournament selection + crossover / mutation, plus
a few random immigrants to keep exploring. Elitism means the best score can never
go down from one generation to the next.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from evotrader.evolution.fitness import Dataset, Evaluation, FitnessConfig, evaluate
from evotrader.evolution.genome import Genome, GenomeFactory
from evotrader.evolution.rotation import Panel, RotationFactory, evaluate_rotation
from evotrader.features import ML_COLUMN, PRICE_KINDS


@dataclass(frozen=True)
class EvolutionConfig:
    population: int = 60
    generations: int = 25
    tournament_size: int = 3
    elite: int = 4
    crossover_rate: float = 0.5
    immigrant_rate: float = 0.1
    max_depth: int = 3
    hall_of_fame: int = 10
    seed: int = 0


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    unique: int
    best_rule: str


@dataclass
class EvolutionResult:
    hall_of_fame: list[tuple[Genome, Evaluation]]
    history: list[GenerationStats] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)  # every individual, if recording

    @property
    def best(self) -> tuple[Genome, Evaluation]:
        return self.hall_of_fame[0]


def available_kinds(datasets: Sequence[Dataset]) -> tuple[str, ...]:
    """Use the ML signal as a building block only if every dataset has it."""
    has_ml = all(ML_COLUMN in ds.bars for ds in datasets)
    return (*PRICE_KINDS, ML_COLUMN) if has_ml else PRICE_KINDS


class Evolver:
    def __init__(
        self,
        datasets: list[Dataset],
        start: str,
        end: str,
        config: EvolutionConfig | None = None,
        fitness_config: FitnessConfig | None = None,
        seeds: Sequence[Genome] = (),
        space: str = "timing",
        record: bool = False,
    ) -> None:
        self.datasets = datasets
        self.start, self.end = start, end
        self.config = config or EvolutionConfig()
        self.fitness_config = fitness_config or FitnessConfig()
        self.rng = np.random.default_rng(self.config.seed)
        kinds = available_kinds(datasets)
        if space == "rotation":
            self.panel: Panel | None = Panel(datasets)
            self.factory = RotationFactory(self.rng, kinds)
        elif space == "timing":
            self.panel = None
            self.factory = GenomeFactory(self.rng, kinds, self.config.max_depth)
        else:
            raise ValueError(f"unknown search space {space!r}")
        self.seeds = list(seeds)
        self.record = record  # keep every individual, its parents and how it was made
        self._cache: dict[str, Evaluation] = {}

    def evaluate_window(self, genome: Genome, start: str, end: str) -> Evaluation:
        """Score any genome of this search space on an arbitrary date window."""
        if self.panel is not None:
            return evaluate_rotation(genome, self.panel, start, end, self.fitness_config)
        return evaluate(genome, self.datasets, start, end, self.fitness_config)

    def score(self, genome: Genome) -> Evaluation:
        key = str(genome)
        if key not in self._cache:
            self._cache[key] = self.evaluate_window(genome, self.start, self.end)
        return self._cache[key]

    def _tournament(self, scored: list[Scored]) -> tuple[Genome, int]:
        picks = self.rng.choice(len(scored), size=self.config.tournament_size, replace=False)
        winner = scored[min(picks)]  # `scored` is sorted best-first
        return winner.genome, winner.id

    def _offspring(self, scored: list[Scored]) -> Birth:
        cfg, factory = self.config, self.factory
        if self.rng.random() < cfg.immigrant_rate:
            return Birth(factory.random_genome(), "immigrant")
        if self.rng.random() < cfg.crossover_rate:
            (a, id_a), (b, id_b) = self._tournament(scored), self._tournament(scored)
            child = factory.crossover(a, b)
            origin = factory.last_op
            if self.rng.random() < 0.3:
                child = factory.mutate(child)
                origin = f"{origin} + {factory.last_op}"
            return Birth(child, origin, (id_a, id_b))
        parent, parent_id = self._tournament(scored)
        return Birth(factory.mutate(parent), factory.last_op, (parent_id,))

    def run(
        self, on_generation: Callable[[GenerationStats], None] | None = None
    ) -> EvolutionResult:
        cfg = self.config
        population = [Birth(g, "seed") for g in self.seeds[: cfg.population]]
        while len(population) < cfg.population:
            population.append(Birth(self.factory.random_genome(), "random"))
        hall: dict[str, tuple[Genome, Evaluation]] = {}
        history: list[GenerationStats] = []
        log: list[dict] = []
        next_id = 0

        for gen in range(cfg.generations):
            members = [Scored(b.genome, self.score(b.genome), next_id + i, b)
                       for i, b in enumerate(population)]
            next_id += len(members)
            scored = sorted(members, key=lambda m: -m.evaluation.fitness)
            for m in scored:
                hall.setdefault(str(m.genome), (m.genome, m.evaluation))
            if self.record:
                log += [_record(m, gen, rank) for rank, m in enumerate(scored)]

            stats = GenerationStats(
                generation=gen,
                best_fitness=scored[0].evaluation.fitness,
                mean_fitness=float(np.mean([m.evaluation.fitness for m in scored])),
                unique=len({str(m.genome) for m in scored}),
                best_rule=str(scored[0].genome),
            )
            history.append(stats)
            if on_generation:
                on_generation(stats)

            if gen < cfg.generations - 1:
                population = [Birth(m.genome, "survivor", (m.id,)) for m in scored[: cfg.elite]]
                population += [self._offspring(scored) for _ in range(cfg.population - cfg.elite)]

        ranked = sorted(hall.values(), key=lambda ge: -ge[1].fitness)
        return EvolutionResult(ranked[: cfg.hall_of_fame], history, log)


@dataclass(frozen=True)
class Birth:
    """A genome plus how it came to exist: the raw material of the evolution log."""

    genome: Genome
    origin: str  # random | seed | immigrant | survivor | <mutation> | crossover (...) [+ ...]
    parents: tuple[int, ...] = ()


@dataclass(frozen=True)
class Scored:
    genome: Genome
    evaluation: Evaluation
    id: int
    birth: Birth


def genome_kinds(genome) -> list[str]:
    """Indicator kinds a genome uses (one entry per condition or term)."""
    if hasattr(genome, "terms"):
        return [t.feature.kind for t in genome.terms]
    return [leaf.feature.kind for leaf in genome.entry.leaves() + genome.exit.leaves()]


def _record(m: Scored, generation: int, rank: int) -> dict:
    e = m.evaluation
    return {
        "id": m.id, "generation": generation, "rank": rank,
        "origin": m.birth.origin, "parents": list(m.birth.parents),
        "rule": str(m.genome), "genome": m.genome.to_dict(),
        "fitness": e.fitness, "score": e.score_mean, "exposure": e.exposure,
        "trades_per_year": e.trades_per_year, "size": m.genome.size,
        "kinds": genome_kinds(m.genome),
    }
