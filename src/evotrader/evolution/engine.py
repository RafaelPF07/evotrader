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
    ) -> None:
        self.datasets = datasets
        self.start, self.end = start, end
        self.config = config or EvolutionConfig()
        self.fitness_config = fitness_config or FitnessConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.factory = GenomeFactory(self.rng, available_kinds(datasets), self.config.max_depth)
        self.seeds = list(seeds)
        self._cache: dict[str, Evaluation] = {}

    def score(self, genome: Genome) -> Evaluation:
        key = str(genome)
        if key not in self._cache:
            self._cache[key] = evaluate(
                genome, self.datasets, self.start, self.end, self.fitness_config
            )
        return self._cache[key]

    def _tournament(self, scored: list[tuple[Genome, Evaluation]]) -> Genome:
        picks = self.rng.choice(len(scored), size=self.config.tournament_size, replace=False)
        return scored[min(picks)][0]  # `scored` is sorted best-first

    def _offspring(self, scored: list[tuple[Genome, Evaluation]]) -> Genome:
        cfg = self.config
        if self.rng.random() < cfg.immigrant_rate:
            return self.factory.random_genome()
        if self.rng.random() < cfg.crossover_rate:
            child = self.factory.crossover(self._tournament(scored), self._tournament(scored))
            return self.factory.mutate(child) if self.rng.random() < 0.3 else child
        return self.factory.mutate(self._tournament(scored))

    def run(
        self, on_generation: Callable[[GenerationStats], None] | None = None
    ) -> EvolutionResult:
        cfg = self.config
        population = self.seeds[: cfg.population]
        while len(population) < cfg.population:
            population.append(self.factory.random_genome())
        hall: dict[str, tuple[Genome, Evaluation]] = {}
        history: list[GenerationStats] = []

        for gen in range(cfg.generations):
            scored = sorted(
                ((g, self.score(g)) for g in population), key=lambda ge: -ge[1].fitness
            )
            for g, e in scored:
                hall.setdefault(str(g), (g, e))

            stats = GenerationStats(
                generation=gen,
                best_fitness=scored[0][1].fitness,
                mean_fitness=float(np.mean([e.fitness for _, e in scored])),
                unique=len({str(g) for g, _ in scored}),
                best_rule=str(scored[0][0]),
            )
            history.append(stats)
            if on_generation:
                on_generation(stats)

            if gen < cfg.generations - 1:
                population = [g for g, _ in scored[: cfg.elite]]
                population += [self._offspring(scored) for _ in range(cfg.population - cfg.elite)]

        ranked = sorted(hall.values(), key=lambda ge: -ge[1].fitness)
        return EvolutionResult(ranked[: cfg.hall_of_fame], history)
