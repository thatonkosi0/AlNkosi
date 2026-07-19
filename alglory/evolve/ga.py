"""Tournament genetic algorithm over one tribe's subpopulation.

`evolve_tribe` is a generator: it yields a GenerationResult after every
generation so callers (the campaign orchestrator) can stream progress and
vault out-of-sample survivors as they appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alglory.backtest.engine import Costs, run_backtest
from alglory.backtest.metrics import Metrics, compute_metrics
from alglory.genome import Genome, crossover, mutate, random_genome

from .fitness import FitnessConfig, OOSGate, fitness, oos_pass

OOS_CANDIDATE_FRACTION = 0.2


@dataclass(frozen=True)
class GAParams:
    population: int = 40
    generations: int = 15
    tournament_k: int = 3
    elite: int = 2
    cx_prob: float = 0.6
    mut_rate: float = 0.3


@dataclass(frozen=True)
class Evaluated:
    genome: Genome
    is_metrics: Metrics
    fit: float


@dataclass
class GenerationResult:
    gen: int
    tribe: str
    evaluated: list[Evaluated]
    survivors: list[int]
    culled: list[int]
    top: Evaluated
    vaulted: list[tuple[Genome, Metrics, Metrics]] = field(default_factory=list)
    # Population bred for gen+1, or None on the final generation. Exposed so
    # the campaign can checkpoint it and resume a crashed run mid-tribe.
    next_population: list[Genome] | None = None


def _evaluate(
    genome: Genome,
    bars: pd.DataFrame,
    costs: Costs,
    fit_cfg: FitnessConfig,
    bars_per_year: float,
) -> Evaluated:
    result = run_backtest(genome, bars, costs)
    metrics = compute_metrics(result, bars_per_year)
    return Evaluated(genome=genome, is_metrics=metrics, fit=fitness(metrics, fit_cfg))


def _tournament(pop: list[Evaluated], k: int, rng: np.random.Generator) -> Genome:
    picks = rng.integers(0, len(pop), size=k)
    best = max(picks, key=lambda i: pop[i].fit)
    return pop[best].genome


def evolve_tribe(
    tribe: str,
    bars_is: pd.DataFrame,
    bars_oos: pd.DataFrame,
    costs: Costs,
    ga: GAParams,
    fit_cfg: FitnessConfig,
    gate: OOSGate,
    seed: int,
    bars_per_year: float,
    initial: list[Genome] | None = None,
    start_gen: int = 0,
    already_vaulted: set[str] | None = None,
):
    rng = np.random.default_rng(seed)
    genomes = (
        list(initial)
        if initial is not None
        else [random_genome(tribe, rng) for _ in range(ga.population)]
    )
    already_vaulted = set(already_vaulted) if already_vaulted else set()

    for gen in range(start_gen, ga.generations):
        evaluated = [
            _evaluate(g, bars_is, costs, fit_cfg, bars_per_year) for g in genomes
        ]
        order = sorted(range(len(evaluated)), key=lambda i: evaluated[i].fit, reverse=True)
        n_survive = max(ga.elite, ga.population // 2)
        survivors = order[:n_survive]
        culled = order[n_survive:]
        top = evaluated[order[0]]

        # Forward-test the generation's best candidates out-of-sample.
        vaulted: list[tuple[Genome, Metrics, Metrics]] = []
        n_candidates = max(1, int(ga.population * OOS_CANDIDATE_FRACTION))
        for i in order[:n_candidates]:
            cand = evaluated[i]
            if cand.fit <= 0.0:
                continue
            key = cand.genome.to_json()
            if key in already_vaulted:
                continue
            oos_result = run_backtest(cand.genome, bars_oos, costs)
            oos_metrics = compute_metrics(oos_result, bars_per_year)
            if oos_pass(oos_metrics, gate):
                vaulted.append((cand.genome, cand.is_metrics, oos_metrics))
                already_vaulted.add(key)

        # Breed the next generation before yielding: elites unchanged, rest
        # via tournament. Yielding it lets the campaign checkpoint the exact
        # population a resumed run should continue from.
        next_genomes: list[Genome] | None = None
        if gen < ga.generations - 1:
            next_genomes = [evaluated[i].genome for i in order[: ga.elite]]
            while len(next_genomes) < ga.population:
                parent_a = _tournament(evaluated, ga.tournament_k, rng)
                if rng.random() < ga.cx_prob:
                    parent_b = _tournament(evaluated, ga.tournament_k, rng)
                    child = crossover(parent_a, parent_b, rng)
                else:
                    child = parent_a
                child = mutate(child, rng, rate=ga.mut_rate)
                next_genomes.append(child)

        yield GenerationResult(
            gen=gen,
            tribe=tribe,
            evaluated=evaluated,
            survivors=survivors,
            culled=culled,
            top=top,
            vaulted=vaulted,
            next_population=next_genomes,
        )

        if next_genomes is None:
            break
        genomes = next_genomes
