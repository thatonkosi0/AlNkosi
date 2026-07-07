import numpy as np

from alglory.backtest.engine import Costs
from alglory.evolve.fitness import FitnessConfig, OOSGate, oos_pass
from alglory.evolve.ga import GAParams, evolve_tribe

GA = GAParams(population=8, generations=3, elite=2)
FIT = FitnessConfig(min_trades=5)
GATE = OOSGate(min_trades=2)
COSTS = Costs(spread=0.00012)
BARS_PER_YEAR = 6240.0


def _run(bars, seed=42):
    bars_is = bars.iloc[:4000].reset_index(drop=True)
    bars_oos = bars.iloc[4000:].reset_index(drop=True)
    return list(
        evolve_tribe(
            "trend",
            bars_is,
            bars_oos,
            COSTS,
            GA,
            FIT,
            GATE,
            seed=seed,
            bars_per_year=BARS_PER_YEAR,
        )
    )


def test_yields_one_result_per_generation(bars_h1):
    results = _run(bars_h1)
    assert len(results) == GA.generations
    assert [r.gen for r in results] == [0, 1, 2]
    assert all(r.tribe == "trend" for r in results)


def test_population_size_invariant(bars_h1):
    for r in _run(bars_h1):
        assert len(r.evaluated) == GA.population


def test_seeded_determinism(bars_h1):
    a = _run(bars_h1, seed=7)
    b = _run(bars_h1, seed=7)
    assert [r.top.fit for r in a] == [r.top.fit for r in b]
    assert a[-1].top.genome == b[-1].top.genome


def test_different_seeds_differ(bars_h1):
    a = _run(bars_h1, seed=1)
    b = _run(bars_h1, seed=2)
    assert [r.top.genome for r in a] != [r.top.genome for r in b]


def test_elites_survive_between_generations(bars_h1):
    results = _run(bars_h1)
    for prev, cur in zip(results, results[1:]):
        prev_sorted = sorted(prev.evaluated, key=lambda e: e.fit, reverse=True)
        elite_genomes = {e.genome for e in prev_sorted[: GA.elite]}
        cur_genomes = {e.genome for e in cur.evaluated}
        assert elite_genomes <= cur_genomes


def test_top_is_best_of_generation(bars_h1):
    for r in _run(bars_h1):
        assert r.top.fit == max(e.fit for e in r.evaluated)


def test_vaulted_pass_gate_and_are_unique(bars_h1):
    seen = set()
    for r in _run(bars_h1):
        for genome, _is_m, oos_m in r.vaulted:
            assert oos_pass(oos_m, GATE)
            key = genome.to_json()
            assert key not in seen
            seen.add(key)


def test_survivors_and_culled_partition_population(bars_h1):
    for r in _run(bars_h1):
        assert set(r.survivors) | set(r.culled) == set(range(GA.population))
        assert set(r.survivors) & set(r.culled) == set()
