import json

import numpy as np
import pytest

from alglory.genome import (
    MANAGEMENT_SPACE,
    PARAM_SPACE,
    TRIBES,
    Genome,
    crossover,
    mutate,
    random_genome,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _assert_legal(g: Genome):
    assert g.tribe in TRIBES
    assert g.signal.kind in TRIBES[g.tribe]
    space = PARAM_SPACE[g.signal.kind]
    for name, value in g.signal.params.items():
        lo, hi, is_int = space[name]
        assert lo <= value <= hi, f"{g.signal.kind}.{name}={value} outside [{lo},{hi}]"
        if is_int:
            assert float(value).is_integer()
    if g.signal.kind == "ma_cross":
        assert g.signal.params["slow"] > g.signal.params["fast"]
    assert g.direction in ("long", "short", "both")
    assert 0.2 <= g.management.sl_atr <= 6.0
    assert 0.2 <= g.management.tp_atr <= 10.0
    assert 0.001 <= g.risk.risk_pct <= 0.03
    if g.filters.session is not None:
        h0, h1 = g.filters.session
        assert 0 <= h0 < 24 and 0 < h1 <= 24 and h0 < h1
    assert g.filters.atr_regime in (None, "high", "low")
    if g.management.breakeven_atr is not None:
        lo, hi, _ = MANAGEMENT_SPACE["breakeven_atr"]
        assert lo <= g.management.breakeven_atr <= hi


def test_tribes_cover_all_param_space_kinds():
    kinds = {k for kinds in TRIBES.values() for k in kinds}
    assert kinds == set(PARAM_SPACE.keys())


@pytest.mark.parametrize("tribe", list(TRIBES))
def test_random_genome_is_legal(tribe):
    rng = _rng(3)
    for _ in range(50):
        _assert_legal(random_genome(tribe, rng))


def test_json_round_trip():
    g = random_genome("trend", _rng(1))
    g2 = Genome.from_json(g.to_json())
    assert g2 == g


def test_from_json_legacy_genome_without_breakeven():
    g = random_genome("trend", _rng(1))
    d = json.loads(g.to_json())
    d["management"].pop("breakeven_atr", None)
    g2 = Genome.from_json(json.dumps(d))
    assert g2.management.breakeven_atr is None


def test_management_genes_favor_adaptive_exits():
    rng = _rng(7)
    genomes = [random_genome("trend", rng) for _ in range(300)]
    trail = sum(g.management.trail_atr is not None for g in genomes)
    breakeven = sum(g.management.breakeven_atr is not None for g in genomes)
    assert trail / 300 > 0.45  # trailing stops are now a common gene
    assert breakeven / 300 > 0.25  # breakeven moves appear regularly


def test_mutate_returns_new_legal_genome():
    rng = _rng(5)
    g = random_genome("mean_reversion", rng)
    changed = 0
    for _ in range(200):
        m = mutate(g, rng, rate=0.5)
        _assert_legal(m)
        assert m.tribe == g.tribe
        if m != g:
            changed += 1
    assert changed > 100  # mutation actually does something


def test_crossover_mixes_gene_groups():
    rng = _rng(9)
    a = random_genome("breakout", rng)
    b = random_genome("breakout", rng)
    child = crossover(a, b, rng)
    _assert_legal(child)
    assert child.signal in (a.signal, b.signal)
    assert child.management in (a.management, b.management)
    assert child.filters in (a.filters, b.filters)
    assert child.risk in (a.risk, b.risk)


def test_crossover_rejects_cross_tribe():
    rng = _rng(2)
    a = random_genome("trend", rng)
    b = random_genome("momentum", rng)
    with pytest.raises(ValueError, match="tribe"):
        crossover(a, b, rng)
