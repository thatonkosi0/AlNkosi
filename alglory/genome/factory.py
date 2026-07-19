"""Genetic operators: random genome creation, mutation, crossover.

All operators are pure — they take an injected numpy Generator and return
new Genome objects, never mutating inputs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .genes import FilterGene, Genome, ManagementGene, RiskGene, SignalGene
from .tribes import (
    MANAGEMENT_SPACE,
    PARAM_SPACE,
    RISK_SPACE,
    TREND_MA_SPACE,
    TRIBES,
)


def _draw(space: tuple[float, float, bool], rng: np.random.Generator) -> float | int:
    lo, hi, is_int = space
    if is_int:
        return int(rng.integers(int(lo), int(hi) + 1))
    return float(rng.uniform(lo, hi))


def _draw_signal(kind: str, rng: np.random.Generator) -> SignalGene:
    params = {name: _draw(space, rng) for name, space in PARAM_SPACE[kind].items()}
    if kind == "ma_cross":
        while params["slow"] <= params["fast"]:
            params["slow"] = _draw(PARAM_SPACE[kind]["slow"], rng)
            params["fast"] = _draw(PARAM_SPACE[kind]["fast"], rng)
    return SignalGene(kind=kind, params=params)


def _draw_session(rng: np.random.Generator) -> tuple[int, int]:
    h0 = int(rng.integers(0, 20))
    h1 = int(rng.integers(h0 + 2, min(h0 + 14, 24) + 1))
    return (h0, min(h1, 24))


def _draw_filters(rng: np.random.Generator) -> FilterGene:
    return FilterGene(
        trend_ma=_draw(TREND_MA_SPACE, rng) if rng.random() < 0.5 else None,
        session=_draw_session(rng) if rng.random() < 0.4 else None,
        atr_regime=str(rng.choice(["high", "low"])) if rng.random() < 0.3 else None,
    )


def _draw_management(rng: np.random.Generator) -> ManagementGene:
    return ManagementGene(
        sl_atr=_draw(MANAGEMENT_SPACE["sl_atr"], rng),
        tp_atr=_draw(MANAGEMENT_SPACE["tp_atr"], rng),
        trail_atr=_draw(MANAGEMENT_SPACE["trail_atr"], rng) if rng.random() < 0.6 else None,
        max_bars=_draw(MANAGEMENT_SPACE["max_bars"], rng) if rng.random() < 0.3 else None,
        breakeven_atr=_draw(MANAGEMENT_SPACE["breakeven_atr"], rng) if rng.random() < 0.4 else None,
    )


def random_genome(tribe: str, rng: np.random.Generator) -> Genome:
    if tribe not in TRIBES:
        raise ValueError(f"Unknown tribe {tribe!r}; expected one of {sorted(TRIBES)}")
    kind = str(rng.choice(TRIBES[tribe]))
    return Genome(
        tribe=tribe,
        direction=str(rng.choice(["long", "short", "both"])),
        signal=_draw_signal(kind, rng),
        filters=_draw_filters(rng),
        management=_draw_management(rng),
        risk=RiskGene(risk_pct=_draw(RISK_SPACE["risk_pct"], rng)),
    )


def _perturb(value: float | int, space: tuple[float, float, bool], rng: np.random.Generator):
    lo, hi, is_int = space
    sigma = (hi - lo) * 0.15
    out = float(value) + rng.normal(0.0, sigma)
    out = float(np.clip(out, lo, hi))
    return int(round(out)) if is_int else out


def mutate(g: Genome, rng: np.random.Generator, rate: float = 0.3) -> Genome:
    signal = g.signal
    if rng.random() < rate:
        params = dict(signal.params)
        name = str(rng.choice(list(params)))
        params[name] = _perturb(params[name], PARAM_SPACE[signal.kind][name], rng)
        if signal.kind == "ma_cross" and params["slow"] <= params["fast"]:
            params["slow"] = int(params["fast"]) + 1
        signal = SignalGene(kind=signal.kind, params=params)
    if rng.random() < rate * 0.3:  # occasionally resample the signal kind within tribe
        signal = _draw_signal(str(rng.choice(TRIBES[g.tribe])), rng)

    filters = g.filters
    if rng.random() < rate:
        filters = _draw_filters(rng)

    management = g.management
    if rng.random() < rate:
        management = replace(
            management,
            sl_atr=_perturb(management.sl_atr, MANAGEMENT_SPACE["sl_atr"], rng),
            tp_atr=_perturb(management.tp_atr, MANAGEMENT_SPACE["tp_atr"], rng),
        )
    if rng.random() < rate * 0.5:
        management = replace(
            management,
            trail_atr=_draw(MANAGEMENT_SPACE["trail_atr"], rng) if rng.random() < 0.6 else None,
            max_bars=_draw(MANAGEMENT_SPACE["max_bars"], rng) if rng.random() < 0.5 else None,
            breakeven_atr=(
                _draw(MANAGEMENT_SPACE["breakeven_atr"], rng) if rng.random() < 0.5 else None
            ),
        )

    risk = g.risk
    if rng.random() < rate:
        risk = RiskGene(risk_pct=_perturb(risk.risk_pct, RISK_SPACE["risk_pct"], rng))

    direction = g.direction
    if rng.random() < rate * 0.3:
        direction = str(rng.choice(["long", "short", "both"]))

    return Genome(
        tribe=g.tribe,
        direction=direction,
        signal=signal,
        filters=filters,
        management=management,
        risk=risk,
    )


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    if a.tribe != b.tribe:
        raise ValueError(f"Cannot cross genomes from different tribes: {a.tribe} x {b.tribe}")
    pick = lambda x, y: x if rng.random() < 0.5 else y  # noqa: E731
    return Genome(
        tribe=a.tribe,
        direction=pick(a.direction, b.direction),
        signal=pick(a.signal, b.signal),
        filters=pick(a.filters, b.filters),
        management=pick(a.management, b.management),
        risk=pick(a.risk, b.risk),
    )
