"""Position sizing — an exact mirror of the EA's LotsForRisk.

The backtester sizes in fractional risk units and ignores the broker's minimum
lot, so it always "trades". Live MT5 enforces a minimum lot, which is why a
risk-based size can round to zero and the EA place nothing. This module
reproduces the LIVE rule (including the min-lot fallback) so deploy-time
surfacing and tests can answer "will this account actually trade?" with a
number instead of a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_MAX_RISK_PCT = 0.05


@dataclass(frozen=True)
class SymbolSpec:
    """Broker contract spec needed to turn a risk fraction into a lot size."""

    tick_value: float  # account-currency value of one tick, per 1.0 lot
    tick_size: float
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 100.0


# EURUSD on a standard USD account: one point (0.00001) is worth $1 per 1.0 lot.
EURUSD_STD = SymbolSpec(tick_value=1.0, tick_size=0.00001)


def loss_per_lot(sl_distance: float, spec: SymbolSpec) -> float:
    """Account-currency loss if a 1.0-lot position is stopped out at ``sl_distance``."""
    return sl_distance / spec.tick_size * spec.tick_value


def lots_for_risk(
    balance: float,
    risk_pct: float,
    sl_distance: float,
    spec: SymbolSpec,
    *,
    min_lot_fallback: bool = True,
    max_risk_pct: float = DEFAULT_MAX_RISK_PCT,
) -> float:
    """Lots the EA would trade — 0.0 means "no trade". Mirrors LotsForRisk exactly."""
    if balance <= 0 or sl_distance <= 0 or spec.tick_value <= 0 or spec.tick_size <= 0:
        return 0.0
    lpl = loss_per_lot(sl_distance, spec)
    lots = math.floor((balance * risk_pct / lpl) / spec.lot_step) * spec.lot_step
    if lots < spec.min_lot:
        if min_lot_fallback:
            min_lot_risk = spec.min_lot * lpl / balance
            if min_lot_risk <= max_risk_pct:
                return spec.min_lot
        return 0.0
    return min(lots, spec.max_lot)


def min_account_for_risk_sizing(risk_pct: float, sl_distance: float, spec: SymbolSpec) -> float:
    """Smallest balance whose risk-based lot already meets the broker minimum."""
    return spec.min_lot * loss_per_lot(sl_distance, spec) / risk_pct


def min_account_to_trade(
    sl_distance: float,
    spec: SymbolSpec,
    *,
    max_risk_pct: float = DEFAULT_MAX_RISK_PCT,
) -> float:
    """Smallest balance at which the min-lot fallback lets the strategy trade at all.

    Below this, even one minimum-lot position would exceed ``max_risk_pct``, so
    the EA refuses (and says so) rather than over-risk the account.
    """
    return spec.min_lot * loss_per_lot(sl_distance, spec) / max_risk_pct
