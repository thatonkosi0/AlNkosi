"""Per-market backtest cost profiles.

Spread is a price distance; commission is money per lot, converted to a price
distance with the symbol's tick size/value before it reaches the backtester.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from alglory.analysis.sizing import SymbolSpec

if TYPE_CHECKING:
    from alglory.evolve.campaign import CampaignConfig


@dataclass(frozen=True)
class CostProfile:
    min_spread: float  # price units; backtests never charge less
    commission_per_lot: float  # account currency per round-trip lot


def cost_profile(symbol: str) -> CostProfile:
    s = symbol.upper()
    if any(k in s for k in ("NAS", "US100", "USTEC", "NDX")):
        return CostProfile(1.5, 0.0)
    if "XAU" in s:
        return CostProfile(0.25, 7.0)
    if s.endswith("JPY"):
        return CostProfile(0.010, 7.0)
    return CostProfile(0.00010, 7.0)


def resolve_profile(cfg: CampaignConfig, symbol: str) -> CostProfile:
    base = cost_profile(symbol)
    over = cfg.cost_overrides.get(symbol, {})
    return CostProfile(
        min_spread=over.get("min_spread", base.min_spread),
        commission_per_lot=over.get("commission_per_lot", base.commission_per_lot),
    )


def commission_price(commission_per_lot: float, spec: SymbolSpec) -> float:
    """Money per lot -> price distance: ticks per lot-money times tick size."""
    return commission_per_lot * spec.tick_size / spec.tick_value
