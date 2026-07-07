"""Strategy genome dataclasses and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace  # noqa: F401  (replace used by factory)


@dataclass(frozen=True)
class SignalGene:
    kind: str
    params: dict[str, float | int]

    def __eq__(self, other):
        return (
            isinstance(other, SignalGene)
            and self.kind == other.kind
            and self.params == other.params
        )

    def __hash__(self):
        return hash((self.kind, tuple(sorted(self.params.items()))))


@dataclass(frozen=True)
class FilterGene:
    trend_ma: int | None
    session: tuple[int, int] | None
    atr_regime: str | None  # 'high' | 'low' | None


@dataclass(frozen=True)
class ManagementGene:
    sl_atr: float
    tp_atr: float
    trail_atr: float | None
    max_bars: int | None


@dataclass(frozen=True)
class RiskGene:
    risk_pct: float


@dataclass(frozen=True)
class Genome:
    tribe: str
    direction: str  # 'long' | 'short' | 'both'
    signal: SignalGene
    filters: FilterGene
    management: ManagementGene
    risk: RiskGene

    def to_json(self) -> str:
        d = asdict(self)
        if d["filters"]["session"] is not None:
            d["filters"]["session"] = list(d["filters"]["session"])
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Genome":
        d = json.loads(s)
        session = d["filters"]["session"]
        return cls(
            tribe=d["tribe"],
            direction=d["direction"],
            signal=SignalGene(kind=d["signal"]["kind"], params=d["signal"]["params"]),
            filters=FilterGene(
                trend_ma=d["filters"]["trend_ma"],
                session=tuple(session) if session is not None else None,
                atr_regime=d["filters"]["atr_regime"],
            ),
            management=ManagementGene(**d["management"]),
            risk=RiskGene(**d["risk"]),
        )
