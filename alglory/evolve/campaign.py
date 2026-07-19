"""Campaign orchestration: data -> GA per tribe -> vault, streaming events.

The emit callable receives plain dicts following the WS event protocol the
dashboard consumes. run_campaign is synchronous and process-agnostic; the
server wraps it in a worker process (see alglory.server.worker).
"""

from __future__ import annotations

import json
import time
import traceback
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field, field_validator

from alglory.backtest.engine import Costs
from alglory.data.cache import BarCache, check_sufficiency
from alglory.data.sample import generate_bars
from alglory.evolve.fitness import FitnessConfig, OOSGate
from alglory.evolve.ga import GAParams, evolve_tribe
from alglory.genome import TRIBES
from alglory.vault.db import Vault

SAMPLE_BARS = 6000
MIN_BARS = 1500
EQUITY_POINTS = 200
TOP_CURVES = 5
BARS_PER_YEAR = {"M15": 24960.0, "H1": 6240.0, "H4": 1560.0, "D1": 260.0}


class CampaignConfig(BaseModel):
    symbols: list[str] = Field(min_length=1)
    timeframe: str = "H1"
    tribes: list[str] = Field(min_length=1)
    population: int = Field(default=40, ge=4, le=400)
    generations: int = Field(default=15, ge=1, le=200)
    oos_fraction: float = Field(default=0.25, gt=0.05, lt=0.9)
    seed: int | None = None
    source: str = "sample"
    min_trades: int = Field(default=30, ge=1)
    dd_cap: float = Field(default=0.35, gt=0.0, le=1.0)
    commission_per_lot: float = Field(default=0.0, ge=0.0)

    @field_validator("timeframe")
    @classmethod
    def _tf(cls, v):
        if v not in BARS_PER_YEAR:
            raise ValueError(f"timeframe must be one of {sorted(BARS_PER_YEAR)}")
        return v

    @field_validator("tribes")
    @classmethod
    def _tribes(cls, v):
        unknown = [t for t in v if t not in TRIBES]
        if unknown:
            raise ValueError(f"unknown tribes {unknown}; valid: {sorted(TRIBES)}")
        return v

    @field_validator("source")
    @classmethod
    def _source(cls, v):
        if v not in ("sample", "mt5"):
            raise ValueError("source must be 'sample' or 'mt5'")
        return v


def _downsample(equity: np.ndarray, n: int = EQUITY_POINTS) -> list[float]:
    if len(equity) <= n:
        return [float(x) for x in equity]
    idx = np.linspace(0, len(equity) - 1, n).astype(int)
    return [float(x) for x in equity[idx]]


def _load_bars(cfg: CampaignConfig, symbol: str, cache: BarCache, emit):
    if cfg.source == "sample":
        emit({"type": "log", "line": f"generating sample data for {symbol} {cfg.timeframe}"})
        return generate_bars(symbol, cfg.timeframe, SAMPLE_BARS, seed=cfg.seed or 7)
    from alglory.data.mt5source import MT5Source

    src = MT5Source()
    status = src.connect()
    if not status.ok:
        raise RuntimeError(status.message)
    emit({"type": "log", "line": f"downloading {symbol} {cfg.timeframe} history from MT5 ({status.account})"})
    bars = src.fetch_bars(symbol, cfg.timeframe, 20000)
    cache.save(symbol, cfg.timeframe, bars)
    return bars


def _wait_while_paused(
    should_pause: Callable[[], bool] | None,
    should_stop: Callable[[], bool],
    emit: Callable[[dict], None],
) -> None:
    """Block between generations while paused; cancellation still wins."""
    if should_pause is None or not should_pause():
        return
    emit({"type": "log", "line": "campaign paused"})
    while should_pause() and not should_stop():
        time.sleep(0.2)
    if not should_stop():
        emit({"type": "log", "line": "campaign resumed"})


def run_campaign(
    cfg: CampaignConfig,
    vault: Vault,
    cache: BarCache,
    emit: Callable[[dict], None],
    should_stop: Callable[[], bool],
    should_pause: Callable[[], bool] | None = None,
) -> None:
    campaign_id = vault.create_campaign(cfg.model_dump_json())
    total_units = len(cfg.symbols) * len(cfg.tribes) * cfg.generations
    emit(
        {
            "type": "campaign_started",
            "campaign_id": campaign_id,
            "config": cfg.model_dump(),
            "total_units": total_units,
        }
    )

    vaulted_total = 0
    status, error = "done", None
    try:
        fit_cfg = FitnessConfig(min_trades=cfg.min_trades, dd_cap=cfg.dd_cap)
        gate = OOSGate()
        ga = GAParams(population=cfg.population, generations=cfg.generations)
        units_done = 0

        for symbol in cfg.symbols:
            bars = _load_bars(cfg, symbol, cache, emit)
            check_sufficiency(bars, MIN_BARS)
            split = int(len(bars) * (1.0 - cfg.oos_fraction))
            bars_is = bars.iloc[:split].reset_index(drop=True)
            bars_oos = bars.iloc[split:].reset_index(drop=True)
            costs = Costs(
                spread=float(bars["spread"].median()),
                commission_per_lot=cfg.commission_per_lot,
            )
            emit(
                {
                    "type": "log",
                    "line": f"{symbol}: {split} bars in-sample / {len(bars) - split} out-of-sample",
                }
            )

            for t_idx, tribe in enumerate(cfg.tribes):
                _wait_while_paused(should_pause, should_stop, emit)
                if should_stop():
                    raise _Cancelled()
                emit({"type": "log", "line": f"breeding tribe '{tribe}' on {symbol}"})
                seed = (cfg.seed if cfg.seed is not None else int(time.time())) + t_idx * 1000
                for result in evolve_tribe(
                    tribe,
                    bars_is,
                    bars_oos,
                    costs,
                    ga,
                    fit_cfg,
                    gate,
                    seed=seed,
                    bars_per_year=BARS_PER_YEAR[cfg.timeframe],
                ):
                    for genome, is_m, oos_m in result.vaulted:
                        from alglory.backtest.engine import run_backtest

                        oos_equity = run_backtest(genome, bars_oos, costs).equity
                        sid = vault.add_strategy(
                            name=None,
                            tribe=tribe,
                            symbol=symbol,
                            timeframe=cfg.timeframe,
                            genome_json=genome.to_json(),
                            is_metrics=is_m.to_dict(),
                            oos_metrics=oos_m.to_dict(),
                            equity=oos_equity,
                            campaign_id=campaign_id,
                        )
                        vaulted_total += 1
                        row = vault.get_strategy(sid)
                        emit(
                            {
                                "type": "strategy_vaulted",
                                "strategy_id": sid,
                                "name": row["name"],
                                "tribe": tribe,
                                "symbol": symbol,
                                "oos_net_profit": oos_m.net_profit,
                            }
                        )

                    order = sorted(
                        range(len(result.evaluated)),
                        key=lambda i: result.evaluated[i].fit,
                        reverse=True,
                    )
                    top_equity = []
                    for i in order[:TOP_CURVES]:
                        ev = result.evaluated[i]
                        if ev.fit <= 0:
                            continue
                        from alglory.backtest.engine import run_backtest

                        eq = run_backtest(ev.genome, bars_is, costs).equity
                        top_equity.append(
                            {"id": f"G{result.gen}-{i}", "points": _downsample(eq)}
                        )

                    units_done += 1
                    emit(
                        {
                            "type": "generation",
                            "campaign_id": campaign_id,
                            "symbol": symbol,
                            "tribe": tribe,
                            "gen": result.gen + 1,
                            "of": cfg.generations,
                            "top_fitness": result.top.fit,
                            "survivors": result.survivors,
                            "culled": result.culled,
                            "population": len(result.evaluated),
                            "top_equity": top_equity,
                            "vaulted_count_total": vaulted_total,
                            "units_done": units_done,
                            "total_units": total_units,
                        }
                    )
                    vault.update_campaign(
                        campaign_id,
                        progress_json=json.dumps(
                            {
                                "units_done": units_done,
                                "total_units": total_units,
                                "vaulted_total": vaulted_total,
                                "symbol": symbol,
                                "tribe": tribe,
                                "gen": result.gen + 1,
                            }
                        ),
                    )
                    _wait_while_paused(should_pause, should_stop, emit)
                    if should_stop():
                        raise _Cancelled()
    except _Cancelled:
        status, error = "cancelled", None
        emit({"type": "log", "line": "campaign cancelled by user"})
    except Exception as exc:  # surfaced to UI; campaign row marked failed
        status = "failed"
        error = f"{exc}"
        emit({"type": "log", "line": f"campaign failed: {error}"})
        emit({"type": "log", "line": traceback.format_exc(limit=3)})

    vault.update_campaign(campaign_id, status=status)
    emit(
        {
            "type": "campaign_finished",
            "campaign_id": campaign_id,
            "status": status,
            "error": error,
            "vaulted_total": vaulted_total,
        }
    )


class _Cancelled(Exception):
    pass
