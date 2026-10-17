"""Campaign orchestration: data -> GA per tribe -> vault, streaming events.

The emit callable receives plain dicts following the WS event protocol the
dashboard consumes. run_campaign is synchronous and process-agnostic; the
server wraps it in a worker process (see alglory.server.worker).
"""

from __future__ import annotations

import json
import time
import traceback
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from alglory.backtest.engine import Costs, effective_spread, run_backtest
from alglory.validation import validate_symbol
from alglory.data.cache import BarCache, InsufficientDataError, check_sufficiency
from alglory.data.sample import generate_bars
from alglory.evolve.costs import commission_price, resolve_profile
from alglory.evolve.fitness import FitnessConfig, HoldoutGate, OOSGate
from alglory.evolve.ga import GAParams, evolve_tribe
from alglory.evolve.holdout import (
    Finalist,
    HoldoutResult,
    _naive_iso,
    judge_finalist,
    keep_best,
)
from alglory.evolve.split import split_bars
from alglory.deploy.mql5 import GUARDRAIL_PRESETS
from alglory.genome import TRIBES, Genome, mutate
from alglory.vault.db import Vault

SAMPLE_BARS = 6000
MIN_BARS = 1500
HISTORY_BARS = 50_000
EQUITY_POINTS = 200
TOP_CURVES = 5
BARS_PER_YEAR = {"M15": 24960.0, "H1": 6240.0, "H4": 1560.0, "D1": 260.0}


class BrokerUnavailableError(RuntimeError):
    """The broker connection itself failed: no market can be loaded."""


class _TfData(NamedTuple):
    """One timeframe of the current market, kept until its holdout stage."""

    bars: pd.DataFrame
    select: pd.DataFrame
    holdout: pd.DataFrame
    costs: Costs


class CampaignConfig(BaseModel):
    symbols: list[str] = Field(min_length=1)
    timeframes: list[str] = Field(default_factory=lambda: ["H1"], min_length=1)
    tribes: list[str] = Field(min_length=1)
    population: int = Field(default=40, ge=4, le=400)
    generations: int = Field(default=15, ge=1, le=200)
    select_fraction: float = Field(default=0.2, gt=0.05, lt=0.9)
    holdout_fraction: float = Field(default=0.2, gt=0.05, lt=0.9)
    min_holdout_trades: int = Field(default=50, ge=1)
    target_preset: str = "personal"
    keep_per_market: int = Field(default=3, ge=1)
    seed: int | None = None
    source: str = "sample"
    min_trades: int = Field(default=30, ge=1)
    dd_cap: float = Field(default=0.35, gt=0.0, le=1.0)
    # Per-symbol overrides of the market cost profile:
    # {"XAUUSD": {"min_spread": 0.4, "commission_per_lot": 0.0}}
    cost_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)
    # Retrain/optimize: seed the population from this vaulted strategy's
    # genome (plus mutants of it) instead of starting from random genomes.
    seed_strategy_id: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_fields(cls, data):
        """Accept the pre-multi-timeframe `timeframe` / `oos_fraction` fields."""
        if isinstance(data, dict):
            data = dict(data)
            if "timeframe" in data:
                tf = data.pop("timeframe")
                data.setdefault("timeframes", [tf])
            if "oos_fraction" in data:
                frac = data.pop("oos_fraction")
                data.setdefault("select_fraction", frac)
        return data

    @model_validator(mode="after")
    def _split_fits(self):
        # checked here, not per market: inside the run it would be caught as
        # "not available from broker" and the campaign would finish empty
        if self.select_fraction + self.holdout_fraction >= 0.9:
            raise ValueError(
                f"select_fraction + holdout_fraction must be below 0.9 (got "
                f"{self.select_fraction} + {self.holdout_fraction}); leave at least "
                f"10% of the history for breeding"
            )
        return self

    @field_validator("timeframes")
    @classmethod
    def _tfs(cls, v):
        bad = [t for t in v if t not in BARS_PER_YEAR]
        if bad:
            raise ValueError(f"timeframes must be from {sorted(BARS_PER_YEAR)}; got {bad}")
        return v

    @field_validator("target_preset")
    @classmethod
    def _preset(cls, v):
        if v not in GUARDRAIL_PRESETS:
            raise ValueError(f"target_preset must be one of {sorted(GUARDRAIL_PRESETS)}")
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

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, v):
        return [validate_symbol(s) for s in v]


def _downsample(equity: np.ndarray, n: int = EQUITY_POINTS) -> list[float]:
    if len(equity) <= n:
        return [float(x) for x in equity]
    idx = np.linspace(0, len(equity) - 1, n).astype(int)
    return [float(x) for x in equity[idx]]


def _load_bars(
    cfg: CampaignConfig, symbol: str, timeframe: str, cache: BarCache, emit
) -> tuple[pd.DataFrame, Costs]:
    """Return (bars, costs) for one symbol; costs are in price units."""
    if cfg.source == "sample":
        emit({"type": "log", "line": f"generating sample data for {symbol} {timeframe}"})
        bars = generate_bars(symbol, timeframe, SAMPLE_BARS, seed=cfg.seed or 7)
        # synthetic bars carry a realistic spread on every bar; no floor needed
        return bars, Costs(spread=effective_spread(bars), commission=0.0)
    from alglory.data.mt5source import MT5Source

    src = MT5Source()
    status = src.connect()
    if not status.ok:
        raise BrokerUnavailableError(status.message)
    emit({"type": "log", "line": f"downloading {symbol} {timeframe} history from MT5 ({status.account})"})
    count = HISTORY_BARS
    while True:
        try:
            bars = src.fetch_bars(symbol, timeframe, count)
            if bars is None or len(bars) == 0:
                raise RuntimeError(f"MT5 returned no {symbol} {timeframe} bars")
            break
        except Exception as exc:
            if count // 2 < MIN_BARS:
                raise
            count //= 2
            emit({"type": "log", "line": f"{symbol} {timeframe}: fetch failed ({exc}); retrying with {count} bars"})
    cache.save(symbol, timeframe, bars)
    profile = resolve_profile(cfg, symbol)
    spread = effective_spread(bars, floor=max(profile.min_spread, src.symbol_spread(symbol)))
    commission = commission_price(profile.commission_per_lot, src.symbol_spec(symbol))
    emit({"type": "log", "line": f"{symbol}: costs spread {spread:.6g} + commission {commission:.6g} (price units)"})
    return bars, Costs(spread=spread, commission=commission)


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


def _seed_population(genome: Genome, size: int, rng: np.random.Generator) -> list[Genome]:
    """The optimize starting field: the vaulted genome plus mutants of it."""
    pop = [genome]
    while len(pop) < size:
        pop.append(mutate(genome, rng, rate=0.5))
    return pop


def _resume_state(
    cfg: CampaignConfig, ckpt: dict
) -> tuple[int, int, int, int, list[Genome] | None]:
    """Where a checkpointed campaign continues: symbol idx, timeframe idx,
    tribe idx, start generation, and the population to seed the resumed
    tribe with."""
    sym_start = cfg.symbols.index(ckpt["symbol"])
    # checkpoints from before multi-timeframe campaigns carry no timeframe
    tf_start = cfg.timeframes.index(ckpt["timeframe"]) if ckpt.get("timeframe") else 0
    tribe_start = cfg.tribes.index(ckpt["tribe"])
    start_gen = int(ckpt["gen"])  # checkpoint gen is 1-based == next 0-based gen
    pop_json = ckpt.get("population")
    population = [Genome.from_json(j) for j in pop_json] if pop_json else None

    if start_gen >= cfg.generations:
        # the checkpointed tribe finished its last generation: move on
        tribe_start += 1
        start_gen, population = 0, None
        if tribe_start >= len(cfg.tribes):
            tribe_start = 0
            tf_start += 1
            if tf_start >= len(cfg.timeframes):
                tf_start = 0
                sym_start += 1
    elif population is None:
        # checkpoint predates population snapshots: redo the tribe (the
        # vault-seeded dedupe set prevents duplicate strategies)
        start_gen = 0
    return sym_start, tf_start, tribe_start, start_gen, population


def _window(symbol: str, timeframe: str, holdout: pd.DataFrame) -> dict:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": _naive_iso(holdout["time"].iloc[0]),
        "end": _naive_iso(holdout["time"].iloc[-1]),
    }


def _overlaps(a: dict, b: dict) -> bool:
    return pd.Timestamp(a["start"]) <= pd.Timestamp(b["end"]) and pd.Timestamp(
        b["start"]
    ) <= pd.Timestamp(a["end"])


def _holdout_stage(
    cfg: CampaignConfig,
    vault: Vault,
    emit: Callable[[dict], None],
    campaign_id: int,
    symbol: str,
    finalists: list[Finalist],
    tf_data: dict[str, _TfData],
    campaign_windows: list[dict],
) -> int:
    """Judge every finalist of one market once on its holdout; vault the best.

    Runs only after all of the market's breeding is finished, so no holdout
    result can influence which genomes were bred or selected. Returns the
    number of strategies vaulted.

    Safe to rerun after a crash inside the stage: windows already recorded
    are not recorded again and strategies this campaign already vaulted for
    the market are not vaulted twice.
    """
    # A resumed market whose already-bred timeframe no longer loads has no
    # holdout slice for that timeframe's finalists: skip them, loudly.
    missing = [
        tf for tf in cfg.timeframes
        if tf not in tf_data and any(f.timeframe == tf for f in finalists)
    ]
    for tf in missing:
        emit({"type": "log", "line": f"{symbol} {tf}: holdout skipped — no data"})
    finalists = [f for f in finalists if f.timeframe in tf_data]
    if not finalists:
        emit({"type": "log", "line": f"{symbol}: no finalists reached the holdout"})
        return 0
    gate = HoldoutGate(min_trades=cfg.min_holdout_trades, max_dd=cfg.dd_cap)
    preset = None if cfg.target_preset == "personal" else GUARDRAIL_PRESETS[cfg.target_preset]

    # An earlier run already recorded this market's windows, so it crashed
    # inside this stage; finish it without duplicating its effects.
    interrupted = any(w["symbol"] == symbol for w in campaign_windows)
    if interrupted:
        emit({"type": "log", "line": f"{symbol}: completing an interrupted holdout stage"})

    # Record the windows as consumed before anything is judged on them.
    reused: dict[str, bool] = {}
    judged_tfs = [tf for tf in cfg.timeframes if any(f.timeframe == tf for f in finalists)]
    for tf in judged_tfs:
        w = _window(symbol, tf, tf_data[tf].holdout)
        prior = vault.consumed_holdout_windows(symbol, tf, exclude_campaign=campaign_id)
        reused[tf] = cfg.seed_strategy_id is not None or any(_overlaps(w, p) for p in prior)
        if w not in campaign_windows:
            campaign_windows.append(w)
    vault.set_holdout_windows(campaign_id, campaign_windows)

    results: list[HoldoutResult] = []
    for f in finalists:
        d = tf_data[f.timeframe]
        r = judge_finalist(f, d.holdout, d.costs, gate, preset, BARS_PER_YEAR[f.timeframe])
        results.append(r)
        head = f"{symbol} {f.timeframe} {f.tribe}"
        line = (
            f"{head}: holdout pass"
            if r.passed
            else f"{head}: holdout fail — {'; '.join(r.reasons)}"
        )
        emit({"type": "log", "line": line})
    if any(reused.values()):
        emit({"type": "log", "line": f"{symbol}: holdout reused — results are weaker evidence"})

    kept = keep_best(results, cfg.keep_per_market)
    n_pass = sum(r.passed for r in results)
    emit(
        {
            "type": "log",
            "line": f"{symbol}: {n_pass} of {len(results)} finalists passed the holdout; "
            f"keeping {len(kept)}",
        }
    )
    vaulted = 0
    for r in kept:
        f = r.finalist
        if interrupted and f.genome.to_json() in vault.campaign_tribe_genomes(
            campaign_id, symbol, f.tribe
        ):
            vaulted += 1  # stored before the crash, never counted
            continue
        d = tf_data[f.timeframe]
        sel_equity = run_backtest(f.genome, d.select, d.costs).equity
        sid = vault.add_strategy(
            name=None,
            tribe=f.tribe,
            symbol=symbol,
            timeframe=f.timeframe,
            genome_json=f.genome.to_json(),
            is_metrics=f.is_metrics.to_dict(),
            oos_metrics=f.select_metrics.to_dict(),
            equity=sel_equity,
            campaign_id=campaign_id,
            stage="holdout_passed",
            holdout_metrics=r.metrics.to_dict(),
            holdout_trades=r.records,
            holdout_reused=reused[f.timeframe],
        )
        # Walk-forward robustness on the full series, as a descriptive label
        # only: it never feeds keep_best or any selection. Best-effort.
        grade = None
        try:
            from alglory.analysis.robustness import robustness

            rob = robustness(
                f.genome, d.bars, d.costs, BARS_PER_YEAR[f.timeframe], timeframe=f.timeframe
            )
            vault.set_robustness(sid, rob["grade"], rob["score"])
            grade = rob["grade"]
        except Exception as exc:  # noqa: BLE001 - scoring must not abort a campaign
            emit({"type": "log", "line": f"robustness scoring failed for strategy {sid}: {exc}"})
        vaulted += 1
        row = vault.get_strategy(sid)
        emit(
            {
                "type": "strategy_vaulted",
                "strategy_id": sid,
                "name": row["name"],
                "tribe": f.tribe,
                "symbol": symbol,
                "timeframe": f.timeframe,
                "oos_net_profit": f.select_metrics.net_profit,
                "holdout_win_rate": r.metrics.win_rate,
                "holdout_profit_factor": r.metrics.profit_factor,
                "holdout_reused": reused[f.timeframe],
                "robustness_grade": grade,
            }
        )
    return vaulted


def run_campaign(
    cfg: CampaignConfig,
    vault: Vault,
    cache: BarCache,
    emit: Callable[[dict], None],
    should_stop: Callable[[], bool],
    should_pause: Callable[[], bool] | None = None,
    resume_campaign_id: int | None = None,
) -> None:
    sym_start = tf_start = tribe_start = resume_gen = 0
    resume_pop: list[Genome] | None = None
    resuming = resume_campaign_id is not None
    position: dict = {}
    finalists_ckpt: dict[str, list[dict]] = {}
    holdout_done: list[str] = []
    campaign_windows: list[dict] = []
    if resuming:
        row = vault.get_campaign(resume_campaign_id)
        if row is None or not row["progress_json"]:
            raise ValueError(f"Campaign {resume_campaign_id} has no checkpoint to resume from.")
        ckpt = json.loads(row["progress_json"])
        campaign_id = resume_campaign_id
        vault.update_campaign(campaign_id, status="running")
        sym_start, tf_start, tribe_start, resume_gen, resume_pop = _resume_state(cfg, ckpt)
        vaulted_total = int(ckpt.get("vaulted_total", 0))
        units_done = int(ckpt.get("units_done", 0))
        position = {k: ckpt.get(k) for k in ("symbol", "timeframe", "tribe", "gen", "population")}
        finalists_ckpt = dict(ckpt.get("finalists") or {})
        holdout_done = list(ckpt.get("holdout_done") or [])
        campaign_windows = json.loads(row.get("holdout_windows_json") or "[]")
        ck_sym = ckpt.get("symbol")
        if ck_sym in cfg.symbols and ck_sym not in holdout_done:
            ck_idx = cfg.symbols.index(ck_sym)
            if sym_start > ck_idx:
                # the market's breeding finished but its holdout stage never
                # completed: run it (every timeframe already bred) first
                sym_start, tf_start, tribe_start = ck_idx, len(cfg.timeframes), 0
                resume_gen, resume_pop = 0, None
    else:
        campaign_id = vault.create_campaign(cfg.model_dump_json())
        vaulted_total = 0
        units_done = 0

    units_per_tf = len(cfg.tribes) * cfg.generations
    total_units = len(cfg.symbols) * len(cfg.timeframes) * units_per_tf

    def persist() -> None:
        vault.update_campaign(
            campaign_id,
            progress_json=json.dumps(
                {
                    "units_done": units_done,
                    "total_units": total_units,
                    "vaulted_total": vaulted_total,
                    **position,
                    "finalists": finalists_ckpt,
                    "holdout_done": holdout_done,
                }
            ),
        )

    emit(
        {
            "type": "campaign_started",
            "campaign_id": campaign_id,
            "config": cfg.model_dump(),
            "total_units": total_units,
            "resumed": resuming,
        }
    )
    if resuming:
        emit(
            {
                "type": "log",
                "line": f"resuming campaign #{campaign_id} from checkpoint "
                f"({units_done}/{total_units} generation units done)",
            }
        )

    status, error = "done", None
    try:
        fit_cfg = FitnessConfig(min_trades=cfg.min_trades, dd_cap=cfg.dd_cap)
        gate = OOSGate()
        ga = GAParams(population=cfg.population, generations=cfg.generations)

        seed_genome, seed_tribe = None, None
        if cfg.seed_strategy_id is not None:
            srow = vault.get_strategy(cfg.seed_strategy_id)
            if srow is None:
                raise RuntimeError(
                    f"Strategy {cfg.seed_strategy_id} is not in the vault; cannot optimize."
                )
            seed_genome = Genome.from_json(srow["genome_json"])
            seed_tribe = srow["tribe"]
            if seed_tribe not in cfg.tribes:
                raise RuntimeError(
                    f"Strategy {cfg.seed_strategy_id}'s tribe {seed_tribe!r} is not in the "
                    f"selected tribes {cfg.tribes}; add it or drop seed_strategy_id."
                )
            emit(
                {
                    "type": "log",
                    "line": f"optimizing {srow['name']}: population seeded from its genome",
                }
            )

        for s_idx, symbol in enumerate(cfg.symbols):
            if s_idx < sym_start or symbol in holdout_done:
                continue
            resumed_market = resuming and s_idx == sym_start
            # finalists keyed by genome|timeframe; holdout data kept per
            # timeframe in memory until this market's holdout stage
            market: dict[str, Finalist] = {}
            if resumed_market:
                for d in finalists_ckpt.get(symbol, []):
                    f = Finalist.from_json(d)
                    market.setdefault(f.key, f)
            finalists_ckpt[symbol] = [f.to_json() for f in market.values()]
            tf_data: dict[str, _TfData] = {}

            def unbred(tf_idx: int) -> int:
                """Generation units of this timeframe not yet in units_done."""
                if not resumed_market or tf_idx > tf_start:
                    return units_per_tf
                if tf_idx < tf_start:
                    return 0
                return units_per_tf - (tribe_start * cfg.generations + resume_gen)

            for tf_idx, timeframe in enumerate(cfg.timeframes):
                # earlier timeframes of a resumed market are loaded (their
                # finalists need the holdout slice) but not bred again
                already_bred = resumed_market and tf_idx < tf_start
                try:
                    bars, costs = _load_bars(cfg, symbol, timeframe, cache, emit)
                    check_sufficiency(bars, MIN_BARS)
                except InsufficientDataError as exc:
                    emit({"type": "log", "line": f"{symbol} {timeframe}: skipped — {exc}"})
                    units_done += unbred(tf_idx)
                    continue
                except BrokerUnavailableError:
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad market must not end the campaign
                    emit(
                        {
                            "type": "log",
                            "line": f"{symbol}: skipped — not available from broker "
                            f"({type(exc).__name__}: {exc})",
                        }
                    )
                    units_done += sum(unbred(i) for i in range(tf_idx, len(cfg.timeframes)))
                    break
                bars_breed, bars_select, bars_holdout = split_bars(
                    bars, cfg.select_fraction, cfg.holdout_fraction
                )
                tf_data[timeframe] = _TfData(bars, bars_select, bars_holdout, costs)
                emit(
                    {
                        "type": "log",
                        "line": f"{symbol} {timeframe}: {len(bars_breed)} bars breed / "
                        f"{len(bars_select)} select / {len(bars_holdout)} holdout",
                    }
                )
                if already_bred:
                    continue

                for t_idx, tribe in enumerate(cfg.tribes):
                    if resumed_market and tf_idx == tf_start and t_idx < tribe_start:
                        continue
                    is_resumed_tribe = (
                        resumed_market and tf_idx == tf_start and t_idx == tribe_start
                    )
                    _wait_while_paused(should_pause, should_stop, emit)
                    if should_stop():
                        raise _Cancelled()
                    emit({"type": "log", "line": f"breeding tribe '{tribe}' on {symbol} {timeframe}"})
                    seed = (cfg.seed if cfg.seed is not None else int(time.time())) + t_idx * 1000
                    initial = None
                    if is_resumed_tribe:
                        initial = resume_pop
                    elif seed_genome is not None and tribe == seed_tribe:
                        initial = _seed_population(
                            seed_genome, cfg.population, np.random.default_rng(seed)
                        )
                    for result in evolve_tribe(
                        tribe,
                        bars_breed,
                        bars_select,
                        costs,
                        ga,
                        fit_cfg,
                        gate,
                        seed=seed,
                        bars_per_year=BARS_PER_YEAR[timeframe],
                        initial=initial,
                        start_gen=resume_gen if is_resumed_tribe else 0,
                        already_vaulted={
                            f.genome.to_json()
                            for f in market.values()
                            if f.timeframe == timeframe
                        },
                    ):
                        # select-gate passers become finalists; nothing is
                        # vaulted (or judged on the holdout) until the market ends
                        for genome, is_m, sel_m in result.vaulted:
                            f = Finalist(genome, tribe, timeframe, is_m, sel_m)
                            market.setdefault(f.key, f)
                        finalists_ckpt[symbol] = [f.to_json() for f in market.values()]

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
                            eq = run_backtest(ev.genome, bars_breed, costs).equity
                            top_equity.append(
                                {"id": f"G{result.gen}-{i}", "points": _downsample(eq)}
                            )

                        units_done += 1
                        emit(
                            {
                                "type": "generation",
                                "campaign_id": campaign_id,
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "tribe": tribe,
                                "gen": result.gen + 1,
                                "of": cfg.generations,
                                "top_fitness": result.top.fit,
                                "survivors": result.survivors,
                                "culled": result.culled,
                                "population": len(result.evaluated),
                                "top_equity": top_equity,
                                "vaulted_count_total": vaulted_total,
                                "finalists": len(market),
                                "units_done": units_done,
                                "total_units": total_units,
                            }
                        )
                        position = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "tribe": tribe,
                            "gen": result.gen + 1,
                            # next generation's population, so a failed or
                            # cancelled campaign can resume mid-tribe
                            "population": (
                                [g.to_json() for g in result.next_population]
                                if result.next_population
                                else None
                            ),
                        }
                        persist()
                        _wait_while_paused(should_pause, should_stop, emit)
                        if should_stop():
                            raise _Cancelled()

            vaulted_total += _holdout_stage(
                cfg, vault, emit, campaign_id, symbol,
                list(market.values()), tf_data, campaign_windows,
            )
            holdout_done.append(symbol)
            finalists_ckpt.pop(symbol, None)  # judged: nothing left to resume
            persist()
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
