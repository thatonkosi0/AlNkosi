# Trustworthy Strategies Implementation Plan (build steps 1–4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Alglory vault only strategies that win >60% of trades and profit after realistic costs on an untouched holdout, across FX majors, gold and Nasdaq on M15/H1/H4, and track them through a demo stage with account-level prop limits.

**Architecture:** Bars split chronologically into breed / select / holdout. The GA breeds on `breed` and screens on `select` (today's OOS role); finalists wait in the campaign checkpoint and meet the holdout once per market after all its timeframe × tribe units finish, then keep-best vaults ≤3 per market with stage `holdout_passed`. Costs come from per-market profiles with commission converted to price units. Demo results are read from MT5 deal history by magic number and judged per strategy (win rate, net) and per account (prop limits).

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, pandas/numpy, SQLite (`alglory/vault/db.py`), MetaTrader5 package, vanilla JS UI (`alglory/ui/`). Tests: `uv run --extra dev pytest`.

**Spec:** `docs/superpowers/specs/2026-10-17-trustworthy-strategies-design.md` (design sections §1–§5 and §7; §6 "New genes" is a separate follow-up plan).

## Global Constraints

- Holdout bar: win rate **> 0.60**, net profit **> 0**, profit factor **≥ 1.1**, trades **≥ 50** (`min_holdout_trades`), plus prop limits when `target_preset` is a prop preset.
- Select-stage gate: existing `OOSGate` plus win rate **> 0.55**.
- Fitness weights: `w_pf=0.4, w_sh=0.2, w_wr=0.4`; `wr_score = clip((win_rate - 0.45) / (0.70 - 0.45), 0, 1)`; fitness stays 0 when net ≤ 0.
- Split defaults: breed 60% / select 20% / holdout 20%, per timeframe on its own bars, chronological, holdout = most recent.
- `keep_per_market` default **3**, ranked by holdout `win_rate × profit_factor`.
- Commission **$7 per lot round trip** for FX, JPY pairs and XAUUSD; **$0** for Nasdaq. Min spread: FX 0.00010, JPY 0.010, XAUUSD 0.25, Nasdaq 1.5. Commission in price units = `commission_per_lot * tick_size / tick_value`.
- Prop presets (existing `GUARDRAIL_PRESETS`): `prop_conservative` daily 3% / total 8%, `prop_aggressive` 4% / 9%; `personal` = no prop checks beyond `dd_cap`.
- Stages: `legacy`, `holdout_passed`, `on_demo`, `demo_passed`, `demo_failed`. Every pre-existing row becomes `legacy`. Demo may arm any stage except `demo_failed`.
- Demo pass: ≥ 21 calendar days since `demo_started_at` AND ≥ 20 closed trades, then win rate > 0.60 and net > 0. Account-level prop breach → every strategy that traded that day → `demo_failed`.
- Executor magic stays `990_000 + strategy_id`; generated EA magic becomes `990_000 + strategy_id`.
- Never arm `live` mode. No new dependencies. Existing vault rows, genome JSON and stored campaign configs must keep loading. Every behaviour change test-first.

## Review Focus

1. **A market/timeframe without enough history** (Nasdaq M15 under `MIN_BARS`, or a holdout that cannot reach 50 trades): that unit is skipped with a log line; the rest of the campaign continues → test in Task 6.
2. **A symbol the broker does not have** (wrong Nasdaq name): the market is skipped with a clear log line naming the symbol; other markets still run → test in Task 6.
3. **Cancel then resume mid-market:** finalists collected before the cancel survive in the checkpoint, the holdout runs exactly once per market, nothing is vaulted twice → test in Task 7.
4. **Old stored campaign configs** (`timeframe: "H1"`, `oos_fraction: 0.25`, no new fields) still parse and resume → test in Task 5.
5. **Demo refresh while MT5 is disconnected:** the endpoint returns 503 with the connection message and changes no stage → test in Task 10.

## File Structure

| File | Responsibility |
|---|---|
| `alglory/evolve/fitness.py` (modify) | win-rate fitness term; select gate win rate; `HoldoutGate` + `holdout_pass` |
| `alglory/analysis/prop.py` (create) | per-strategy and combined prop metrics from trade streams |
| `alglory/evolve/split.py` (create) | three-way chronological split |
| `alglory/evolve/costs.py` (create) | per-market `CostProfile`, commission conversion |
| `alglory/backtest/engine.py` (modify) | `Costs.commission` in price units |
| `alglory/evolve/holdout.py` (create) | finalists, one-shot holdout judging, keep-best |
| `alglory/evolve/campaign.py` (modify) | config fields, timeframe axis, finalists, holdout stage, optimise fix |
| `alglory/vault/db.py` (modify) | stage / holdout / demo columns, campaign holdout windows |
| `alglory/analysis/demo.py` (create) | demo stats, account balance reconstruction, stage decision |
| `alglory/data/mt5source.py`, `alglory/data/fake_mt5.py` (modify) | `deals_since()` |
| `alglory/deploy/mql5.py` (modify) | per-strategy `magic` |
| `alglory/server/app.py` (modify) | arm rules, precheck, demo refresh, deploy magic, list filters |
| `alglory/ui/*.js`, `index.html` (modify) | CONTROL multi-TF/markets/preset; VAULT stages, demo + account panel |

---

### Task 1: Win-rate fitness, select-gate win rate, holdout gate

**Files:**
- Modify: `alglory/evolve/fitness.py`
- Test: `tests/test_fitness.py` (create)

**Interfaces:**
- Produces: `FitnessConfig(min_trades=30, dd_cap=0.35, pf_weight=0.4, sharpe_weight=0.2, wr_weight=0.4)`; `OOSGate(min_pf=1.1, max_dd=0.30, min_trades=8, min_win_rate=0.55)`; `HoldoutGate(min_win_rate=0.60, min_pf=1.1, min_trades=50)`; `holdout_pass(m: Metrics, gate: HoldoutGate) -> bool` (net > 0; win rate strictly greater; pf ≥; trades ≥).

- [ ] **Step 1: Write the failing tests**, building `Metrics(net_profit, max_drawdown, profit_factor, sharpe, win_rate, trades, avg_yearly_return)` directly:
  - `test_fitness_rewards_win_rate_when_profitable`: identical metrics except win_rate 0.50 vs 0.68 → higher fitness for 0.68.
  - `test_fitness_zero_when_losing_even_with_high_win_rate`: net −0.05, win_rate 0.80 → `fitness == 0.0`.
  - `test_wr_score_saturates_at_70pct`: win_rate 0.70 and 0.90 give equal fitness.
  - `test_select_gate_requires_55pct_win_rate`: pf 1.5, 40 trades, win 0.55 → `oos_pass` False; win 0.56 → True.
  - `test_holdout_pass_bar`: (win 0.61, pf 1.2, 50 trades, net 0.02) → True; each of win 0.60, pf 1.09, trades 49, net −0.01 alone → False.
- [ ] **Step 2: Run** `uv run --extra dev pytest tests/test_fitness.py -v` → FAIL (`HoldoutGate` missing).
- [ ] **Step 3: Implement** in `fitness.py`: `wr_weight`, defaults 0.4/0.2/0.4, the `wr_score` term, `min_win_rate` on `OOSGate` checked in `oos_pass`, `HoldoutGate` + `holdout_pass`.
- [ ] **Step 4: Run** `uv run --extra dev pytest -o addopts="" -q` → all pass (adjust only assertions on exact fitness numbers; never weaken a behavioural assertion).
- [ ] **Step 5: Commit** `feat: win-rate fitness term and holdout gate`.

### Task 2: Prop metrics from trade streams

**Files:**
- Create: `alglory/analysis/prop.py`
- Test: `tests/test_prop.py`

**Interfaces:**
- Consumes: `alglory.backtest.engine.Trade` (`exit_i`, `pnl_pct`), `alglory.deploy.mql5.Guardrails`.
- Produces:
  - `@dataclass(frozen=True) PropMetrics(worst_day_loss: float, max_drawdown: float)` — positive fractions of balance.
  - `trade_stream(trades: list[Trade], times: pd.Series, genome_risk: float, risk_pct: float) -> pd.DataFrame` — columns `exit_time` (= `times[exit_i]`), `ret = pnl_pct / genome_risk * risk_pct`.
  - `stream_from_records(records: list[dict], risk_pct: float) -> pd.DataFrame` — from stored `{"exit_time": iso, "r": float}` records (Task 4), `ret = r * risk_pct`.
  - `prop_metrics(stream: pd.DataFrame) -> PropMetrics` — compound `ret` in `exit_time` order; worst day = largest fall from a calendar day's opening balance to its close; drawdown = max peak-to-trough of the per-trade balance.
  - `combined_prop(streams: list[pd.DataFrame]) -> PropMetrics` — concat, sort by `exit_time`, `prop_metrics`.
  - `breaches(m: PropMetrics, g: Guardrails) -> list[str]` — reasons when over `daily_loss_pct` / `max_dd_pct`, else `[]`.

- [ ] **Step 1: Write the failing tests:**
  - `test_rescales_r_multiples_to_preset_risk`: `pnl_pct=-0.02`, genome risk 0.02, preset risk 0.005 → `ret == -0.005`.
  - `test_worst_day_groups_by_exit_calendar_day`: three −1% trades one day, +5% the next → `worst_day_loss ≈ 1 - 0.99**3`.
  - `test_drawdown_peak_to_trough`: +10%, −5%, −5% → `max_drawdown ≈ 1 - 0.95*0.95`.
  - `test_combined_prop_catches_joint_breach`: two streams each −2% the same day → each alone `< 0.03`, combined `> 0.03`, `breaches(combined, GUARDRAIL_PRESETS["prop_conservative"])` non-empty.
- [ ] **Step 2: Run** `uv run --extra dev pytest tests/test_prop.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** `alglory/analysis/prop.py`.
- [ ] **Step 4: Run** the test file → PASS.
- [ ] **Step 5: Commit** `feat: prop-firm daily-loss and drawdown metrics from trade streams`.

### Task 3: Per-market cost profiles and commission in price units

**Files:**
- Create: `alglory/evolve/costs.py`
- Modify: `alglory/backtest/engine.py` (`Costs`), `alglory/evolve/campaign.py` (`_load_bars`, config), `alglory/analysis/robustness.py` (`rescore_vault`), `alglory/server/app.py` (rescore caller), every other `Costs(... commission_per_lot=...)` call site
- Test: `tests/test_costs.py`; update `tests/test_campaign.py`, `tests/test_robustness.py`

**Interfaces:**
- Consumes: `effective_spread(bars, floor)` (engine), `SymbolSpec` (analysis/sizing).
- Produces:
  - `Costs(spread: float, commission: float = 0.0)` — field renamed from `commission_per_lot`, documented as price units; `round_trip = spread + commission`. Find callers with `grep -rn "commission_per_lot" alglory tests`.
  - `@dataclass(frozen=True) CostProfile(min_spread: float, commission_per_lot: float)`.
  - `cost_profile(symbol: str) -> CostProfile`: upper-case name contains `NAS`, `US100`, `USTEC` or `NDX` → (1.5, 0.0); contains `XAU` → (0.25, 7.0); ends with `JPY` → (0.010, 7.0); else (0.00010, 7.0).
  - `resolve_profile(cfg: CampaignConfig, symbol: str) -> CostProfile` — `cfg.cost_overrides[symbol]` keys override `cost_profile(symbol)`.
  - `commission_price(commission_per_lot: float, spec: SymbolSpec) -> float` = `commission_per_lot * spec.tick_size / spec.tick_value`.
  - `CampaignConfig.cost_overrides: dict[str, dict[str, float]] = {}`. Remove `min_spread_pips` and `commission_per_lot` fields; stored configs containing them still parse (ignored).
  - `_load_bars(cfg, symbol, timeframe, cache, emit) -> tuple[pd.DataFrame, Costs]` — MT5: `Costs(spread=effective_spread(bars, floor=max(profile.min_spread, src.symbol_spread(symbol))), commission=commission_price(profile.commission_per_lot, src.symbol_spec(symbol)))`; sample: `Costs(spread=effective_spread(bars), commission=0.0)`. Emit a log line with the spread and commission used.
  - `rescore_vault(..., *, commission: float = 0.0)` (price units).
  - History pull in `_load_bars` (MT5): request **50,000** bars (survey 2026-10-17: M15 reaches Oct 2024, H1 reaches 2018 at that size); when `fetch_bars` raises or returns nothing ("Terminal: Call failed" was seen for every H4 request and for NZDUSD), retry with half the count, down to `MIN_BARS`, then let the error propagate (Task 6 turns it into a skip).

- [ ] **Step 1: Write the failing tests** in `tests/test_costs.py`:
  - `test_load_bars_halves_request_on_terminal_failure`: a fake source whose `fetch_bars` fails above 12,500 bars → `_load_bars` returns 12,500 bars after the 50,000 → 25,000 → 12,500 retries.
  - `test_profiles_by_market`: EURUSD → (0.0001, 7.0); USDJPY → (0.01, 7.0); XAUUSD → (0.25, 7.0); NAS100 and USTEC → (1.5, 0.0).
  - `test_commission_to_price_units`: tick_size 1e-5, tick_value 1.0, $7 → `7e-5`.
  - `test_costs_round_trip_adds_price_units`: `Costs(spread=1e-4, commission=7e-5).round_trip == pytest.approx(1.7e-4)`.
  - `test_cost_overrides_win`: `cost_overrides={"XAUUSD": {"min_spread": 0.4, "commission_per_lot": 0.0}}` → `resolve_profile(cfg, "XAUUSD") == CostProfile(0.4, 0.0)`.
  - `test_old_config_with_commission_fields_parses` (stored config with `commission_per_lot` and `min_spread_pips`).
- [ ] **Step 2: Run** `uv run --extra dev pytest tests/test_costs.py -v` → FAIL.
- [ ] **Step 3: Implement** `costs.py`, the `Costs` rename across callers, the `_load_bars` change, `rescore_vault`; replace the `_spread_floor` tests in `tests/test_campaign.py` (superseded by profiles).
- [ ] **Step 4: Run** the full suite → PASS.
- [ ] **Step 5: Commit** `feat: per-market cost profiles; commission converted to price units`.

### Task 4: Vault schema for stages, holdout and demo

**Files:**
- Modify: `alglory/vault/db.py`
- Test: `tests/test_vault.py`

**Interfaces:**
- Produces:
  - New `strategies` columns added in `_migrate` (idempotent): `stage TEXT`, `holdout_json TEXT`, `ho_win_rate REAL`, `ho_profit_factor REAL`, `ho_trades REAL`, `ho_net_profit REAL`, `holdout_trades_json TEXT` (list of `{"exit_time": iso, "r": float}`), `holdout_reused INTEGER DEFAULT 0`, `demo_started_at TEXT`, `demo_preset TEXT`, `demo_json TEXT`. `_migrate` then sets `stage='legacy'` where `stage IS NULL`.
  - New `campaigns` column `holdout_windows_json TEXT`.
  - `add_strategy(..., stage: str = "legacy", holdout_metrics: dict | None = None, holdout_trades: list[dict] | None = None, holdout_reused: bool = False) -> int`.
  - `set_stage(sid: int, stage: str, *, demo_started_at: str | None = None, demo_preset: str | None = None, demo_json: dict | None = None) -> None` — writes only non-None keywords; ValueError for a stage outside the five.
  - `get_strategy` returns `holdout_trades` (parsed list) and `demo` (parsed dict) alongside the raw columns.
  - `list_strategies(..., stage: str | None = None, include_legacy: bool = True)`; `_SORTABLE` gains the four `ho_*` columns.
  - `set_holdout_windows(cid: int, windows: list[dict]) -> None`; `consumed_holdout_windows(symbol: str, timeframe: str, exclude_campaign: int | None = None) -> list[dict]` (`{"symbol","timeframe","start","end"}`, ISO strings).

- [ ] **Step 1: Write the failing tests:**
  - `test_migration_marks_existing_rows_legacy`: build a DB with the current (pre-change) schema, insert a row, open `Vault` twice → row `stage == "legacy"`, no error on the second open.
  - `test_add_strategy_with_holdout`: stage `holdout_passed`, holdout win 0.63, two trade records → `ho_win_rate == 0.63`, `holdout_trades` round-trips.
  - `test_set_stage_rejects_unknown`: `set_stage(sid, "live")` → ValueError.
  - `test_list_filters_stage_and_hides_legacy`.
  - `test_consumed_holdout_windows`: windows from two campaigns for EURUSD/H1 returned; GBPUSD excluded; `exclude_campaign` honoured.
- [ ] **Step 2: Run** `uv run --extra dev pytest tests/test_vault.py -v` → FAIL.
- [ ] **Step 3: Implement** (extend the existing `_migrate` loop pattern).
- [ ] **Step 4: Run** the full suite → PASS.
- [ ] **Step 5: Commit** `feat: vault stages, holdout and demo columns`.

### Task 5: Three-way split and multi-timeframe campaign config

**Files:**
- Create: `alglory/evolve/split.py`
- Modify: `alglory/evolve/campaign.py` (`CampaignConfig`, unit loop, `_resume_state`, totals, events), `alglory/server/app.py` (any `cfg.timeframe` use)
- Test: `tests/test_split.py`, `tests/test_campaign.py`

**Interfaces:**
- Produces:
  - `split_bars(bars: pd.DataFrame, select_fraction: float = 0.2, holdout_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` — (breed, select, holdout), chronological, each `reset_index(drop=True)`; ValueError if fractions sum ≥ 0.9.
  - `CampaignConfig` gains `timeframes: list[str]` (each in `BARS_PER_YEAR`, min length 1), `select_fraction=0.2`, `holdout_fraction=0.2`, `min_holdout_trades=50` (ge 1), `target_preset="personal"` (key of `GUARDRAIL_PRESETS`), `keep_per_market=3` (ge 1). A `model_validator(mode="before")` maps `timeframe` → `timeframes=[timeframe]` and `oos_fraction` → `select_fraction`; `timeframe`/`oos_fraction` are no longer fields.
  - Unit order symbol → timeframe → tribe → generation; `total_units = len(symbols) * len(timeframes) * len(tribes) * generations`; checkpoint gains `"timeframe"`; `_resume_state(cfg, ckpt) -> tuple[int, int, int, int, list[Genome] | None]` (symbol idx, timeframe idx, tribe idx, start gen, population).
  - `generation` and `strategy_vaulted` events gain `"timeframe"`.

- [ ] **Step 1: Write the failing tests:**
  - `test_split_is_chronological_60_20_20` (1000 bars → 600/200/200; holdout = last 200 rows by time).
  - `test_split_rejects_oversized_fractions`.
  - `test_legacy_config_still_parses` (Review Focus 4): `CampaignConfig(symbols=["EURUSD"], timeframe="H1", tribes=["trend"], oos_fraction=0.25)` → `timeframes == ["H1"]`, `select_fraction == 0.25`.
  - `test_campaign_runs_every_timeframe`: TINY with `timeframes=["H1","H4"]` (sample source) → `generation` events carry both timeframes; `total_units` doubles.
  - `test_resume_state_includes_timeframe`: checkpoint at H4 with `timeframes=["H1","H4"]` → timeframe index 1.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: `split.py`; config fields + validator; timeframe loop calling `_load_bars(cfg, symbol, timeframe, ...)` and `split_bars`; GA receives `bars_breed` / `bars_select` where it took `bars_is` / `bars_oos`; `BARS_PER_YEAR[timeframe]` replaces `cfg.timeframe` uses. Keep today's per-generation vaulting for now — Task 6 replaces it.
- [ ] **Step 4: Run** the full suite → PASS.
- [ ] **Step 5: Commit** `feat: three-way split and multi-timeframe campaigns`.

### Task 6: Finalists, once-per-market holdout, keep-best, skips, optimise fix

**Files:**
- Create: `alglory/evolve/holdout.py`
- Modify: `alglory/evolve/campaign.py`
- Test: `tests/test_holdout.py`, `tests/test_campaign.py`

**Interfaces:**
- Consumes: `HoldoutGate`, `holdout_pass` (Task 1); `trade_stream`, `prop_metrics`, `breaches` (Task 2); `add_strategy(stage=..., holdout_metrics=..., holdout_trades=..., holdout_reused=...)`, `set_holdout_windows`, `consumed_holdout_windows` (Task 4); `split_bars`, config fields (Task 5).
- Produces:
  - `@dataclass Finalist(genome: Genome, tribe: str, timeframe: str, is_metrics: Metrics, select_metrics: Metrics)` with `to_json() -> dict` / `from_json(d) -> Finalist`; `key = genome.to_json() + "|" + timeframe`.
  - `@dataclass HoldoutResult(finalist: Finalist, metrics: Metrics, records: list[dict], passed: bool, reasons: list[str])` — `records` = `[{"exit_time": iso, "r": pnl_pct / genome.risk.risk_pct}]`.
  - `judge_finalist(f: Finalist, holdout_bars: pd.DataFrame, costs: Costs, gate: HoldoutGate, preset: Guardrails | None, bars_per_year: float) -> HoldoutResult` — exactly one `run_backtest`; failed `holdout_pass` conditions and prop `breaches` (when `preset` given) go into `reasons`.
  - `keep_best(results: list[HoldoutResult], k: int) -> list[HoldoutResult]` — passers sorted by `metrics.win_rate * metrics.profit_factor` desc, top k.
  - Campaign flow per market: each generation's select-passers become `Finalist`s in `checkpoint["finalists"][symbol]`, deduped by `key`. After the market's last timeframe × tribe unit: judge each finalist once on its timeframe's holdout (bars/costs cached per timeframe for the market), `keep_best(..., cfg.keep_per_market)`, vault kept ones with `stage="holdout_passed"`, `oos_metrics` = select metrics, `holdout_metrics`, `holdout_trades=records`, `holdout_reused` = `cfg.seed_strategy_id is not None` or any overlap with `consumed_holdout_windows(symbol, tf, exclude_campaign=campaign_id)`; `set_holdout_windows` with this market's windows; log one line per finalist (`"{symbol} {tf} {tribe}: holdout pass"` or `"...: holdout fail — {'; '.join(reasons)}"`), plus `"holdout reused — results are weaker evidence"` when flagged; then the `strategy_vaulted` events; append the symbol to `checkpoint["holdout_done"]`. Robustness scoring at insert stays.
  - `target_preset` → `preset = None if cfg.target_preset == "personal" else GUARDRAIL_PRESETS[cfg.target_preset]`.
  - Optimise fix: pass the computed `initial` to `evolve_tribe` (today line ~282 passes `resume_pop if is_resumed_tribe else None`, so seeding never happens).
  - Skips: `InsufficientDataError` for a symbol/timeframe → log `"{symbol} {tf}: skipped — {msg}"`, continue; a `_load_bars` error for an unknown symbol → log `"{symbol}: skipped — not available from broker ({msg})"`, continue with the next market.

- [ ] **Step 1: Write the failing tests** — `tests/test_holdout.py`:
  - `test_judge_rejects_high_win_rate_loser` (monkeypatch `run_backtest` in `holdout.py` to return trades with 75% wins and net < 0) → `passed False`, a reason mentions net profit.
  - `test_judge_applies_prop_preset`: a 4% losing day → passes with `preset=None`, fails with `prop_conservative` and a daily-loss reason.
  - `test_keep_best_caps_and_orders`: five passers + two failers, k=3 → three passers, descending win×pf.
  - `tests/test_campaign.py`:
    - `test_holdout_runs_once_per_market_after_all_units`: TINY, `timeframes=["H1","H4"]`, `min_holdout_trades=5` → no `strategy_vaulted` before the market's last `generation` event; every vaulted row `stage == "holdout_passed"` and `ho_trades >= 5`.
    - `test_keep_per_market_respected`: `keep_per_market=1` → ≤ 1 vaulted per symbol.
    - `test_insufficient_history_skips_unit_not_campaign` (Review Focus 1): monkeypatch `_load_bars` to raise `InsufficientDataError` for H4 → status `done`, H1 `generation` events present, a log line contains `"H4: skipped"`.
    - `test_unknown_symbol_skips_market` (Review Focus 2): symbols `["EURUSD","NOPE"]`, `_load_bars` raises `RuntimeError` for NOPE → status `done`, log contains `"NOPE: skipped"`.
    - `test_optimise_seeds_population`: spy on `evolve_tribe` → `initial` contains the seed genome.
    - `test_optimise_results_flagged_reused`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `holdout.py` and the campaign flow.
- [ ] **Step 4: Run** the full suite → PASS.
- [ ] **Step 5: Commit** `feat: once-per-market holdout with keep-best; fix optimise seeding`.

### Task 7: Resume and cancel preserve finalists

**Files:**
- Modify: `alglory/evolve/campaign.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Consumes: Task 6 checkpoint keys `finalists`, `holdout_done`.
- Produces: finalists persisted in `progress_json` after every generation; resume reloads them and skips holdout for symbols in `holdout_done`; cancel/fail never runs the holdout for an unfinished market.

- [ ] **Step 1: Write the failing test** `test_cancel_resume_keeps_finalists_and_judges_once` (Review Focus 3): TINY with two timeframes, `stop_after` cancelling after the first timeframe → no `strategy_vaulted` and no holdout log lines; `run_campaign(..., resume_campaign_id=cid)` → one holdout line per finalist, unique vaulted ids, vaulted ≤ `keep_per_market`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite → PASS.
- [ ] **Step 5: Commit** `fix: campaign resume keeps finalists and judges the holdout once`.

### Task 8: Per-strategy EA magic

**Files:**
- Modify: `alglory/deploy/mql5.py`, `alglory/server/app.py` (deploy endpoint), `alglory/analysis/parity.py` (`export_bundle` pass-through)
- Test: `tests/test_deploy.py`, `tests/test_server.py`

**Interfaces:**
- Produces: `generate_ea(genome, *, name, symbol, timeframe, guardrails, magic: int = 990117) -> str` emitting `input long   InpMagic = {magic};`; `/api/deploy/{sid}` passes `magic=_EXECUTOR_MAGIC_BASE + sid`; `export_bundle(..., magic: int = 990117)`.

- [ ] **Step 1: Write the failing tests** `test_ea_magic_is_per_strategy` (`magic=990245` → `"InpMagic = 990245;"` in the code) and `test_deploy_uses_strategy_magic` (API output contains `990000 + sid`).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite → PASS.
- [ ] **Step 5: Commit** `feat: generated EAs use the strategy's own magic number`.

### Task 9: Demo stats and stage decision

**Files:**
- Create: `alglory/analysis/demo.py`
- Modify: `alglory/data/mt5source.py`, `alglory/data/fake_mt5.py`
- Test: `tests/test_demo.py`, `tests/test_live_broker.py`

**Interfaces:**
- Consumes: `PropMetrics` (Task 2).
- Produces:
  - `MT5Source.deals_since(start: datetime) -> pd.DataFrame` — columns `time` (UTC), `magic` (int), `entry` (`"in"`/`"out"`), `profit`, `commission`, `swap` (account currency) from `mt5.history_deals_get(start, datetime.now(UTC))`; `FakeMT5Source.deals_since` returns one `out` deal per simulated close (realised profit) plus `in` deals per open.
  - `demo_stats(deals: pd.DataFrame, magic: int) -> dict` → `{"trades", "wins", "win_rate", "net"}` over that magic's `out` deals; trade result = `profit + commission + swap`.
  - `account_prop(deals: pd.DataFrame, current_balance: float) -> tuple[PropMetrics, dict[date, float]]` — rebuild the balance path backwards from `current_balance` using all deals (every magic and manual trades); measure the Alglory-magic (990000–999999) P&L on that path. Returns the account's `PropMetrics` and each calendar day's loss as a positive fraction of that day's opening balance (days with a net gain map to 0.0).
  - `decide_stage(stats: dict, started_at: datetime, now: datetime, breached: bool) -> tuple[str, str | None]`: breached → (`"demo_failed"`, `"prop limit breached"`); `< 21` days → (`"on_demo"`, None); `≥ 21` days and `< 20` trades → (`"on_demo"`, `"needs more trades"`); else `win_rate > 0.60 and net > 0` → (`"demo_passed"`, None) otherwise (`"demo_failed"`, reason naming the failed condition).

- [ ] **Step 1: Write the failing tests:**
  - `test_demo_stats_counts_only_its_magic_out_deals`.
  - `test_demo_stats_includes_commission_and_swap` (profit +5, commission −7 → counted as a loss).
  - `test_account_prop_combines_strategies`: two magics −$200 each on the same day, balance $10,000 → `worst_day_loss ≈ 0.04`.
  - `test_decide_stage_rules`: one case per branch, including 21 days / 20 trades / win 0.60 → `demo_failed` (must be > 0.60).
  - `test_fake_mt5_records_close_deals` in `tests/test_live_broker.py`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite → PASS.
- [ ] **Step 5: Commit** `feat: demo stats, account-level prop check and stage rules`.

### Task 10: Arm rules, pre-arm combined check, demo refresh API

**Files:**
- Modify: `alglory/server/app.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `stream_from_records`, `combined_prop`, `breaches` (Task 2); `set_stage`, `get_strategy()["holdout_trades"]` (Task 4); `deals_since`, `demo_stats`, `account_prop`, `decide_stage` (Task 9).
- Produces:
  - `POST /api/executor/arm`: 409 `"Strategy {sid} failed its demo; it cannot be re-armed."` for `demo_failed`; in `demo` mode, after a successful arm, `set_stage(sid, "on_demo", demo_started_at=<UTC now ISO, only if unset>, demo_preset=req.preset)`.
  - `POST /api/executor/precheck` body `{strategy_ids: list[int], preset: str}` → `{"worst_day_loss", "max_drawdown", "breaches": [...], "excluded": [{"id", "reason": "legacy — no holdout"}]}` from holdout records rescaled to `GUARDRAIL_PRESETS[preset].risk_pct`.
  - `POST /api/demo/refresh` → connects MT5 (503 with the connection message on failure, nothing written); fetches `deals_since(earliest demo_started_at)`; account check against each strategy's `demo_preset`; per `on_demo` strategy: `demo_stats`, `breached` = it closed a trade on a day where `account_prop`'s daily loss > its `demo_preset`'s `daily_loss_pct`, or the account `max_drawdown` > `max_dd_pct`; `decide_stage`; `set_stage(..., demo_json={**stats, "note": note})`. Returns `{"strategies": [{"id","name","stage","note","legacy","trades","win_rate","net"}], "account": {"worst_day_loss","max_drawdown","breaches"}}`. `legacy` is true when the strategy has no `holdout_json`.

- [ ] **Step 1: Write the failing tests** (existing fake executor / fake MT5 fixtures):
  - `test_arm_refuses_demo_failed`; `test_arm_demo_sets_on_demo_and_start_time`; `test_arm_allows_legacy`.
  - `test_precheck_flags_joint_breach_and_excludes_legacy`.
  - `test_demo_refresh_promotes_and_keeps_waiting` (strategy A: 25 trades, 70% wins, 22 days → `demo_passed`; strategy B: 10 trades, 22 days → `on_demo`, note "needs more trades").
  - `test_demo_refresh_503_when_mt5_down_changes_nothing` (Review Focus 5).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite → PASS.
- [ ] **Step 5: Commit** `feat: demo-stage API — arm rules, combined prop precheck, refresh`.

### Task 11: UI — campaign scope and vault stages

**Files:**
- Modify: `alglory/ui/index.html`, `alglory/ui/control.js`, `alglory/ui/vault.js`, `alglory/ui/executor.js`, `alglory/server/app.py` (list endpoint query params)
- Test: `tests/test_server.py` (API params); UI checked in the running app

**Interfaces:**
- Consumes: Tasks 5, 6, 10.
- Produces:
  - CONTROL: M15/H1/H4 checkboxes (all on by default); symbols prefilled `EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,USDCAD,NZDUSD,XAUUSD,USTEC` (`USTEC` is Nasdaq on the owner's MetaQuotes-Demo account); target-preset select; `keep_per_market` and `min_holdout_trades` inputs; posts `timeframes`.
  - VAULT: stage badge, stage filter, "show legacy" toggle (off by default), holdout win %, PF and trades as headline columns, "legacy — no holdout" label, "Refresh demo results" button rendering per-strategy demo stats and the account panel (worst day, drawdown, breaches; the 503 message when MT5 is closed).
  - Arm flow calls `/api/executor/precheck` and shows breaches/exclusions before confirming.
  - `GET /api/strategies?stage=&include_legacy=` → `Vault.list_strategies(stage=..., include_legacy=...)`.

- [ ] **Step 1: Write the failing test** `test_list_strategies_stage_filter_and_legacy_toggle`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the params and the UI.
- [ ] **Step 4: Run** the full suite → PASS; then start the app (`uv run --extra mt5 python -m alglory --no-browser --port 8777`) and check: a 2-timeframe sample campaign starts from CONTROL; VAULT hides legacy by default and shows stages; demo refresh renders the account panel or the 503 message.
- [ ] **Step 5: Commit** `feat(ui): multi-timeframe campaign controls and vault stage view`.

### Task 12: End-to-end check on real data

**Files:** none (verification; findings reported to the owner).

- [ ] **Step 1:** `uv run --extra dev pytest -o addopts="" -q` → all pass.
- [ ] **Step 2:** Start the app; run an MT5 campaign: symbols `EURUSD,XAUUSD`, timeframes M15/H1/H4, all tribes, population 40, generations 15, `target_preset=prop_conservative`.
- [ ] **Step 3:** Confirm: status `done`; every new row `stage=holdout_passed`, `ho_win_rate > 0.60`, `ho_net_profit > 0`, `ho_trades ≥ 50`, ≤ 3 per symbol; campaign row has `holdout_windows_json`; log lists each finalist's holdout result and any skipped units.
- [ ] **Step 4:** Report finalists / holdout passers / kept per symbol and timeframe. Zero passers is a valid result, not a failure.
