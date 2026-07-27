# ALGLORY Live Executor — design spec

**Status:** proposed · **Date:** 2026-07-27

## Problem

MetaTrader 5 has no API to attach an Expert Advisor to a chart or toggle
"Algo Trading" — those are GUI-only. So today ALGLORY automates generate →
write → compile, but the final *attach + enable* step is manual, one chart at a
time. The Python `MetaTrader5` package **can** place orders (`order_send`),
which opens a second path: run the strategy logic in Python and trade the
account directly — no chart, no EA, no manual attach.

## Goal

A `LiveExecutor` that runs one or more vaulted strategies against a *running*
MT5 terminal, placing and managing real orders via `order_send`, using the
**exact same signal code as the backtester** so behaviour is provably
consistent. Deployment (EA files) stays as-is; this is an *alternative* live
path, opt-in and demo-first.

## Non-goals

- Not a replacement for the generated EAs (which keep the "runs even with
  ALGLORY off, fully portable" property). Both paths coexist.
- No auto-start and no auto-arming: a human explicitly arms live trading. The
  system never begins sending orders on its own.

## Reuse (do not re-implement)

| Concern | Existing source of truth |
|---|---|
| Raw signal per bar | `alglory/backtest/engine.py:_raw_signals` / `build_signals` |
| Filters (session, trend, ATR regime) | same engine module |
| Stop/target/trail/breakeven management | mirror `update_stops`/`close_position` in `engine.py` |
| Position sizing (incl. min-lot fallback) | `alglory/analysis/sizing.py:lots_for_risk` |
| Guardrail presets | `alglory/deploy/mql5.py:GUARDRAIL_PRESETS` |
| Bar fetch | `alglory/data/mt5source.py:MT5Source.fetch_bars` |

The single most important invariant: **the executor's signal on bar *i* must
equal the backtester's signal on bar *i*.** This is enforced by a test
(below), not by hope.

## New surface area

### 1. `MT5Source` trading methods (currently read-only)

Add, guarded by `_require_connection()`:

- `place_market(symbol, direction, lots, sl, tp, magic, comment) -> OrderResult`
- `modify_position(symbol, magic, sl, tp) -> bool`
- `close_position(symbol, magic) -> bool`
- `positions(magic=None) -> list[PositionInfo]`
- `account() -> AccountInfo`  (balance, equity, currency)
- `symbol_spec(symbol) -> SymbolSpec`  (tick_value/size, min/step/max lot →
  feeds `sizing.lots_for_risk` with *real* broker numbers instead of the
  `EURUSD_STD` estimate)

All wrap `mt5.order_send` / `mt5.positions_get` / `mt5.account_info` and
degrade to explicit message-carrying failures, matching the module's existing
style. Each managed strategy owns a **unique magic number** so positions are
never confused across strategies on the same symbol.

### 2. `alglory/live/executor.py`

```
StrategyRuntime        # one vaulted strategy: genome, symbol, tf, guard, magic, state
  - peak_equity, day_start_equity, halted, be_armed, entry_bar, entry_atr

LiveExecutor
  - __init__(source: MT5Source, runtimes, *, mode="dry_run"|"demo"|"live",
             poll: BarClock)
  - on_new_bar(rt): fetch bars -> signal -> filters -> guardrails -> size ->
                    order via source (unless dry_run) -> record intent event
  - manage(rt): trail / breakeven / time-exit on the open position
  - tick(): for each rt, manage() always; on a *new closed bar*, on_new_bar()
  - start()/stop()/status(): lifecycle; emits events like CampaignManager
```

Bar-close driven: a `BarClock` polls `fetch_bars(..., 2)` and fires when the
last closed bar's timestamp advances (condition-based, not a fixed sleep).
Guardrails (daily-loss halt, max-DD permanent halt) are enforced in Python,
mirroring `GuardrailsAllowTrading` in the EA.

### 3. Server integration

- `POST /api/executor/arm`  body: `{strategy_ids, mode, preset}` → validates,
  builds runtimes, but **requires `mode` explicitly** and refuses `live`
  unless a separate `confirm_live: true` is present.
- `POST /api/executor/stop` → flat-and-halt kill switch.
- `GET  /api/executor/status` → per-strategy state, open positions, day P&L.
- Runs on a worker thread analogous to `alglory/server/worker.py:CampaignManager`
  (single instance, event queue drained to the WS stream).

## Modes & safety

- **`dry_run`** (default): computes and logs intended orders, sends nothing.
  This is the parity/QA mode.
- **`demo`**: sends orders only if `account().trade_mode` is DEMO; refuses on a
  real account.
- **`live`**: requires `confirm_live` at the API boundary AND a config flag.
  Even then, the *user* is the one arming it — ALGLORY never self-arms.
- Global caps: max concurrent open positions, max aggregate risk, hard
  kill-switch. A too-small account is refused loudly (reuse the sizing floor).
- Every order the executor would send is emitted as an event first, so the UI
  can show intent before (or instead of) execution.

## Testing (London-school, mock the broker)

- `FakeMT5Source`: in-memory account + positions; records `order_send` calls.
- **Parity test (critical):** feed the same bars to `run_backtest` and to the
  executor in `dry_run`; assert the ordered list of entry signals matches
  1:1. Guards the core invariant.
- Sizing/guardrail unit tests reuse `alglory/analysis/sizing.py` and assert the
  daily-loss / max-DD halts trigger.
- Bar-clock test: synthetic feed advancing bar timestamps triggers exactly one
  `on_new_bar` per closed bar.
- Mode-gate tests: `live` refused without `confirm_live`; `demo` refused on a
  non-demo account.

## Risks / open questions

- Requires the MT5 terminal running and logged in for the executor's lifetime
  (same constraint as any auto-trader). If the process dies, open positions are
  left with their broker-side SL/TP — document this.
- Slippage, requotes, partial fills: `place_market` must surface retcodes (the
  EA already logs these; the executor should too).
- One process trading N strategies vs one EA per chart: the executor centralises
  risk (easy global kill-switch) but becomes a single point of failure.
- Legal: this places real orders. It stays opt-in, demo-first, human-armed;
  ALGLORY provides tooling, not financial advice.

## Rollout

1. `MT5Source` trading methods + `symbol_spec` (+ tests with `FakeMT5Source`).
2. `LiveExecutor` in `dry_run` only, with the parity test green.
3. Server endpoints + worker + WS events, `dry_run`/`demo` only.
4. UI: an "Executor" panel showing armed strategies, intents, day P&L, kill.
5. `live` mode behind `confirm_live` + config flag, last.
