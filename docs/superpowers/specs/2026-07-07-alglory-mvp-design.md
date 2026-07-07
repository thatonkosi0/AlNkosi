# Alglory MVP — Design Spec

**Date:** 2026-07-07
**Status:** Approved by user (with blanket approval for recommended defaults)

## What this is

Alglory is a local-first "genetic bot factory": it breeds populations of forex/CFD
trading strategies with a genetic algorithm, backtests them against MT5 price
history, forward-tests survivors out-of-sample, stores winners in a vault, and
generates ready-to-compile MQL5 Expert Advisors for deployment to MetaTrader 5.

It replicates the *capability set* of algory.app (VagaFX Ltd) but is an original
implementation with its own branding ("ALGLORY"), own copy, and own visual design.
No assets, copy text, logos, or code from Algory are used.

## Decisions locked in

| Decision | Choice |
|---|---|
| First deliverable | End-to-end MVP (real engine + real backtests + vault + live dashboard) |
| Market / execution target | Forex/CFD via MetaTrader 5 (user has MT5 + account installed) |
| Platform | Local web app (Python backend, browser dashboard) |
| Stack | Python 3.11+, FastAPI, official `MetaTrader5` package, DEAP GA, numpy backtester, SQLite vault, vanilla JS + canvas-2D frontend |

## Architecture

One Python package (`alglory`) started via `python -m alglory`:
uvicorn serves the static frontend, a REST API, and a WebSocket event stream.
Campaigns run in a separate worker process; the engine emits structured events
(generation complete, strategy culled, strategy vaulted, campaign phase change)
that the server relays to the browser over WebSocket. Persistence: SQLite
(vault + campaigns) and parquet files (price data cache).

```
browser (canvas dashboard)
   │  REST + WebSocket
FastAPI server ── SQLite vault ── parquet price cache
   │  multiprocessing
engine worker: data → genome → backtest → evolve → vault → deploy
   │
MetaTrader5 terminal (history download; Experts folder for generated EAs)
```

## Components

### 1. `alglory/data` — market data
- Downloads bar history from the user's MT5 terminal (official `MetaTrader5`
  package), caches per symbol/timeframe to parquet.
- Pre-flight sufficiency check before a campaign: history must cover the
  requested in-sample + out-of-sample window or the campaign refuses to start
  with a clear message.
- Cost model: spread (from symbol info) + configurable commission per lot.
- Bundled sample data (a few years of one symbol, CSV→parquet) so tests and
  first-run demos never require a live MT5 connection.

### 2. `alglory/genome` — strategy representation
A strategy is a typed chromosome with gene groups:
- **Signals**: entry conditions from an indicator library — MA cross, RSI
  threshold/cross, MACD, Bollinger touch/breakout, Donchian channel breakout,
  momentum. Parameters (periods, thresholds) are genes.
- **Bias/filters**: long/short/both; higher-timeframe trend filter (MA slope);
  time-of-day session filter; ATR volatility regime filter.
- **Execution**: market entry on signal (MVP); gene reserved for stop/limit later.
- **Management**: stop-loss and take-profit as ATR multiples; optional trailing
  stop; optional max-bars-in-trade time exit.
- **Risk**: fixed-fractional position sizing (percent risk per trade).

**Tribes** = strategy families: trend-following, mean-reversion, breakout,
momentum. Each tribe constrains which signal genes are legal and runs as its own
GA subpopulation. Tribe is stored with the strategy and drives map clustering.

Genomes serialize to/from JSON (vault storage, EA codegen input).

### 3. `alglory/backtest` — vectorized backtester
- numpy bar-based simulation on the chosen timeframe.
- Conservative intra-bar resolution: if a bar's range hits both SL and TP,
  assume SL first (worst case).
- Applies spread on entry, commission per side.
- Outputs: trade list, equity curve, metrics — net profit, max drawdown,
  profit factor, Sharpe (annualized), win rate, trade count.
- Correctness is test-pinned with small hand-computed fixtures.

### 4. `alglory/evolve` — genetic engine (DEAP)
- Per-tribe subpopulations; tournament selection, uniform + parameter-Gaussian
  mutation, single-point crossover on gene groups, elitism.
- **Fitness** computed on the in-sample window only:
  `fitness = net_profit_factor_blend` — profit factor and Sharpe blended, with
  penalties for low trade count (< configurable minimum) and excessive drawdown.
- **Forward test**: after each generation, candidate survivors are evaluated on
  the held-out out-of-sample window (most recent X% of data, default 25%).
  Only strategies passing OOS thresholds (positive net profit, PF ≥ min,
  max DD ≤ cap) are written to the vault.
- Deterministic under a fixed seed (test-pinned).
- Emits events per generation: population stats, top fitness, survivors,
  culled ids, vaulted ids — these drive the dashboard animation.
- Campaign state checkpoints each generation → pause/resume/crash-resume.

### 5. `alglory/vault` — SQLite persistence
- Tables: `strategies` (id, name, tribe, symbol, timeframe, genome JSON,
  IS metrics, OOS metrics, equity curve blob, created_at, campaign_id),
  `campaigns` (config JSON, status, progress, timestamps).
- Query API with filters (symbol, timeframe, tribe, metric ranges, sort).

### 6. `alglory/deploy` — MQL5 EA generation
- Jinja2 template of a complete MQL5 Expert Advisor implementing the indicator
  library + management logic; the genome's rules and parameters are baked in
  as generated code/inputs.
- Guardrail screen before generation: per-trade risk %, daily loss cap,
  max total drawdown cap — with presets: Personal, Prop-firm conservative,
  Prop-firm aggressive. Guardrails are compiled into the EA (it stops trading
  when breached).
- Output written to the MT5 `Experts\Alglory\` folder (auto-detected via the
  MetaTrader5 API, fallback to user-chosen path) + instructions shown in UI.
  User compiles in MetaEditor (one keystroke). Auto-attach is out of scope.

### 7. `alglory/server` — FastAPI
- REST: campaigns (create/list/get/pause/resume/cancel), vault (list/filter/
  get/delete), deploy (validate + generate), status (MT5 connection, data
  cache), insights (aggregations).
- WebSocket `/ws/events`: relays engine events; reattaches to a running
  campaign on reconnect (close browser, reopen, everything reattaches).
- Input validation via pydantic on every endpoint.

### 8. `alglory/ui` — dashboard (vanilla JS + canvas 2D)
Own cyber-terminal aesthetic (original palette/branding, not Algory's assets).
FX quality toggle: FULL / LITE / OFF.
- **Command Deck**: canvas evolution field. Idle: vault strategies as nodes
  clustered by tribe, slowly orbiting. During a campaign: driven by real
  WebSocket events — survivors divide (offspring pinch off), culled nodes fall
  away, vaulted strategies flash toward a vault counter. Side stats: generation,
  survivors, top fitness, vaulted, elapsed/ETA. Embedded terminal streaming
  engine log lines; live equity-curve chart of current generation leaders.
- **Control**: campaign form — symbols, timeframe, population size ("power"),
  generations, tribes to include, IS/OOS split, costs, min-trade + DD
  guardrails; pause/resume/cancel.
- **Vault**: filterable/sortable table; row inspect (genome breakdown by gene
  group, equity curve, IS vs OOS metrics side by side); deploy action.
- **Map**: whole vault as zoomable/pannable clustered galaxy; hover-inspect;
  cluster click → stats callout (count, best/worst, averages).
- **Insights**: aggregate bars — profit by symbol, timeframe, tribe.

## Error handling
- MT5 disconnected → persistent status banner + retry button; API returns
  explicit 503-with-reason, never silent failure.
- Campaign worker is a separate process; a crash marks the campaign FAILED
  with the traceback surfaced in the UI terminal, resumable from the last
  generation checkpoint.
- Data insufficiency, invalid genome params, and form inputs all fail fast
  with human-readable messages.

## Testing (pytest, 80%+ on engine code)
- Backtester: hand-computed fixture scenarios (entries, SL/TP, costs, equity).
- Genome: serialization round-trip, mutation/crossover legality per tribe.
- Evolve: seeded determinism, fitness penalties, OOS gate behavior.
- Vault: CRUD + filters.
- Deploy: generated MQL5 contains expected rules/params for known genomes
  (string-level assertions; compilation is manual).
- End-to-end smoke: tiny campaign on bundled sample data (no MT5 needed)
  produces vaulted strategies and a valid event stream.
- Server: API tests via httpx TestClient.

## Out of scope (phase 2+)
Retrain/optimize vault actions; EA auto-attach to charts; portfolio
correlation heatmaps + diversity grades; WebGL/GPU visuals; multi-user,
auth, hosting; news/fundamental data; walk-forward matrix analysis.

## Risk notes
- Trading leveraged products is high-risk; backtested results do not predict
  future returns. Alglory is a strategy research tool, not financial advice.
  The UI carries this disclaimer.
- The MetaTrader5 Python package only works on Windows with a running
  terminal — acceptable: this machine is Windows and MT5 is installed.
