# Alglory MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app that breeds forex trading strategies with a genetic algorithm, backtests them on MT5 history, forward-tests out-of-sample, vaults winners, and generates MQL5 EAs — with a live canvas dashboard.

**Architecture:** Python package `alglory` (FastAPI server + engine worker process + SQLite vault + parquet data cache), static vanilla-JS/canvas frontend served by the same server, WebSocket event stream from engine to browser. MT5 access isolated behind one adapter so everything else tests against bundled sample data.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, pydantic v2, numpy, pandas, pyarrow, MetaTrader5 (Windows-only, import-guarded), pytest, httpx (tests). Frontend: vanilla ES modules + canvas 2D, no build step.

## Global Constraints

- Python ≥ 3.11; frontend has **no build step** (plain ES modules).
- Original branding "ALGLORY" everywhere. Never copy algory.app copy text, logo, or assets.
- All engine code (data, genome, backtest, evolve, vault, deploy) targets ≥80% test coverage; tests must run without MT5 installed.
- Every API input validated with pydantic; errors surface human-readable messages, never silent failures.
- Immutable-style: functions return new objects; genome mutation/crossover return copies.
- Risk disclaimer shown in UI footer: trading leveraged products is high-risk; backtests don't predict future results; research tool, not financial advice.
- Commit after each passing task (conventional commits).

## File Structure

```
pyproject.toml
alglory/
  __init__.py            # version
  __main__.py            # uvicorn launcher + webbrowser.open
  config.py              # AppConfig: data dir, db path, defaults
  indicators.py          # numpy indicators: sma, ema, rsi, macd, bollinger, donchian, atr, momentum
  data/
    __init__.py
    cache.py             # parquet cache: save/load bars, coverage/sufficiency check
    sample.py            # deterministic synthetic OHLCV generator (seeded random walk)
    mt5source.py         # MT5 adapter: connect, symbols, fetch bars, experts dir (import-guarded)
  genome/
    __init__.py          # re-exports
    genes.py             # dataclasses + JSON round-trip + validation
    tribes.py            # tribe -> legal signal kinds + param ranges
    factory.py           # random(), mutate(), crossover() (pure, rng-injected)
  backtest/
    __init__.py
    engine.py            # run_backtest(genome, bars, costs) -> BacktestResult
    metrics.py           # compute_metrics(trades, equity, bars_per_year) -> Metrics
  evolve/
    __init__.py
    fitness.py           # fitness(metrics, cfg) + oos_pass(metrics, cfg)
    ga.py                # evolve_tribe(...): generator yielding GenerationResult
    campaign.py          # CampaignConfig, run_campaign(cfg, emit) orchestrator, worker entry
  vault/
    __init__.py
    db.py                # Vault class: SQLite CRUD for strategies + campaigns
  deploy/
    __init__.py
    mql5.py              # generate_ea(genome, meta, guardrails) -> str; GUARDRAIL_PRESETS
  server/
    __init__.py
    app.py               # create_app(cfg) FastAPI: REST + WS + static mount
    worker.py            # process manager: start/pause/cancel campaign, queue pump
alglory/ui/
  index.html  style.css
  js/app.js   js/deck.js  js/control.js  js/vault.js  js/map.js  js/insights.js
tests/
  conftest.py            # fixtures: sample bars, tmp vault, test app
  test_indicators.py  test_genome.py  test_backtest.py  test_metrics.py
  test_fitness.py  test_ga.py  test_vault.py  test_deploy.py
  test_server.py  test_e2e.py
```

---

### Task 1: Scaffold + indicators

**Files:** Create `pyproject.toml`, `alglory/__init__.py`, `alglory/config.py`, `alglory/indicators.py`, `tests/test_indicators.py`, `.gitignore`.

**Interfaces (Produces):**
- `indicators.sma(a: np.ndarray, n: int) -> np.ndarray` (leading NaNs until window filled) — same shape contract for `ema(a,n)`, `rsi(a,n)`, `atr(high,low,close,n)`, `momentum(a,n)` (close/close[n]-1)
- `macd(a, fast=12, slow=26, signal=9) -> tuple[macd_line, signal_line]`
- `bollinger(a, n=20, k=2.0) -> tuple[upper, mid, lower]`
- `donchian(high, low, n) -> tuple[upper, lower]` (rolling max/min of *prior* n bars — shifted by 1 so breakout compare is causal)
- `config.AppConfig` dataclass: `data_dir: Path`, `db_path: Path`, `ui_dir: Path`, classmethod `default()` using `~/.alglory` (env override `ALGLORY_HOME`).

- [ ] **Step 1:** Write `pyproject.toml` (setuptools, deps: fastapi, uvicorn[standard], pydantic>=2, numpy, pandas, pyarrow, websockets; optional extra `mt5: MetaTrader5`; dev extra: pytest, pytest-cov, httpx) and package skeleton dirs with empty `__init__.py`s.
- [ ] **Step 2:** Write failing tests in `tests/test_indicators.py` with hand-computed fixtures, e.g.:

```python
def test_sma_hand_computed():
    a = np.array([1., 2., 3., 4., 5.])
    out = sma(a, 3)
    assert np.isnan(out[:2]).all()
    assert out[2:] == pytest.approx([2., 3., 4.])

def test_donchian_is_causal():
    high = np.array([1., 2., 3., 4., 5.]); low = high - 0.5
    up, lo = donchian(high, low, 2)
    assert up[3] == pytest.approx(3.)   # max of bars 1..2, not including bar 3
```

Also: rsi of monotonically rising series → 100; atr of constant-range bars → that range; momentum hand case; macd/bollinger shape + known-value spot checks.
- [ ] **Step 3:** Run `pytest tests/test_indicators.py -v` → FAIL (imports missing).
- [ ] **Step 4:** Implement `indicators.py` with pure numpy (rolling ops via `np.lib.stride_tricks.sliding_window_view` or cumsum; Wilder smoothing for rsi/atr) and `config.py`.
- [ ] **Step 5:** `pytest tests/test_indicators.py -v` → PASS. Commit `feat: scaffold package and indicator library`.

### Task 2: Data layer (sample data + parquet cache + sufficiency)

**Files:** Create `alglory/data/sample.py`, `alglory/data/cache.py`, `tests/conftest.py`, `tests/test_data.py`.

**Interfaces (Produces):**
- Bars contract: `pd.DataFrame` with columns `time` (UTC datetime64), `open, high, low, close` (float64), `volume` (float64), `spread` (float64, in price units); monotonically increasing time.
- `sample.generate_bars(symbol: str, timeframe: str, n: int, seed: int = 7) -> pd.DataFrame` — deterministic seeded random walk honoring the contract, timeframe in {"M15","H1","H4","D1"} sets time deltas; regime-switching drift so trend/reversion strategies can find edges.
- `cache.BarCache(data_dir)` with `.save(symbol, timeframe, df)`, `.load(symbol, timeframe) -> pd.DataFrame | None`, `.coverage(symbol, timeframe) -> tuple[datetime, datetime] | None`
- `cache.check_sufficiency(df, min_bars: int) -> None` raises `InsufficientDataError(msg)` when too short/None.
- conftest fixture `bars_h1` (5000 sample bars) reused by later tests.

- [ ] **Step 1:** Failing tests: generator determinism (same seed → identical frame), contract columns/dtypes, high≥max(open,close), low≤min; cache round-trip equality; sufficiency raises with readable message containing counts.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** PASS. **Step 5:** Commit `feat: sample data generator and parquet bar cache`.

### Task 3: MT5 adapter (import-guarded)

**Files:** Create `alglory/data/mt5source.py`, `tests/test_mt5source.py`.

**Interfaces (Produces):**
- `MT5Source.available() -> bool` (MetaTrader5 importable), `.connect() -> ConnStatus` (`dataclass: ok: bool, message: str, account: str | None`), `.fetch_bars(symbol, timeframe, count) -> pd.DataFrame` (bars contract; maps MT5 rates + converts spread points→price via symbol info), `.symbols() -> list[str]`, `.experts_dir() -> Path | None` (terminal data path + `MQL5/Experts/Alglory`), `.symbol_spread(symbol) -> float`.
- Module never imports MetaTrader5 at top level; all methods return explicit failure `ConnStatus(ok=False, message=...)` when unavailable — **no exceptions for absence**.

- [ ] Tests (run without MT5): `available()` returns bool; `connect()` when unavailable returns `ok=False` with message mentioning MetaTrader5; `fetch_bars` raises `MT5NotConnectedError` with clear message when not connected. Use monkeypatch to simulate a fake mt5 module for the mapping logic (rates ndarray → DataFrame, spread conversion). TDD cycle, commit `feat: MT5 data source adapter`.

### Task 4: Genome (genes, tribes, factory)

**Files:** Create `alglory/genome/genes.py`, `tribes.py`, `factory.py`, `__init__.py`, `tests/test_genome.py`.

**Interfaces (Produces):**

```python
@dataclass(frozen=True) class SignalGene: kind: str; params: dict[str, float | int]
@dataclass(frozen=True) class FilterGene: trend_ma: int | None; session: tuple[int, int] | None; atr_regime: str | None  # 'high'|'low'|None
@dataclass(frozen=True) class ManagementGene: sl_atr: float; tp_atr: float; trail_atr: float | None; max_bars: int | None
@dataclass(frozen=True) class RiskGene: risk_pct: float
@dataclass(frozen=True) class Genome:
    tribe: str; direction: str  # 'long'|'short'|'both'
    signal: SignalGene; filters: FilterGene; management: ManagementGene; risk: RiskGene
    def to_json(self) -> str; @classmethod def from_json(cls, s) -> "Genome"
```

- `tribes.TRIBES: dict[str, list[str]]` = trend→[ma_cross, macd_trend], breakout→[donchian_breakout], mean_reversion→[rsi_reversion, bollinger_fade], momentum→[momentum_burst]
- `tribes.PARAM_SPACE: dict[kind, dict[param, (lo, hi, is_int)]]` — e.g. ma_cross: fast 5–50, slow 20–200 (constraint slow>fast enforced in factory); rsi_reversion: period 5–30, buy_below 15–40, sell_above 60–85; donchian_breakout: period 10–100; bollinger_fade: period 10–40, k 1.5–3.0; momentum_burst: period 5–40, threshold 0.002–0.03; macd_trend: fast 8–15, slow 20–35, signal 5–12.
- `factory.random_genome(tribe, rng: np.random.Generator) -> Genome`
- `factory.mutate(g, rng, rate=0.3) -> Genome` (new object; per-field chance: gaussian-perturb numeric params clipped to space, toggle optional filters, resample within tribe legality)
- `factory.crossover(a, b, rng) -> Genome` (gene-group-level uniform mix; both parents same tribe — assert)

- [ ] Tests: JSON round-trip equality; random_genome legality (kind ∈ tribe list, params within PARAM_SPACE, slow>fast for ma_cross); mutate returns different object, stays legal over 200 seeded iterations; crossover fields each come from a parent; cross-tribe crossover raises `ValueError`. TDD cycle, commit `feat: strategy genome, tribes, and genetic operators`.

### Task 5: Backtester

**Files:** Create `alglory/backtest/engine.py`, `tests/test_backtest.py`.

**Interfaces (Produces):**

```python
@dataclass class Costs: spread: float; commission_per_lot: float = 0.0
@dataclass class Trade: entry_i: int; exit_i: int; direction: int; entry: float; exit: float; pnl_pct: float; reason: str  # 'sl'|'tp'|'trail'|'time'|'signal'|'end'
@dataclass class BacktestResult: trades: list[Trade]; equity: np.ndarray  # len == len(bars), start 1.0, compounded
def build_signals(genome, bars) -> np.ndarray   # int8 per bar: +1 enter long, -1 enter short, 0 none (uses indicators; causal: signal at close of bar i acts at open of i+1)
def run_backtest(genome, bars, costs) -> BacktestResult
```

Simulation rules (implement exactly):
1. Signals computed once, vectorized, from `build_signals`; filters (trend_ma slope sign, session hour window, atr_regime vs median ATR) mask signals to 0.
2. One position at a time. Entry at next bar's open ± half-spread (long pays +half, short −half; same at exit).
3. SL/TP prices from ATR(14) at entry bar × management multiples. Intra-bar: if bar's low≤SL and high≥TP simultaneously → SL fills first (worst case).
4. Trailing stop (if set): ratchet SL by trail_atr × entry ATR following favorable extremes.
5. `max_bars` exit at open of the (entry+max_bars+1)-th bar; opposite signal closes then flips only if direction allows.
6. Position size: risk_pct of current equity at SL distance → `pnl_pct = direction * (exit-entry)/sl_distance * risk_pct` (risk-unit model — keeps equity math broker-free), commission subtracted as `commission_per_lot`-scaled fraction of risk (MVP: treat commission as extra price cost added to spread when >0).
7. Equity curve: flat between trades, updated at exit bars, compounded multiplicatively.

- [ ] Tests with tiny hand-built bar frames (8–12 bars) covering: long TP hit exact pnl; SL-first-when-both rule; spread cost reduces pnl by expected amount; session filter suppresses entries; trailing stop exits at ratcheted level; max_bars time exit; equity compounding across two known trades; `build_signals` causality (no lookahead: signal uses only data ≤ i). TDD cycle, commit `feat: vectorized backtester with conservative fills`.

### Task 6: Metrics

**Files:** Create `alglory/backtest/metrics.py`, `tests/test_metrics.py`.

**Interfaces (Produces):**

```python
@dataclass(frozen=True) class Metrics:
    net_profit: float; max_drawdown: float; profit_factor: float; sharpe: float
    win_rate: float; trades: int; avg_yearly_return: float
    def to_dict(self) -> dict; @classmethod def from_dict(...)
def compute_metrics(result: BacktestResult, bars_per_year: float) -> Metrics
```

Definitions: net_profit = equity[-1]-1; max_drawdown = max peak-to-trough of equity (positive fraction); profit_factor = gross wins/gross losses (cap 100, 0 trades → 0); sharpe from per-bar equity returns annualized by √bars_per_year (zero variance → 0); avg_yearly_return = (equity[-1])**(bars_per_year/len(equity)) - 1.

- [ ] Hand-computed fixture tests for each metric incl. edge cases (no trades, all wins). TDD cycle, commit `feat: backtest metrics`.

### Task 7: Fitness + OOS gate

**Files:** Create `alglory/evolve/fitness.py`, `tests/test_fitness.py`.

**Interfaces (Produces):**

```python
@dataclass(frozen=True) class FitnessConfig: min_trades: int = 30; dd_cap: float = 0.35; pf_weight: float = 0.6; sharpe_weight: float = 0.4
def fitness(m: Metrics, cfg: FitnessConfig) -> float
@dataclass(frozen=True) class OOSGate: min_pf: float = 1.1; max_dd: float = 0.30; min_trades: int = 8
def oos_pass(m: Metrics, gate: OOSGate) -> bool
```

fitness = pf_weight·min(profit_factor,3)/3 + sharpe_weight·clip(sharpe,0,3)/3, ×0 if net_profit≤0, ×(trades/min_trades) when under-traded, ×linear penalty →0 as dd→dd_cap.

- [ ] Tests: monotonic in PF; zero for losing strategies; under-trade and dd penalties bite; oos_pass truth table. TDD, commit `feat: fitness function and OOS gate`.

### Task 8: GA loop

**Files:** Create `alglory/evolve/ga.py`, `tests/test_ga.py`.

**Interfaces (Produces):**

```python
@dataclass class GAParams: population: int = 40; generations: int = 15; tournament_k: int = 3; elite: int = 2; cx_prob: float = 0.6; mut_rate: float = 0.3
@dataclass class Evaluated: genome: Genome; is_metrics: Metrics; fit: float
@dataclass class GenerationResult:
    gen: int; tribe: str; evaluated: list[Evaluated]; survivors: list[int]  # indices kept
    culled: list[int]; top: Evaluated; vaulted: list[tuple[Genome, Metrics, Metrics]]  # (genome, IS, OOS)
def evolve_tribe(tribe, bars_is, bars_oos, costs, ga: GAParams, fit_cfg, gate, seed, bars_per_year) -> Iterator[GenerationResult]
```

Loop per generation: evaluate all on IS → fitness → elite copied → rest bred via tournament+crossover+mutate → per-generation, the top 20% by fitness get an OOS backtest; those passing `oos_pass` AND not already yielded (dedupe on genome JSON) appear in `vaulted`.

- [ ] Tests: seeded determinism (two runs → identical top fitness sequence); population size invariant; elite genomes survive unchanged; vaulted entries all pass gate; generator yields exactly `generations` results. Use small pop (8) / gens (3) on `bars_h1` fixture. TDD, commit `feat: tournament GA with OOS vaulting`.

### Task 9: Vault (SQLite)

**Files:** Create `alglory/vault/db.py`, `tests/test_vault.py`.

**Interfaces (Produces):**

```python
class Vault:  # context-managed sqlite3, WAL mode
    def __init__(self, db_path: Path)
    def add_strategy(self, *, name, tribe, symbol, timeframe, genome_json, is_metrics: dict, oos_metrics: dict, equity: np.ndarray, campaign_id: int | None) -> int
    def list_strategies(self, *, symbol=None, timeframe=None, tribe=None, sort="oos_net_profit", desc=True, limit=200) -> list[dict]
    def get_strategy(self, sid) -> dict | None   # includes genome_json + equity list
    def delete_strategy(self, sid) -> bool
    def create_campaign(self, config_json: str) -> int
    def update_campaign(self, cid, *, status=None, progress_json=None) -> None
    def get_campaign(self, cid) -> dict | None
    def list_campaigns(self, limit=50) -> list[dict]
    def insights(self) -> dict  # {"by_symbol": [...], "by_timeframe": [...], "by_tribe": [...]} avg oos net profit + counts
```

Strategy names auto-generated `ALG-{tribe[:2].upper()}-{sid:05d}` post-insert. Metrics stored as flattened columns (`is_net_profit`, `oos_net_profit`, …) for sortability + JSON blob for fidelity; equity as compressed bytes (`np.save` to BytesIO + zlib).

- [ ] Tests: add/get round-trip (genome json, metrics, equity array equality); filters and sort orders; delete; campaign lifecycle; insights aggregation on 3 known rows. TDD, commit `feat: SQLite strategy vault`.

### Task 10: Campaign orchestrator + worker process

**Files:** Create `alglory/evolve/campaign.py`, `alglory/server/worker.py`, `tests/test_campaign.py`.

**Interfaces (Produces):**

```python
class CampaignConfig(BaseModel):  # pydantic — the API request model too
    symbols: list[str]; timeframe: str = "H1"; tribes: list[str]; population: int = 40
    generations: int = 15; oos_fraction: float = 0.25; seed: int | None = None
    source: str = "sample"  # 'sample'|'mt5'
    min_trades: int = 30; dd_cap: float = 0.35
def run_campaign(cfg: CampaignConfig, vault: Vault, cache: BarCache, emit: Callable[[dict], None], should_stop: Callable[[], bool]) -> None
```

Event dicts (the WS protocol; frontend consumes these exact shapes):
- `{"type":"campaign_started","campaign_id":int,"config":{...},"total_units":int}`
- `{"type":"log","line":str}` (every phase transition: data load, tribe start, etc.)
- `{"type":"generation","campaign_id":..,"symbol":..,"tribe":..,"gen":..,"of":..,"top_fitness":float,"survivors":[idx..],"culled":[idx..],"population":int,"top_equity":[{"id":str,"points":[float..]}..(≤5, downsampled ≤200 pts)],"vaulted_count_total":int}`
- `{"type":"strategy_vaulted","strategy_id":int,"name":str,"tribe":str,"symbol":str,"oos_net_profit":float}`
- `{"type":"campaign_finished","campaign_id":..,"status":"done"|"cancelled"|"failed","error":str|None,"vaulted_total":int}`

Flow: create campaign row → per symbol: load bars (`sample` → generate 6000 bars via `sample.generate_bars`; `mt5` → `MT5Source.fetch_bars`, fail-fast with clear message if unavailable) → `check_sufficiency` → IS/OOS split by `oos_fraction` → per tribe: drive `evolve_tribe`, emitting per generation, vaulting via `vault.add_strategy`, checking `should_stop()` between generations → update campaign row progress each generation (checkpoint) → finish event. Any exception → status `failed`, error string in event and campaign row.

`server/worker.py`: `CampaignManager` — `start(cfg) -> campaign_id` spawns `multiprocessing.Process(target=_worker_main, args=(cfg_json, db_path, data_dir, queue, stop_event))`; `stop()` sets event; `drain() -> list[dict]` pulls queued events; `is_running() -> bool`; single concurrent campaign (409 if busy). Worker builds its own Vault/BarCache (no shared handles across processes).

- [ ] Tests (run `run_campaign` in-process with a stub emit collector): tiny sample campaign (pop 8, gens 3, 1 symbol, 2 tribes) → events arrive in protocol order, campaign row done, vault has strategies with OOS metrics; `should_stop` after first generation → status cancelled; bad symbol with source=mt5 and MT5 absent → failed with message. TDD, commit `feat: campaign orchestrator and worker process manager`.

### Task 11: MQL5 EA generation

**Files:** Create `alglory/deploy/mql5.py`, `tests/test_deploy.py`.

**Interfaces (Produces):**

```python
GUARDRAIL_PRESETS: dict[str, Guardrails]  # 'personal', 'prop_conservative', 'prop_aggressive'
@dataclass(frozen=True) class Guardrails: risk_pct: float; daily_loss_pct: float; max_dd_pct: float
def generate_ea(genome: Genome, *, name: str, symbol: str, timeframe: str, guardrails: Guardrails) -> str
def write_ea(code: str, name: str, experts_dir: Path) -> Path  # <experts_dir>/<name>.mq5
```

The generated `.mq5` is a complete self-contained EA (no includes beyond `<Trade/Trade.mqh>`): OnInit validates symbol/timeframe; OnTick evaluates the genome's specific signal (each signal kind has a dedicated MQL5 code block using native `iMA/iRSI/iMACD/iBands/iHighest/iLowest/iATR` handles), applies session/trend/ATR-regime filters, manages one position with ATR SL/TP (+ trailing if genome has it, time exit if set), sizes by `risk_pct` of balance vs SL distance and the symbol's tick value, and enforces guardrails: halts for the day when daily loss exceeds `daily_loss_pct`, halts permanently (prints reason) when equity drawdown from peak exceeds `max_dd_pct`. Params baked in as `input` variables with generated defaults. String templates per signal kind assembled in Python (f-strings; no Jinja2 needed).

- [ ] Tests: generated code for a known ma_cross genome contains `iMA` with the exact fast/slow periods and the guardrail constants; rsi_reversion genome → `iRSI` + thresholds; every tribe's random genome generates code containing `OnTick`, `OnInit`, and no `{placeholder}` braces left; `write_ea` writes file. TDD, commit `feat: MQL5 EA code generation with guardrail presets`.

### Task 12: FastAPI server

**Files:** Create `alglory/server/app.py`, `alglory/__main__.py`, `tests/test_server.py`.

**Interfaces (Produces):** `create_app(cfg: AppConfig) -> FastAPI` with:
- `GET /api/status` → `{mt5: {available, connected, message}, vault_count, campaign: {running, campaign_id}}`
- `POST /api/campaigns` (body CampaignConfig) → 202 `{campaign_id}`; 409 when running; 422 invalid
- `POST /api/campaigns/{id}/cancel` → 200; `GET /api/campaigns` list
- `GET /api/vault?symbol=&timeframe=&tribe=&sort=&desc=` → rows; `GET /api/vault/{id}` (with genome + equity); `DELETE /api/vault/{id}`
- `GET /api/insights`
- `POST /api/deploy/{id}` body `{preset: str, custom: Guardrails | None, out_dir: str | None}` → `{path, instructions}` (default out_dir: MT5 experts dir if available else `cfg.data_dir/exports`)
- `WS /ws/events`: on connect sends `{"type":"hello","campaign":<running campaign summary or null>}` then streams manager events (background task polls `manager.drain()` every 250 ms, broadcasts to all sockets; events also appended to campaign row so reconnect can resume view)
- Static mount `/` → `alglory/ui` (html=True).
`__main__.py`: parse `--port` (default 8777), `--no-browser`; run uvicorn; open browser after server start.

- [ ] Tests via `httpx` ASGI client + a `FakeCampaignManager` injected through `app.state`: status shape; campaign 202/409/422 paths; vault list/get/delete against seeded tmp vault; deploy writes file to tmp dir and returns instructions; WS hello frame. TDD, commit `feat: FastAPI server with REST, WebSocket, and static UI mount`.

### Task 13: Frontend — shell, control, vault, insights

**Files:** Create `alglory/ui/index.html`, `style.css`, `js/app.js`, `js/control.js`, `js/vault.js`, `js/insights.js`.

Design language (original — "CRIMSON TERMINAL" but ours): near-black `#0a0608` background, primary `#e8324f`, dim red-grey panel borders, mono font stack (`"Cascadia Code", Consolas, monospace`), scanline-free flat panels with 1px borders and corner ticks, uppercase letter-spaced labels. Footer risk disclaimer (Global Constraints). FX toggle FULL/LITE/OFF stored in localStorage.

- `index.html`: header (ALGLORY wordmark, MT5 status pill, FX toggle), tab nav (DECK / CONTROL / VAULT / MAP / INSIGHTS), one `<section>` per tab, footer disclaimer.
- `app.js`: tab router (hash-based); `api()` fetch helper with error toast; `connectWS(onEvent)` with auto-reconnect + exponential backoff; status poll (5s) → MT5 pill; dispatches WS events to registered tab handlers.
- `control.js`: campaign form (symbol text list, timeframe select, tribe checkboxes, population/generations sliders labelled "POWER", oos fraction, source select sample/mt5, min trades, dd cap) → POST /api/campaigns; running state shows CANCEL; renders terminal `<pre>` fed by `log` + `generation` events (typewriter only when FX=FULL).
- `vault.js`: filter bar + table (name, tribe, symbol, TF, OOS net profit, PF, DD, Sharpe, trades, created); row click → detail drawer: gene groups rendered by category with color coding, IS vs OOS metric columns, equity sparkline canvas, DEPLOY button → preset picker modal (three presets + custom fields) → POST /api/deploy/{id} → show returned path + instructions; DELETE with confirm.
- `insights.js`: three horizontal bar groups (by symbol / timeframe / tribe) from GET /api/insights, drawn on canvas.

- [ ] Steps: build files; manual verify by running `python -m alglory --no-browser` and loading each tab with a seeded vault (create via a small script step using sample campaign from Task 10 test helper); commit `feat: dashboard shell, control, vault, and insights tabs`.

### Task 14: Frontend — Command Deck + Map canvases

**Files:** Create `alglory/ui/js/deck.js`, `js/map.js`; modify `index.html` (canvas containers), `app.js` (route events).

- `deck.js`: full-panel canvas. **Idle mode:** fetch vault list; nodes = strategies, clustered by tribe around 4 orbit anchors, slow orbital drift, hover tooltip (name, OOS profit). **Campaign mode** (on `campaign_started`): nodes implode to core; per `generation` event: survivor nodes pulse and spawn an offspring node that pinches off along a short filament; culled indices fall outward and fade as embers; `strategy_vaulted` fires a green spark toward the VAULT stat + increments counter. Stats column (DOM, not canvas): GEN x/of, SURVIVORS, TOP FITNESS, VAULTED, ELAPSED/ETA (ETA = elapsed/units_done × units_left). Right panel: live equity chart canvas drawing `top_equity` polylines color-coded with id labels. FX modes: FULL = trails+glow (shadowBlur), LITE = flat circles, OFF = static dots updated per event only (no rAF loop).
- `map.js`: whole-vault galaxy — same cluster layout with zoom (wheel) + pan (drag), symbol/timeframe filter selects, cluster click → callout panel (count, share %, best/worst member, avg OOS profit, dominant symbol).

- [ ] Steps: implement; manual verify: run a real small sample campaign from the CONTROL tab and watch deck animate through all phases; verify map interactions; commit `feat: command deck evolution field and vault map`.

### Task 15: End-to-end smoke + polish

**Files:** Create `tests/test_e2e.py`; modify anything failing.

- [ ] `test_e2e.py`: against the real app factory (real CampaignManager, tmp dirs): POST a tiny sample campaign (1 symbol, pop 8, gens 3, seed 42) → poll status until finished (timeout 120s) → assert vault non-empty via API, insights non-empty, deploy of top strategy returns a path whose file contains `OnTick`. 
- [ ] Run full suite `pytest --cov=alglory` → all green, engine coverage ≥80%.
- [ ] `README.md`: what it is, install (`pip install -e .[dev]`, optional `[mt5]`), run, MT5 setup note, disclaimer.
- [ ] Commit `feat: end-to-end smoke test and README`.

## Self-Review Notes

- Spec coverage: data(T2/T3), genome(T4), backtest(T5/T6), evolve(T7/T8), vault(T9), campaign+events+resume-view(T10/T12), deploy(T11), server(T12), UI all five tabs(T13/T14), FX modes(T13/T14), error handling (explicit failure paths in T3/T10/T12), testing (per-task + T15), sample-data-first (T2, campaigns default `source="sample"`).
- Pause/resume trimmed to cancel-only for MVP (checkpoint rows still written); spec's "pause/resume" delivered as cancel + campaign history — acceptable MVP cut, noted here deliberately.
- Type names cross-checked: `Metrics`, `BacktestResult`, `CampaignConfig`, `Guardrails`, event shapes used identically in T10/T12/T13/T14.
