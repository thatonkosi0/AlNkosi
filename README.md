# ALGLORY — your personal bot factory

Alglory breeds forex/CFD trading strategies with a genetic algorithm, backtests
them against price history, forward-tests the survivors out-of-sample, stores
the winners in a local vault, and generates ready-to-compile MetaTrader 5
Expert Advisors — all from a live, local web dashboard.

> **RISK WARNING:** Alglory is strategy research software, not financial
> advice. Trading leveraged products carries a high risk of loss. Backtested
> results — including out-of-sample results — do not predict future
> performance. You may lose more than your initial deposit.

## How it works

1. **CONTROL** — configure a campaign: symbols, timeframe, strategy tribes
   (trend / breakout / mean-reversion / momentum), population ("power"),
   generations, and honesty guardrails (minimum trades, drawdown cap).
2. **Evolution** — each tribe breeds as its own subpopulation: tournament
   selection, crossover, mutation, elitism. Fitness is scored on the
   in-sample window only; every generation's best candidates are then
   forward-tested on a held-out out-of-sample window. Only OOS survivors
   reach the vault.
3. **DECK** — watch it happen live: survivors divide, the culled fall away
   as embers, vaulted strategies spark green. Telemetry, generation-leader
   equity curves, and the raw engine log stream in real time.
4. **VAULT / MAP / INSIGHTS** — inspect every strategy's gene makeup and
   IS-vs-OOS metrics, browse the vault as a clustered galaxy, and see which
   markets, timeframes, and tribes are earning.
5. **DEPLOY** — pick a guardrail preset (personal / prop-firm conservative /
   prop-firm aggressive) and Alglory writes a complete `.mq5` EA with the
   strategy's rules, risk sizing, daily-loss halt, and max-drawdown halt
   baked in. Compile in MetaEditor (F7) and attach to a chart.

## Install

```powershell
pip install -e .[dev]        # core + test deps
pip install -e .[mt5]        # optional: MetaTrader5 bridge (Windows only)
```

Python 3.11+ required. The `MetaTrader5` package requires Windows and a
running, logged-in MT5 terminal; without it Alglory still works fully on
bundled synthetic sample data (source = SAMPLE in the CONTROL tab).

## Run

```powershell
python -m alglory            # starts on http://127.0.0.1:8777 and opens browser
python -m alglory --no-browser --port 9000
```

## Test

```powershell
python -m pytest --cov=alglory
```

The whole suite — including the end-to-end campaign test — runs without MT5
using deterministic synthetic data.

## Architecture

```
browser (canvas dashboard: deck / control / vault / map / insights)
   │  REST + WebSocket
FastAPI server ── SQLite vault ── parquet bar cache
   │  spawned worker process (crash-isolated, keeps running if browser closes)
engine: data → genome → backtest → evolve (GA + OOS gate) → vault → MQL5 codegen
   │
MetaTrader 5 terminal (history download · Experts folder for generated EAs)
```

Key design points:

- **Anti-curve-fitting:** fitness never sees the out-of-sample window; the
  vault gate (positive OOS profit, PF ≥ 1.1, DD ≤ 30%, ≥ 8 OOS trades) does.
- **Conservative fills:** if a bar's range covers both stop and target, the
  stop fills first; spread is paid on both sides.
- **Deterministic:** campaigns with a seed reproduce exactly (test-pinned).
- **Local-first:** no accounts, no cloud; everything lives in `~/.alglory`.
