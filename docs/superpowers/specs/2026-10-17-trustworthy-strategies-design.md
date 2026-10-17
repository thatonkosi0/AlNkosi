# Trustworthy Strategies — Design

**Date:** 2026-10-17
**Status:** draft for approval (builder-checked; owner answers folded in)

## Intent

Alglory should produce *fewer* strategies that can be trusted on a funded
account — either a prop-firm challenge or the owner's own broker account —
rather than many that only look good in the vault.

Today the vault overstates edge. Investigation on 2026-10-17 found:

- Fitness (`evolve/fitness.py`) scores only profit factor and Sharpe; win
  rate, the owner's primary target, plays no part in breeding.
- The out-of-sample gate passes on as few as 8 trades. In campaign 3 (real
  MT5 EURUSD H1 data, corrected costs) 111 of 141 vaulted strategies had
  under 30 OOS trades (median 16).
- The OOS window is used for selection: every generation's top 20% is
  tested on it, so after a campaign it is no longer unseen. There is no
  untouched holdout.
- Measured against the owner's bar, only 4 of 141 campaign-3 strategies win
  >60% of trades, are profitable, and have ≥50 OOS trades.
- The gene vocabulary is thin: six single-indicator entry signals and
  stop/target/trail/breakeven/time exits only. High-win-rate systems
  typically need indicator-based exits (e.g. close a mean-reversion trade
  when RSI returns to 50) and confluence entries.

## The owner's definition of a trustworthy strategy

A strategy qualifies only if **all** hold:

1. **Win rate above 60%** of closed trades.
2. **Profitable overall** after realistic costs. A high win rate with a net
   loss is rejected outright.
3. Proven in this order, each stage on data/time the previous stage never
   influenced:
   1. bred on historical data;
   2. passes the bar on **unseen history**;
   3. passes the bar on the **demo account for a few weeks**;
   4. only then is it a funded-account candidate.
4. For prop-firm use: respects the preset's **daily-loss** and
   **max-drawdown** limits (`GUARDRAIL_PRESETS` in `deploy/mql5.py`:
   `prop_conservative` 3% daily / 8% total, `prop_aggressive` 4% / 9%).

Scope the owner set: **timeframes M15, H1, H4**, keeping the best per
market; **markets: the seven FX majors (EURUSD, GBPUSD, USDJPY, USDCHF,
AUDUSD, USDCAD, NZDUSD), gold (`XAUUSD`) and Nasdaq**. Overnight campaigns
are acceptable.

## Goals

- G1. A chronological three-way split: **breed / select / holdout**, with
  the holdout touched exactly once per finalist.
- G2. Holdout gate = the owner's bar (win rate > 60%, net profit > 0 after
  costs) plus a minimum trade count and, per chosen preset, prop-firm limits.
- G3. Fitness that pushes the search toward profitable high-win-rate
  strategies.
- G4. One campaign spans multiple symbols × timeframes {M15, H1, H4} and
  keeps the best per market.
- G5. Per-market cost profiles (FX, JPY pairs, gold, Nasdaq).
- G6. New genes: indicator exits and a confluence (second-condition) entry
  filter — implemented identically in backtest, live executor and MQL5.
- G7. A strategy **stage** in the vault (bred → holdout-passed → on-demo →
  demo-passed / demo-failed) with demo results pulled from MT5.

## Non-goals

- Real-money automation changes. `live` mode stays as is; this work never
  arms it.
- Machine-learning models or external data feeds.
- A UI framework change; UI edits stay vanilla JS in the existing tabs.
- Re-scoring or migrating existing strategies (campaigns 1–3) beyond
  marking them `legacy`.
- Modelling swap in backtests (my call). Overnight swap will show up in
  demo results; if it decides outcomes it becomes a follow-up.

## Anti-goals (failure even if it "works")

- The holdout influences which strategies get bred or selected — e.g. a
  campaign loop that retries until something passes the holdout.
- A strategy reaches the vault on a high win rate while losing money.
- Vault or UI numbers presented as proof of edge before the demo stage.
- New genes that behave differently in the backtest, live executor or the
  generated EA (silent parity drift — see `analysis/parity.py`).

## Design

### 1. Three-way chronological split (G1)

Per symbol × timeframe, bars split oldest→newest:

| Slice | Default share | Who sees it |
|---|---|---|
| breed | 60% | GA fitness only |
| select | 20% | per-generation candidate screening (today's OOS role) |
| holdout | 20% (most recent) | each campaign finalist, once, at campaign end |

- Finalists = the select-passers accumulated over all generations of a
  symbol/timeframe/tribe unit (today's `result.vaulted` list), deduplicated.
- Each finalist is backtested on the holdout exactly once; pass/fail is
  final for that campaign. No re-breeding or re-selection based on holdout
  results.
- Only holdout-passers enter the vault (stage `holdout_passed`). Select-
  stage metrics are stored too, for diagnosis, but are not the headline.
- **Per-timeframe split** (my call): each timeframe uses 20% of its *own*
  bars as holdout, so every timeframe can reach the 50-trade minimum (a
  shared calendar cut would leave H4 with ~2 months). Each strategy is
  only ever judged on its own timeframe's holdout, so M15 breeding data
  falling inside the H4 holdout period does not leak into any H4 strategy.
- **Holdout reuse across campaigns** (my call): record each campaign's
  holdout windows (symbol, timeframe, start/end) in the campaign row. When a
  new campaign's holdout overlaps one already consumed, the UI and log say
  so plainly ("holdout reused — results are weaker evidence") and the
  strategy carries a `holdout_reused` flag. Not blocked; a fresh holdout
  only exists once new bars arrive.
- **Optimise mode** (`seed_strategy_id`, my call): allowed, but its
  offspring are always flagged `holdout_reused`, because the seed's own
  selection already consumed that holdout.
- **When the holdout runs** (my call): once per *market*, after all of that
  market's timeframe × tribe units have finished. Finalists live in the
  campaign checkpoint (not the vault) until then; resume dedups against
  the checkpoint. A cancelled or failed campaign shows no holdout results
  for unfinished markets, so nobody sees holdout outcomes mid-campaign.

### 2. Holdout gate (G2)

Pass requires all of:

- win rate > 0.60;
- net profit > 0 after the market's cost profile;
- profit factor ≥ 1.1 (my call: keeps the existing gate's margin);
- trades ≥ `min_holdout_trades`, default **50** (my call; shown to and not
  rejected by the owner);
- if the campaign's `target_preset` is a prop preset: worst calendar-day
  loss ≤ `daily_loss_pct` and max drawdown ≤ `max_dd_pct`, computed on the
  holdout equity curve at the preset's `risk_pct` (per-trade P&L rescaled
  from R-multiples, which `Trade` already supports via `pnl_pct / risk_pct`).

`target_preset` is a new campaign setting, default `personal` (no prop
checks beyond the existing `dd_cap`).

### 3. Fitness (G3)

`fitness()` keeps returning 0 when net profit ≤ 0. Otherwise the score adds
a win-rate term:

    score = w_pf * pf_score + w_sh * sharpe_score + w_wr * wr_score
    wr_score = clip((win_rate - 0.45) / (0.70 - 0.45), 0, 1)

Default weights (my call, tunable in `FitnessConfig`): `w_pf=0.4,
w_sh=0.2, w_wr=0.4`. The existing min-trades scaling and drawdown penalty
stay. The select-stage gate also requires win rate > 0.55 (a looser
pre-filter than the holdout's 0.60, so near-misses can still improve).

### 4. Multi-symbol, multi-timeframe campaigns (G4)

- `CampaignConfig.timeframe: str` becomes `timeframes: list[str]` (accept
  the old single `timeframe` field for backward compatibility with stored
  configs and the API).
- Work units = symbol × timeframe × tribe; progress/checkpoint/resume
  extend over the new axis.
- **Keep the best** (my call): after holdout, rank passers per symbol by
  holdout win rate × profit factor (every passer already has ≥50 holdout
  trades, so holdouts of different lengths compare on per-trade quality)
  and vault at most `keep_per_market` (default 3) across all timeframes
  and tribes. Others are discarded with a log line.
- History per pull: up to 20,000 bars per symbol/timeframe (current
  limit). On M15 that is ~10 months — the builder should verify actual
  MetaQuotes-Demo depth per market/timeframe and raise the pull size where
  the broker allows (a 50,000-bar request did not complete within 5
  minutes in the initial survey, so fetch in chunks or accept the cap).
- Nasdaq's symbol name on the owner's broker is still to be confirmed
  (`XAUUSD` confirmed for gold); campaign config takes the broker's
  symbol names verbatim.

### 5. Per-market cost profiles (G5)

Replace the single `min_spread_pips` with a cost profile resolved per
symbol: `{min_spread (price units), commission_per_lot (account
currency, round trip)}`. Commission: **$7 per lot round trip** (owner's
answer). Spread minimums and the non-FX commission are my calls, editable
in config:

| Market | Min spread | Commission / lot (round trip) |
|---|---|---|
| FX majors (non-JPY) | 0.00010 (1 pip) | $7 |
| JPY pairs | 0.010 (1 pip) | $7 |
| XAUUSD | 0.25 | $7 |
| Nasdaq | 1.5 index points | $0 (index CFDs are usually spread-only) |

**Units bug to fix:** today `Costs.round_trip = spread + commission_per_lot`
adds money-per-lot to a price distance. Commission must be converted to
price units through the symbol spec before it is added:
`commission_price = commission_per_lot * tick_size / tick_value`.

Floor stays `max(profile.min_spread, live broker spread, effective_spread(bars))`
as introduced in commit 7b3a14d.

### 6. New genes (G6)

Both optional on `Genome` with `None` defaults so all existing vaulted
genomes still load.

- **ExitGene** — indicator-based exit in addition to SL/TP:
  - `rsi_mid`: close when RSI(period) crosses back through a level
    (default 50);
  - `bb_mid`: close when price crosses the Bollinger middle band;
  - `ma_touch`: close when price crosses an MA(period).
- **ConfirmGene** — a second entry condition that must also hold:
  - `trend`: price above/below MA(period) in trade direction (generalises
    today's `trend_ma` filter; keep `trend_ma` for compatibility);
  - `rsi_zone`: RSI(period) within a band;
  - `adx_min`: ADX(period) ≥ threshold (trend strength) or ≤ threshold
    (ranging), direction set by the gene.
- Each new gene must be implemented in: `backtest/engine.py`
  (`build_signals` / exit loop), the live executor (shares
  `build_signals`; exit check added in `decide`/`_manage_stops`), and
  `deploy/mql5.py` — with tests in all three and the
  `test_parity_holds_*` invariant extended to cover them.
- Mutation/crossover/random creation extend to the new genes; tribes may
  bias which exits they draw (e.g. mean_reversion favours `rsi_mid` /
  `bb_mid`).

### 7. Strategy stages and demo tracking (G7)

- New vault column `stage`: `legacy` (every strategy vaulted before this
  change — campaigns 1–3), `holdout_passed`, `on_demo`, `demo_passed`,
  `demo_failed`.
- **Arming rules:** demo mode may arm `holdout_passed` **and `legacy`**
  strategies (owner's answer). A legacy strategy on demo is labelled
  "legacy — no holdout" everywhere its demo results appear, so its record
  is never mistaken for one that cleared the full ladder. `demo_failed`
  strategies cannot be re-armed (my call).
- Arming moves the strategy to `on_demo` and records the start time.
- Demo runs through the Python live executor only (my call). Results come
  from MT5 deal history filtered by the strategy's magic number
  (`_EXECUTOR_MAGIC_BASE + id`). Separately, the generated EA's hard-coded
  `InpMagic = 990117` becomes per-strategy (`990000 + id`) so a manually
  attached EA never mixes its deals with another strategy's.
- **Per-strategy demo stats:** win rate, net profit, trade count.
- **Account-level prop limits** (owner runs several strategies on one
  account): worst calendar-day loss and max drawdown are measured on the
  **whole demo account** (all Alglory magics combined, in account
  currency, against the balance at the start of each day / the peak), and
  checked against the arming preset's `daily_loss_pct` / `max_dd_pct`. The
  preset is the one chosen at arm time.
- **Before arming several strategies together:** the app combines their
  holdout trade streams (rescaled to the arm preset's `risk_pct`) into one
  daily P&L series and shows the combined worst day and drawdown, warning
  if either breaches the preset. Strategies that pass alone can still
  breach together.
- **Demo pass rule** (my call): at least **3 calendar weeks** since arming
  *and* at least **20 closed trades**, then the owner's bar (win rate >
  60%, net > 0). Fewer than 20 trades after 3 weeks: stays `on_demo`,
  flagged "needs more trades". An account-level prop breach while a
  strategy is `on_demo` marks every strategy that traded that day
  `demo_failed` immediately. Evaluated on demand ("refresh demo results")
  and shown on the VAULT tab; never re-armed or promoted to live
  automatically.
- VAULT tab: stage badge, filter by stage, demo stats next to holdout
  stats, plus an account panel with the combined daily-loss and drawdown
  readings. Legacy strategies hidden by default (a toggle shows them).

## Constraints

- Python 3.11, FastAPI, pandas/numpy, vanilla JS UI — no new dependencies
  unless unavoidable.
- All behaviour changes test-first (`uv run --extra dev pytest`).
- Backward compatible: existing vault rows, genome JSON and campaign
  configs must still load.
- Never arm `live` mode; demo mode keeps its `is_demo` account check.

## Left to the builder

- Exact indicator parameter ranges for the new genes.
- How to chunk MT5 history downloads; caching strategy for multi-TF bars.
- Event/progress payload shapes for the new campaign axes (keep the UI's
  existing event protocol working).
- Whether `keep_per_market` ranking uses holdout fitness or a simpler
  holdout win-rate × profit-factor product, as long as it never feeds back
  into breeding.

## Suggested build order

1. Fitness win-rate term + three-way split + holdout gate (largest effect
   on trustworthiness; small surface).
2. Per-market cost profiles.
3. Multi-symbol × multi-timeframe campaigns with keep-best.
4. Stage column + demo tracking.
5. New genes (largest surface: engine + executor + MQL5 + parity).
