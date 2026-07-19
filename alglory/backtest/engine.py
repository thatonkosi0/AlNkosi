"""Bar-based backtest engine.

Semantics (kept deliberately conservative):
- Signals are computed at bar close i and act at the open of bar i+1.
- One position at a time; entry price is next bar's open plus half the
  round-trip cost in the trade direction (cost = spread + commission).
- SL/TP are anchored to the raw entry open, distances in ATR(14)-at-signal-bar
  multiples. If one bar's range covers both SL and TP, the SL fills first.
- Trailing stop (if the genome has one) ratchets after each completed bar,
  using that bar's ATR so the trail adapts to current volatility (matching
  the generated MT5 EA, which trails off live ATR).
- Breakeven (if the genome has one) moves the stop to the entry price once
  the bar's favorable extreme has run breakeven_atr * entry-ATR.
- Equity is flat between trades and compounds multiplicatively at exits.
- Position sizing uses a risk-unit model: pnl_pct = R-multiple * risk_pct,
  which keeps equity math broker-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alglory.genome import Genome
from alglory.indicators import atr, bollinger, donchian, ema, macd, momentum, rsi, sma

ATR_PERIOD = 14


@dataclass(frozen=True)
class Costs:
    spread: float
    commission_per_lot: float = 0.0

    @property
    def round_trip(self) -> float:
        return self.spread + self.commission_per_lot


@dataclass(frozen=True)
class Trade:
    entry_i: int
    exit_i: int
    direction: int  # +1 long, -1 short
    entry: float
    exit: float
    pnl_pct: float
    reason: str  # 'sl' | 'tp' | 'trail' | 'be' | 'time' | 'signal' | 'end'


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity: np.ndarray = field(default_factory=lambda: np.array([1.0]))


def _raw_signals(genome: Genome, bars: pd.DataFrame) -> np.ndarray:
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    kind = genome.signal.kind
    p = genome.signal.params
    n = len(bars)
    sig = np.zeros(n, dtype=np.int8)

    if kind == "ma_cross":
        fast = sma(close, int(p["fast"]))
        slow = sma(close, int(p["slow"]))
        above = fast > slow
        prev = np.concatenate(([False], above[:-1]))
        valid = ~np.isnan(fast) & ~np.isnan(slow)
        sig[above & ~prev & valid] = 1
        sig[~above & prev & valid] = -1
    elif kind == "macd_trend":
        line, signal_line = macd(close, int(p["fast"]), int(p["slow"]), int(p["signal"]))
        above = line > signal_line
        prev = np.concatenate(([False], above[:-1]))
        valid = ~np.isnan(line) & ~np.isnan(signal_line)
        sig[above & ~prev & valid] = 1
        sig[~above & prev & valid] = -1
    elif kind == "donchian_breakout":
        up, lo = donchian(high, low, int(p["period"]))
        with np.errstate(invalid="ignore"):
            sig[close > up] = 1
            sig[close < lo] = -1
    elif kind == "rsi_reversion":
        r = rsi(close, int(p["period"]))
        with np.errstate(invalid="ignore"):
            below = r < p["buy_below"]
            above = r > p["sell_above"]
        prev_below = np.concatenate(([False], below[:-1]))
        prev_above = np.concatenate(([False], above[:-1]))
        sig[below & ~prev_below] = 1  # entering oversold -> buy
        sig[above & ~prev_above] = -1
    elif kind == "bollinger_fade":
        up, _, lo = bollinger(close, int(p["period"]), float(p["k"]))
        with np.errstate(invalid="ignore"):
            sig[close < lo] = 1
            sig[close > up] = -1
    elif kind == "momentum_burst":
        m = momentum(close, int(p["period"]))
        with np.errstate(invalid="ignore"):
            sig[m > p["threshold"]] = 1
            sig[m < -p["threshold"]] = -1
    else:
        raise ValueError(f"Unknown signal kind {kind!r}")
    return sig


def build_signals(genome: Genome, bars: pd.DataFrame) -> np.ndarray:
    sig = _raw_signals(genome, bars)
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    f = genome.filters

    if f.trend_ma is not None:
        trend = sma(close, int(f.trend_ma))
        slope = np.concatenate(([np.nan], np.diff(trend)))
        with np.errstate(invalid="ignore"):
            sig = np.where((sig == 1) & ~(slope > 0), 0, sig)
            sig = np.where((sig == -1) & ~(slope < 0), 0, sig)
        sig = sig.astype(np.int8)

    if f.session is not None:
        h0, h1 = f.session
        hours = pd.to_datetime(bars["time"]).dt.hour.to_numpy()
        in_session = (hours >= h0) & (hours < h1)
        sig = np.where(in_session, sig, 0).astype(np.int8)

    if f.atr_regime is not None:
        a = atr(high, low, close, ATR_PERIOD)
        med = pd.Series(a).rolling(100, min_periods=20).median().to_numpy()
        with np.errstate(invalid="ignore"):
            is_high = a > med
        allowed = is_high if f.atr_regime == "high" else ~is_high
        allowed &= ~np.isnan(a) & ~np.isnan(med)
        sig = np.where(allowed, sig, 0).astype(np.int8)

    if genome.direction == "long":
        sig = np.where(sig < 0, 0, sig).astype(np.int8)
    elif genome.direction == "short":
        sig = np.where(sig > 0, 0, sig).astype(np.int8)
    return sig


def run_backtest(
    genome: Genome,
    bars: pd.DataFrame,
    costs: Costs,
    signals: np.ndarray | None = None,
) -> BacktestResult:
    n = len(bars)
    open_ = bars["open"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    atr_series = atr(high, low, close, ATR_PERIOD)
    if signals is None:
        signals = build_signals(genome, bars)

    mgmt = genome.management
    risk_pct = genome.risk.risk_pct
    half_cost = costs.round_trip / 2.0

    equity = np.full(n, 1.0)
    eq = 1.0
    trades: list[Trade] = []

    pos = None  # dict with: dir, entry_i, entry, entry_atr, sl, tp, sl_dist, sl_source, be_armed

    def update_stops(i: int, trail_atr_value: float) -> None:
        """Adjust stops off completed bar i: arm breakeven, ratchet the trail.

        The trail distance uses the supplied ATR (current bar's, so it adapts
        to volatility); when the SL ratchets, the TP shifts outward in
        lockstep so winners keep running behind an adaptive corridor.
        """
        d = pos["dir"]
        if mgmt.breakeven_atr is not None and not pos["be_armed"]:
            trigger = pos["entry"] + d * mgmt.breakeven_atr * pos["entry_atr"]
            reached = high[i] >= trigger if d == 1 else low[i] <= trigger
            if reached:
                pos["be_armed"] = True
                if (d == 1 and pos["entry"] > pos["sl"]) or (
                    d == -1 and pos["entry"] < pos["sl"]
                ):
                    pos["sl"] = pos["entry"]
                    pos["sl_source"] = "be"
        if mgmt.trail_atr is not None:
            dist = mgmt.trail_atr * trail_atr_value
            candidate = high[i] - dist if d == 1 else low[i] + dist
            if (d == 1 and candidate > pos["sl"]) or (d == -1 and candidate < pos["sl"]):
                pos["tp"] += candidate - pos["sl"]
                pos["sl"] = candidate
                pos["sl_source"] = "trail"

    def close_position(i: int, price: float, reason: str) -> None:
        nonlocal eq, pos
        d = pos["dir"]
        entry_eff = pos["entry"] + d * half_cost
        exit_eff = price - d * half_cost
        r_multiple = d * (exit_eff - entry_eff) / pos["sl_dist"]
        pnl = r_multiple * risk_pct
        eq *= 1.0 + pnl
        trades.append(
            Trade(
                entry_i=pos["entry_i"],
                exit_i=i,
                direction=d,
                entry=pos["entry"],
                exit=price,
                pnl_pct=pnl,
                reason=reason,
            )
        )
        pos = None

    for i in range(1, n):
        if pos is not None:
            d = pos["dir"]
            # 1. time exit at this bar's open
            if mgmt.max_bars is not None and (i - pos["entry_i"]) >= mgmt.max_bars:
                close_position(i, open_[i], "time")
            # 2. opposite signal closes at this bar's open
            elif signals[i - 1] == -d:
                close_position(i, open_[i], "signal")
            else:
                # 3. SL/TP intra-bar; SL first when both hit
                sl, tp = pos["sl"], pos["tp"]
                sl_hit = low[i] <= sl if d == 1 else high[i] >= sl
                tp_hit = high[i] >= tp if d == 1 else low[i] <= tp
                if sl_hit:
                    close_position(i, sl, pos["sl_source"])
                elif tp_hit:
                    close_position(i, tp, "tp")
                else:
                    # 4. adjust stops off this completed bar, trailing with
                    # its ATR so the distance adapts to current volatility
                    bar_atr = atr_series[i]
                    if np.isnan(bar_atr):
                        bar_atr = pos["entry_atr"]
                    update_stops(i, bar_atr)

        if pos is None and signals[i - 1] != 0 and not np.isnan(atr_series[i - 1]):
            d = int(signals[i - 1])
            entry = open_[i]
            entry_atr = atr_series[i - 1]
            sl_dist = mgmt.sl_atr * entry_atr
            tp_dist = mgmt.tp_atr * entry_atr
            pos = {
                "dir": d,
                "entry_i": i,
                "entry": entry,
                "entry_atr": entry_atr,
                "sl": entry - d * sl_dist,
                "tp": entry + d * tp_dist,
                "sl_dist": sl_dist,
                "sl_source": "sl",
                "be_armed": False,
            }
            # entry bar itself can hit SL/TP
            sl, tp = pos["sl"], pos["tp"]
            sl_hit = low[i] <= sl if d == 1 else high[i] >= sl
            tp_hit = high[i] >= tp if d == 1 else low[i] <= tp
            if sl_hit:
                close_position(i, sl, "sl")
            elif tp_hit:
                close_position(i, tp, "tp")
            else:
                # stops start adjusting from the entry bar's extreme; the
                # trail anchors to the entry ATR until the next bar completes
                update_stops(i, entry_atr)

        equity[i] = eq

    if pos is not None:
        close_position(n - 1, close[-1], "end")
        equity[-1] = eq

    return BacktestResult(trades=trades, equity=equity)
