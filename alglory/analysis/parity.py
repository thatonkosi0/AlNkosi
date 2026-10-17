"""Backtest <-> MT5 Strategy Tester parity harness (proving Layer 2).

The 129-test suite proves the Python side; it never proves that a vaulted
edge survives inside MetaTrader. This harness closes that gap:

  1. `export_bundle` writes everything needed to reproduce a strategy in MT5's
     Strategy Tester on the SAME bars — the compiled-ready `.mq5`, the exact
     OHLC window as CSV, the Python trade blotter, and a README of tester steps.
  2. You run the Strategy Tester once and save its HTML report.
  3. `parse_mt5_html_report` + `compare_summaries` diff the two result summaries
     (trade count, net profit, profit factor) into a MATCH / DIVERGE verdict.

A DIVERGE verdict is the useful signal: it means the live/tester behaviour does
NOT match the backtest, so the strategy is not yet trustworthy — exactly the
question behind "does it actually do what it's supposed to do".
"""

from __future__ import annotations

import math
import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from alglory.backtest.engine import BacktestResult, Costs, run_backtest
from alglory.backtest.metrics import compute_metrics
from alglory.deploy.mql5 import Guardrails, generate_ea, write_ea
from alglory.genome import Genome


def python_summary(result: BacktestResult, bars_per_year: float) -> dict:
    """Normalised summary of a Python backtest, comparable to an MT5 report."""
    m = compute_metrics(result, bars_per_year)
    return {
        "trades": int(m.trades),
        "net_profit": float(m.net_profit),
        "profit_factor": float(m.profit_factor),
    }


_NUM = r"[-+]?[0-9][0-9\s,]*\.?[0-9]*"


def _find(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else None


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    cleaned = raw.replace(" ", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_mt5_html_report(text: str) -> dict:
    """Extract trade count, net profit and profit factor from an MT5 report.

    Tolerant of MT5's ``Label:</td><td>value`` tester-report layout and of a
    plain ``Label: value`` text dump. Missing fields come back as None.
    """
    net = _to_float(_find(rf"Total\s*Net\s*Profit\D*({_NUM})", text))
    pf = _to_float(_find(rf"Profit\s*Factor\D*({_NUM})", text))
    trades_raw = _find(rf"Total\s*Trades\D*({_NUM})", text)
    if trades_raw is None:
        trades_raw = _find(rf"Total\s*Deals\D*({_NUM})", text)
    trades = _to_float(trades_raw)
    return {
        "trades": int(trades) if trades is not None else None,
        "net_profit": net,
        "profit_factor": pf,
    }


def read_mt5_report(path: str | Path) -> str:
    """Read a saved Strategy Tester report; MT5 writes these as UTF-16."""
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


class _RowCollector(HTMLParser):
    """Flatten every <tr> in a document into a list of cell texts."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _exit_bucket(comment: str) -> str:
    """MT5 tags stop exits 'sl …' / 'tp …'; anything else is an EA close."""
    word = comment.strip().split(" ", 1)[0].lower()
    return word if word in ("sl", "tp") else "close"


def parse_mt5_deals(text: str) -> list[dict]:
    """Rebuild round-trip trades from an MT5 report's Deals table.

    Each ``in`` deal is paired with the next ``out`` deal (the EA holds one
    position at a time). Returns dicts with ``entry_time``, ``exit_time``,
    ``direction`` (1 long / -1 short) and ``exit_reason`` bucketed to
    'sl' | 'tp' | 'close' — the tester's comment cannot tell a trailed or
    breakeven stop from the original SL, nor a time exit from a signal close.
    """
    p = _RowCollector()
    p.feed(text)
    rows = p.rows
    start = next((i for i, r in enumerate(rows) if r == ["Deals"]), None)
    if start is None or start + 1 >= len(rows):
        return []
    header = rows[start + 1]
    col = {name: i for i, name in enumerate(header)}
    if not {"Time", "Type", "Direction", "Comment"} <= col.keys():
        return []

    trades: list[dict] = []
    open_deal: dict | None = None
    for r in rows[start + 2:]:
        if len(r) != len(header):
            break  # next report section
        direction = r[col["Direction"]].lower()
        when = pd.to_datetime(r[col["Time"]], format="%Y.%m.%d %H:%M:%S")
        if direction == "in":
            side = r[col["Type"]].lower()
            open_deal = {"entry_time": when, "direction": 1 if side == "buy" else -1}
        elif direction == "out" and open_deal is not None:
            trades.append(
                {
                    **open_deal,
                    "exit_time": when,
                    "exit_reason": _exit_bucket(r[col["Comment"]]),
                }
            )
            open_deal = None
    return trades


_PY_EXIT_BUCKET = {
    "sl": "sl", "trail": "sl", "be": "sl",  # all are a stop hit inside MT5
    "tp": "tp",
    "time": "close", "signal": "close", "end": "close",  # EA-initiated closes
}


def compare_trades(
    py_trades: pd.DataFrame,
    mt5_trades: list[dict],
    bars: pd.DataFrame,
    *,
    tol: float = 0.03,
) -> dict:
    """Diff the backtest blotter against MT5 trades, one trade at a time.

    Trades are aligned by entry bar (MT5 fill times floored onto ``bars``),
    then direction, exit bar and bucketed exit reason are compared. Unlike
    ``compare_summaries`` this catches management genes (trail, breakeven,
    time exit) that move exits without moving the trade count.

    ``first_divergence`` is the headline: once one trade differs every later
    one can shift, so the earliest mismatch is where to start looking.
    ``tol`` is the fraction of divergent trades still called a MATCH, to
    absorb tick-vs-bar fill noise.
    """
    times = pd.to_datetime(bars["time"]).reset_index(drop=True)

    def bar_of(t) -> int:
        return int(times.searchsorted(pd.Timestamp(t), side="right")) - 1

    mt5_by_entry = {bar_of(t["entry_time"]): t for t in mt5_trades}
    py_by_entry = {int(r.entry_i): r for r in py_trades.itertuples()}

    divergences: list[dict] = []
    matched = 0
    for entry_i, r in py_by_entry.items():
        m = mt5_by_entry.get(entry_i)
        if m is None:
            divergences.append({"entry_i": entry_i, "fields": ["missing_in_mt5"]})
            continue
        fields = []
        if int(r.direction) != m["direction"]:
            fields.append("direction")
        if int(r.exit_i) != bar_of(m["exit_time"]):
            fields.append("exit_i")
        if _PY_EXIT_BUCKET.get(r.reason, r.reason) != m["exit_reason"]:
            fields.append("exit_reason")
        if fields:
            divergences.append({"entry_i": entry_i, "fields": fields})
        else:
            matched += 1
    extra = sorted(set(mt5_by_entry) - set(py_by_entry))
    divergences += [{"entry_i": i, "fields": ["extra_in_mt5"]} for i in extra]
    divergences.sort(key=lambda d: d["entry_i"])

    total = max(len(py_by_entry), len(mt5_by_entry), 1)
    ok = len(divergences) / total <= tol
    return {
        "match": ok,
        "verdict": "MATCH" if ok else "DIVERGE",
        "matched": matched,
        "divergent": len(divergences),
        "first_divergence": divergences[0] if divergences else None,
        "divergences": divergences,
        "missing_in_mt5": sorted(set(py_by_entry) - set(mt5_by_entry)),
        "extra_in_mt5": extra,
    }


def compare_summaries(
    py: dict,
    mt5: dict,
    *,
    trade_tol: float = 0.15,
    pf_tol: float = 0.25,
    profit_sign_must_match: bool = True,
) -> dict:
    """Diff a Python summary against an MT5 summary into a verdict.

    ``trade_tol`` is the allowed relative gap in trade count (defaults to 15%).
    ``pf_tol`` is the allowed relative gap in profit factor (defaults to 25%);
    profit factor is unitless, so unlike net profit it compares directly.
    A profit-sign disagreement (backtest wins, MT5 loses, or vice-versa) is
    always a divergence when ``profit_sign_must_match`` is set.
    """
    reasons: list[str] = []

    p_tr, m_tr = py.get("trades"), mt5.get("trades")
    if m_tr is None:
        reasons.append("MT5 report has no trade count.")
    elif p_tr in (None, 0) and m_tr == 0:
        pass
    else:
        base = max(p_tr or 0, 1)
        rel = abs((m_tr or 0) - (p_tr or 0)) / base
        if rel > trade_tol:
            reasons.append(
                f"trade count differs by {rel:.0%} (python {p_tr} vs mt5 {m_tr}); "
                f"tolerance {trade_tol:.0%}."
            )

    p_np, m_np = py.get("net_profit"), mt5.get("net_profit")
    if profit_sign_must_match and p_np is not None and m_np is not None:
        if (p_np > 0) != (m_np > 0):
            reasons.append(
                f"net-profit sign disagrees (python {p_np:+.4f} vs mt5 {m_np:+.2f})."
            )

    p_pf, m_pf = py.get("profit_factor"), mt5.get("profit_factor")
    if p_pf is not None and m_pf is not None and math.isfinite(p_pf) and p_pf > 0:
        rel = abs(m_pf - p_pf) / p_pf
        if rel > pf_tol:
            reasons.append(
                f"profit factor differs by {rel:.0%} (python {p_pf:.2f} vs mt5 {m_pf:.2f}); "
                f"tolerance {pf_tol:.0%}."
            )

    return {
        "match": not reasons,
        "verdict": "MATCH" if not reasons else "DIVERGE",
        "reasons": reasons,
        "python": py,
        "mt5": mt5,
    }


def export_bundle(
    genome: Genome,
    bars: pd.DataFrame,
    *,
    name: str,
    symbol: str,
    timeframe: str,
    guardrails: Guardrails,
    costs: Costs,
    bars_per_year: float,
    out_dir: Path,
    magic: int = 990117,
) -> dict:
    """Write the full reproduce-in-MT5 bundle; return the emitted paths + summary.

    Produces, under ``out_dir``:
      - ``<name>.mq5``            the EA to compile in MetaEditor
      - ``bars.csv``             the exact OHLC window (MT5 History-Center import)
      - ``python_trades.csv``    the backtest blotter (entry/exit/direction/pnl)
      - ``python_summary.json``  trades / net_profit / profit_factor
      - ``README.txt``           step-by-step Strategy Tester instructions
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_backtest(genome, bars, costs)
    summary = python_summary(result, bars_per_year)

    ea_path = write_ea(
        generate_ea(
            genome, name=name, symbol=symbol, timeframe=timeframe,
            guardrails=guardrails, magic=magic,
        ),
        name,
        out_dir,
    )

    bars_csv = out_dir / "bars.csv"
    bars.to_csv(bars_csv, index=False)

    # bar-open times let compare_trades line the blotter up with MT5 deals
    times = pd.to_datetime(bars["time"]).reset_index(drop=True)
    trades_csv = out_dir / "python_trades.csv"
    pd.DataFrame(
        [
            {
                "entry_i": t.entry_i,
                "exit_i": t.exit_i,
                "entry_time": times[t.entry_i],
                "exit_time": times[t.exit_i],
                "direction": t.direction,
                "entry": t.entry,
                "exit": t.exit,
                "pnl_pct": t.pnl_pct,
                "reason": t.reason,
            }
            for t in result.trades
        ]
    ).to_csv(trades_csv, index=False)

    summary_json = out_dir / "python_summary.json"
    summary_json.write_text(
        pd.Series(summary).to_json(indent=2), encoding="utf-8"
    )

    readme = out_dir / "README.txt"
    readme.write_text(_readme(name, symbol, timeframe, summary), encoding="utf-8")

    return {
        "ea": ea_path,
        "bars_csv": bars_csv,
        "trades_csv": trades_csv,
        "summary_json": summary_json,
        "readme": readme,
        "python_summary": summary,
    }


def _readme(name: str, symbol: str, timeframe: str, summary: dict) -> str:
    return (
        f"ALGLORY parity bundle for {name} ({symbol} {timeframe})\n"
        f"{'=' * 60}\n\n"
        f"Python backtest on bars.csv:\n"
        f"  trades        {summary['trades']}\n"
        f"  net_profit    {summary['net_profit']:+.4f}  (fraction of starting equity)\n"
        f"  profit_factor {summary['profit_factor']:.3f}\n\n"
        "Reproduce in MetaTrader 5 on the SAME bars:\n"
        f"  1. MetaEditor (F4) -> open {name}.mq5 -> compile (F7).\n"
        "  2. MT5 -> View -> Strategy Tester (Ctrl+R).\n"
        f"  3. Expert: {name}. Symbol: {symbol}. Period: {timeframe}.\n"
        "  4. Set the tester date range to match bars.csv (first/last 'time' rows).\n"
        "     Use 'Every tick based on real ticks' for the most faithful run.\n"
        "  5. Keep the EA inputs at their defaults (they encode the vaulted genome).\n"
        "  6. Run, then right-click the Results tab -> 'Save as Report' (HTML).\n"
        "  7. Diff it here (summary first, then trade by trade):\n"
        "       import json, pandas as pd\n"
        "       from alglory.analysis.parity import (read_mt5_report, parse_mt5_html_report,\n"
        "           parse_mt5_deals, compare_summaries, compare_trades)\n"
        "       html = read_mt5_report('report.html')\n"
        "       py  = json.load(open('python_summary.json'))\n"
        "       print(compare_summaries(py, parse_mt5_html_report(html)))\n"
        "       print(compare_trades(pd.read_csv('python_trades.csv'),\n"
        "                            parse_mt5_deals(html), pd.read_csv('bars.csv')))\n\n"
        "A DIVERGE verdict means the tester did NOT reproduce the backtest edge:\n"
        "treat the strategy as unproven until it matches. compare_trades'\n"
        "first_divergence names the first bar where the EA and backtest part ways.\n"
    )
