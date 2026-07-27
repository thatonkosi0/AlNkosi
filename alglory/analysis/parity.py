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

import re
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


def compare_summaries(
    py: dict,
    mt5: dict,
    *,
    trade_tol: float = 0.15,
    profit_sign_must_match: bool = True,
) -> dict:
    """Diff a Python summary against an MT5 summary into a verdict.

    ``trade_tol`` is the allowed relative gap in trade count (defaults to 15%).
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
            genome, name=name, symbol=symbol, timeframe=timeframe, guardrails=guardrails
        ),
        name,
        out_dir,
    )

    bars_csv = out_dir / "bars.csv"
    bars.to_csv(bars_csv, index=False)

    trades_csv = out_dir / "python_trades.csv"
    pd.DataFrame(
        [
            {
                "entry_i": t.entry_i,
                "exit_i": t.exit_i,
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
        "  7. Diff it here:\n"
        "       from alglory.analysis.parity import parse_mt5_html_report, compare_summaries\n"
        "       import json\n"
        "       mt5 = parse_mt5_html_report(open('report.html', encoding='utf-8').read())\n"
        "       py  = json.load(open('python_summary.json'))\n"
        "       print(compare_summaries(py, mt5))\n\n"
        "A DIVERGE verdict means the tester did NOT reproduce the backtest edge:\n"
        "treat the strategy as unproven until it matches.\n"
    )
