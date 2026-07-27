import numpy as np
import pandas as pd

from alglory.analysis.activity import activity_for_genome, trade_activity
from alglory.analysis.parity import (
    compare_summaries,
    export_bundle,
    parse_mt5_html_report,
)
from alglory.backtest.engine import BacktestResult, Costs, Trade
from alglory.data.sample import generate_bars
from alglory.deploy.mql5 import GUARDRAIL_PRESETS
from alglory.genome import random_genome

H1_BARS_PER_YEAR = 6240.0


def _flat_bars(n):
    ones = np.ones(n)
    return pd.DataFrame({"open": ones, "high": ones, "low": ones, "close": ones})


def _trade(i):
    return Trade(entry_i=i, exit_i=i + 1, direction=1, entry=1.0, exit=1.01, pnl_pct=0.01, reason="tp")


# ---- activity ---------------------------------------------------------


def test_trade_activity_counts_and_frequency():
    # 12 entries evenly spaced across 6000 H1 bars (~50 weeks) -> ~0.24 trades/wk
    entries = list(range(0, 6000, 500))
    res = BacktestResult(trades=[_trade(i) for i in entries], equity=np.array([1.0]))
    a = trade_activity(res, _flat_bars(6000), "H1")
    assert a["trades"] == 12
    assert 0.2 < a["trades_per_week"] < 0.3
    assert a["median_gap_bars"] == 500
    assert "ACTIVITY" in a["verdict"]  # low/very-low, not "NEVER"


def test_trade_activity_zero_trades_flags_real_bug():
    a = trade_activity(BacktestResult(trades=[], equity=np.array([1.0])), _flat_bars(6000), "H1")
    assert a["trades"] == 0
    assert a["bars_per_trade"] == float("inf")
    assert "NEVER TRADES" in a["verdict"]


def test_activity_for_genome_runs_real_backtest():
    bars = generate_bars("EURUSD", "H1", 4000, seed=7)
    g = random_genome("trend", np.random.default_rng(3))
    a = activity_for_genome(g, bars, Costs(spread=0.0001), "H1")
    assert a["bars"] == 4000
    assert a["trades"] >= 0
    assert a["verdict"]


# ---- parity -----------------------------------------------------------


def test_parse_mt5_html_report_extracts_fields():
    html = (
        "<tr><td>Total Net Profit:</td><td>1 234.56</td></tr>"
        "<tr><td>Profit Factor:</td><td>1.87</td></tr>"
        "<tr><td>Total Trades:</td><td>42</td></tr>"
    )
    r = parse_mt5_html_report(html)
    assert r["trades"] == 42
    assert abs(r["net_profit"] - 1234.56) < 1e-6
    assert abs(r["profit_factor"] - 1.87) < 1e-6


def test_parse_mt5_html_report_missing_fields_are_none():
    r = parse_mt5_html_report("<html>nothing useful here</html>")
    assert r == {"trades": None, "net_profit": None, "profit_factor": None}


def test_compare_summaries_match_within_tolerance():
    py = {"trades": 40, "net_profit": 0.3, "profit_factor": 1.5}
    out = compare_summaries(py, {"trades": 43, "net_profit": 120.0, "profit_factor": 1.4})
    assert out["match"] is True
    assert out["verdict"] == "MATCH"


def test_compare_summaries_diverges_on_sign_and_count():
    py = {"trades": 40, "net_profit": 0.3, "profit_factor": 1.5}
    out = compare_summaries(py, {"trades": 5, "net_profit": -50.0, "profit_factor": 0.8})
    assert out["match"] is False
    assert out["verdict"] == "DIVERGE"
    assert len(out["reasons"]) >= 2  # both trade-count and profit-sign flagged


# ---- sizing (min-lot fallback) ---------------------------------------


def test_risk_sizing_rounds_to_zero_on_small_account_without_fallback():
    from alglory.analysis.sizing import EURUSD_STD, lots_for_risk

    # ~52-pip SL, 1% risk: a $100 standard account can't afford 0.01 lot at risk
    sl = 0.0052
    assert lots_for_risk(100, 0.01, sl, EURUSD_STD, min_lot_fallback=False) == 0.0


def test_min_lot_fallback_lets_small_account_trade_within_cap():
    from alglory.analysis.sizing import EURUSD_STD, lots_for_risk

    sl = 0.0052  # loss/lot ~ $520; min-lot risk on $500 ~ 1.04% (<= 5% cap)
    assert lots_for_risk(500, 0.01, sl, EURUSD_STD, min_lot_fallback=True) == 0.01


def test_min_lot_fallback_refuses_when_min_lot_risk_exceeds_cap():
    from alglory.analysis.sizing import EURUSD_STD, lots_for_risk

    sl = 0.0052  # min-lot risk on a $50 account ~ 10.4% (> 5% cap) -> no trade
    assert lots_for_risk(50, 0.01, sl, EURUSD_STD, min_lot_fallback=True) == 0.0


def test_min_account_thresholds_are_ordered_and_positive():
    from alglory.analysis.sizing import (
        EURUSD_STD,
        min_account_for_risk_sizing,
        min_account_to_trade,
    )

    sl = 0.0052
    floor = min_account_to_trade(sl, EURUSD_STD, max_risk_pct=0.05)
    risk_based = min_account_for_risk_sizing(0.01, sl, EURUSD_STD)
    assert 0 < floor < risk_based  # fallback lets you trade below the risk-sizing floor


def test_export_bundle_writes_all_artifacts(tmp_path):
    bars = generate_bars("EURUSD", "H1", 2000, seed=7)
    g = random_genome("trend", np.random.default_rng(1))
    out = export_bundle(
        g,
        bars,
        name="TestBot",
        symbol="EURUSD",
        timeframe="H1",
        guardrails=GUARDRAIL_PRESETS["personal"],
        costs=Costs(spread=0.0001),
        bars_per_year=H1_BARS_PER_YEAR,
        out_dir=tmp_path / "bundle",
    )
    assert out["ea"].exists() and out["ea"].suffix == ".mq5"
    assert out["bars_csv"].exists()
    assert out["trades_csv"].exists()
    assert out["readme"].exists()
    assert "OnTick" in out["ea"].read_text(encoding="utf-8")
    assert out["python_summary"]["trades"] >= 0
