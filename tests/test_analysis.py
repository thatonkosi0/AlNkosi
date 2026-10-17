import numpy as np
import pandas as pd

from alglory.analysis.activity import activity_for_genome, trade_activity
from alglory.analysis.parity import (
    compare_summaries,
    compare_trades,
    export_bundle,
    parse_mt5_deals,
    parse_mt5_html_report,
    read_mt5_report,
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


def _row(*cells):
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


_DEALS_HDR = ("Time", "Deal", "Symbol", "Type", "Direction", "Volume", "Price",
              "Order", "Commission", "Swap", "Profit", "Balance", "Comment")

# Shape of an MT5 Strategy Tester report: an Orders table first (must be
# ignored), then Deals with a balance row and SL/TP tags in the comment.
MT5_REPORT = (
    "<table>"
    '<tr><th colspan="13"><div><b>Orders</b></div></th></tr>'
    + _row("Open Time", "Order", "Symbol", "Type", "Volume", "Price", "S / L",
           "T / P", "Time", "State", "Comment")
    + _row("2024.01.02 10:00:05", "2", "EURUSD", "buy", "0.10", "1.1000", "", "",
           "2024.01.02 10:00:05", "filled", "ALGLORY")
    + '<tr><th colspan="13"><div><b>Deals</b></div></th></tr>'
    + _row(*_DEALS_HDR)
    + _row("2024.01.01 00:00:00", "1", "", "balance", "", "", "", "", "0", "0",
           "10 000.00", "10 000.00", "")
    + _row("2024.01.02 10:00:05", "2", "EURUSD", "buy", "in", "0.10", "1.1000",
           "2", "0", "0", "0", "10 000.00", "ALGLORY")
    + _row("2024.01.02 14:30:11", "3", "EURUSD", "sell", "out", "0.10", "1.0950",
           "3", "0", "0", "-50.00", "9 950.00", "sl 1.09500")
    + _row("2024.01.03 09:00:02", "4", "EURUSD", "sell", "in", "0.10", "1.0900",
           "4", "0", "0", "0", "9 950.00", "ALGLORY")
    + _row("2024.01.03 12:00:00", "5", "EURUSD", "buy", "out", "0.10", "1.0800",
           "5", "0", "0", "100.00", "10 050.00", "tp 1.08000")
    + _row("2024.01.04 08:00:01", "6", "EURUSD", "buy", "in", "0.10", "1.0850",
           "6", "0", "0", "0", "10 050.00", "ALGLORY")
    + _row("2024.01.04 11:00:03", "7", "EURUSD", "sell", "out", "0.10", "1.0860",
           "7", "0", "0", "10.00", "10 060.00", "")
    + "</table>"
)


def test_parse_mt5_deals_pairs_in_and_out_into_trades():
    trades = parse_mt5_deals(MT5_REPORT)
    assert [t["direction"] for t in trades] == [1, -1, 1]
    assert [t["exit_reason"] for t in trades] == ["sl", "tp", "close"]
    assert trades[0]["entry_time"] == pd.Timestamp("2024-01-02 10:00:05")
    assert trades[0]["exit_time"] == pd.Timestamp("2024-01-02 14:30:11")


def test_read_mt5_report_handles_utf16_and_utf8(tmp_path):
    # MT5 saves tester reports as UTF-16; a hand-edited copy may be UTF-8
    u16 = tmp_path / "report16.html"
    u16.write_text(MT5_REPORT, encoding="utf-16")
    u8 = tmp_path / "report8.html"
    u8.write_text(MT5_REPORT, encoding="utf-8")
    assert read_mt5_report(u16) == MT5_REPORT
    assert read_mt5_report(u8) == MT5_REPORT


def test_parse_mt5_deals_without_deals_table_is_empty():
    assert parse_mt5_deals("<html>no tables</html>") == []


def _hourly_bars(n):
    return pd.DataFrame({"time": pd.date_range("2024-01-02", periods=n, freq="h")})


def _py_trades(rows):
    """rows: (entry_i, exit_i, direction, reason) as written by export_bundle."""
    return pd.DataFrame(rows, columns=["entry_i", "exit_i", "direction", "reason"])


def _mt5_trade(bars, entry_i, exit_i, direction, reason, lag="5s"):
    # tester fills land a few seconds into the bar
    t = bars["time"]
    return {
        "entry_time": t[entry_i] + pd.Timedelta(lag),
        "exit_time": t[exit_i] + pd.Timedelta(lag),
        "direction": direction,
        "exit_reason": reason,
    }


def test_compare_trades_identical_runs_match():
    bars = _hourly_bars(100)
    py = _py_trades([(10, 15, 1, "sl"), (30, 34, -1, "tp"), (50, 60, 1, "time")])
    mt5 = [_mt5_trade(bars, 10, 15, 1, "sl"), _mt5_trade(bars, 30, 34, -1, "tp"),
           _mt5_trade(bars, 50, 60, 1, "close")]
    out = compare_trades(py, mt5, bars)
    assert out["verdict"] == "MATCH"
    assert out["matched"] == 3
    assert out["first_divergence"] is None


def test_compare_trades_buckets_trail_and_breakeven_as_sl():
    bars = _hourly_bars(100)
    py = _py_trades([(10, 15, 1, "trail"), (30, 34, -1, "be"), (50, 55, 1, "signal")])
    mt5 = [_mt5_trade(bars, 10, 15, 1, "sl"), _mt5_trade(bars, 30, 34, -1, "sl"),
           _mt5_trade(bars, 50, 55, 1, "close")]
    assert compare_trades(py, mt5, bars)["verdict"] == "MATCH"


def test_compare_trades_reports_first_exit_mismatch():
    # a broken trailing stop in the EA: same entries, but the tester rides to TP
    bars = _hourly_bars(100)
    py = _py_trades([(10, 15, 1, "sl"), (30, 34, -1, "trail"), (50, 60, 1, "trail")])
    mt5 = [_mt5_trade(bars, 10, 15, 1, "sl"), _mt5_trade(bars, 30, 40, -1, "tp"),
           _mt5_trade(bars, 50, 70, 1, "tp")]
    out = compare_trades(py, mt5, bars)
    assert out["verdict"] == "DIVERGE"
    assert out["first_divergence"]["entry_i"] == 30
    assert "exit_reason" in out["first_divergence"]["fields"]
    assert "exit_i" in out["first_divergence"]["fields"]


def test_compare_trades_rare_exit_mismatch_still_diverges():
    # a gene that misfires on 1 trade in 20 (5%) is a translation bug, not
    # tick-vs-bar fill noise; the default tolerance must not hide it
    bars = _hourly_bars(500)
    rows = [(i, i + 5, 1, "sl") for i in range(10, 410, 20)]
    py = _py_trades(rows)
    mt5 = [_mt5_trade(bars, e, x, d, r) for e, x, d, r in rows]
    mt5[7]["exit_reason"] = "tp"
    assert compare_trades(py, mt5, bars)["verdict"] == "DIVERGE"


def test_compare_trades_reports_missing_and_extra_entries():
    bars = _hourly_bars(100)
    py = _py_trades([(10, 15, 1, "sl"), (30, 34, -1, "tp")])
    mt5 = [_mt5_trade(bars, 10, 15, 1, "sl"), _mt5_trade(bars, 40, 44, 1, "tp")]
    out = compare_trades(py, mt5, bars)
    assert out["verdict"] == "DIVERGE"
    assert out["missing_in_mt5"] == [30]
    assert out["extra_in_mt5"] == [40]
    assert out["first_divergence"]["entry_i"] == 30


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


def test_compare_summaries_diverges_on_profit_factor_gap():
    # same count, same sign, but the tester's edge is far weaker
    py = {"trades": 40, "net_profit": 0.3, "profit_factor": 1.8}
    out = compare_summaries(py, {"trades": 40, "net_profit": 10.0, "profit_factor": 1.05})
    assert out["verdict"] == "DIVERGE"
    assert any("profit factor" in r for r in out["reasons"])


def test_compare_summaries_profit_factor_within_tolerance_matches():
    py = {"trades": 40, "net_profit": 0.3, "profit_factor": 1.50}
    out = compare_summaries(py, {"trades": 40, "net_profit": 10.0, "profit_factor": 1.40})
    assert out["verdict"] == "MATCH"


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


def test_export_bundle_trades_carry_bar_open_times(tmp_path):
    # MT5 deals are timestamped, the backtest blotter is bar-indexed; the
    # bundle must bridge the two so a per-trade diff can line them up.
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
    trades = pd.read_csv(out["trades_csv"])
    assert len(trades) > 0
    times = pd.to_datetime(bars["time"])
    for row in trades.itertuples():
        assert pd.Timestamp(row.entry_time) == times.iloc[row.entry_i]
        assert pd.Timestamp(row.exit_time) == times.iloc[row.exit_i]
