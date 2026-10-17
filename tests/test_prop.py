import pandas as pd
import pytest

from alglory.analysis.prop import (
    breaches,
    combined_prop,
    prop_metrics,
    stream_from_records,
    trade_stream,
)
from alglory.backtest.engine import Trade
from alglory.deploy.mql5 import GUARDRAIL_PRESETS


def _stream(rows):
    return pd.DataFrame({"exit_time": pd.to_datetime([t for t, _ in rows]), "ret": [r for _, r in rows]})


def test_rescales_r_multiples_to_preset_risk():
    times = pd.Series(pd.date_range("2024-01-01", periods=5, freq="h"))
    t = Trade(entry_i=0, exit_i=3, direction=1, entry=1.0, exit=0.98, pnl_pct=-0.02, reason="sl")
    s = trade_stream([t], times, genome_risk=0.02, risk_pct=0.005)
    assert s["ret"].iloc[0] == pytest.approx(-0.005)
    assert s["exit_time"].iloc[0] == times[3]


def test_stream_from_records_scales_by_risk():
    s = stream_from_records([{"exit_time": "2024-01-02T10:00:00", "r": -1.0}], risk_pct=0.005)
    assert s["ret"].iloc[0] == pytest.approx(-0.005)
    assert s["exit_time"].iloc[0] == pd.Timestamp("2024-01-02 10:00")


def test_worst_day_groups_by_exit_calendar_day():
    s = _stream([
        ("2024-01-01 09:00", -0.01), ("2024-01-01 10:00", -0.01), ("2024-01-01 11:00", -0.01),
        ("2024-01-02 09:00", 0.05),
    ])
    assert prop_metrics(s).worst_day_loss == pytest.approx(1 - 0.99**3)


def test_drawdown_peak_to_trough():
    s = _stream([("2024-01-01", 0.10), ("2024-01-02", -0.05), ("2024-01-03", -0.05)])
    assert prop_metrics(s).max_drawdown == pytest.approx(1 - 0.95 * 0.95)


def test_all_winning_stream_has_zero_loss():
    m = prop_metrics(_stream([("2024-01-01", 0.01), ("2024-01-02", 0.02)]))
    assert m.worst_day_loss == 0 and m.max_drawdown == 0


def test_combined_prop_catches_joint_breach():
    a = _stream([("2024-01-01 09:00", -0.02)])
    b = _stream([("2024-01-01 10:00", -0.02)])
    g = GUARDRAIL_PRESETS["prop_conservative"]
    assert prop_metrics(a).worst_day_loss < 0.03
    assert prop_metrics(b).worst_day_loss < 0.03
    both = combined_prop([a, b])
    assert both.worst_day_loss > 0.03
    assert breaches(both, g)
    assert breaches(prop_metrics(a), g) == []
