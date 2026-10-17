import sys
import types

import numpy as np
import pandas as pd
import pytest

from alglory.data.mt5source import ConnStatus, MT5NotConnectedError, MT5Source


def test_available_returns_bool():
    src = MT5Source()
    assert isinstance(src.available(), bool)


def test_connect_unavailable_is_explicit_failure(monkeypatch):
    src = MT5Source()
    monkeypatch.setattr(src, "available", lambda: False)
    status = src.connect()
    assert isinstance(status, ConnStatus)
    assert status.ok is False
    assert "MetaTrader5" in status.message


def test_fetch_bars_without_connection_raises_clear_error():
    src = MT5Source()
    with pytest.raises(MT5NotConnectedError, match="not connected"):
        src.fetch_bars("EURUSD", "H1", 100)


def _fake_mt5_module():
    mt5 = types.SimpleNamespace()
    mt5.TIMEFRAME_M15 = 15
    mt5.TIMEFRAME_H1 = 16385
    mt5.TIMEFRAME_H4 = 16388
    mt5.TIMEFRAME_D1 = 16408

    rates = np.array(
        [
            (1700000000, 1.10, 1.11, 1.09, 1.105, 100, 12, 0),
            (1700003600, 1.105, 1.12, 1.10, 1.115, 150, 14, 0),
        ],
        dtype=[
            ("time", "<i8"),
            ("open", "<f8"),
            ("high", "<f8"),
            ("low", "<f8"),
            ("close", "<f8"),
            ("tick_volume", "<i8"),
            ("spread", "<i8"),
            ("real_volume", "<i8"),
        ],
    )
    mt5.copy_rates_from_pos = lambda symbol, tf, start, count: rates
    mt5.symbol_info = lambda symbol: types.SimpleNamespace(point=0.00001, spread=12)
    return mt5


def test_fetch_bars_maps_rates_and_converts_spread():
    src = MT5Source()
    src._mt5 = _fake_mt5_module()
    src._connected = True
    df = src.fetch_bars("EURUSD", "H1", 2)
    assert list(df.columns) == ["time", "open", "high", "low", "close", "volume", "spread"]
    assert len(df) == 2
    assert df["close"].iloc[1] == pytest.approx(1.115)
    # spread points -> price units via symbol point size
    assert df["spread"].iloc[0] == pytest.approx(12 * 0.00001)
    assert str(df["time"].dtype).startswith("datetime64")
    assert df["time"].is_monotonic_increasing


def test_fetch_bars_unknown_timeframe():
    src = MT5Source()
    src._mt5 = _fake_mt5_module()
    src._connected = True
    with pytest.raises(ValueError, match="timeframe"):
        src.fetch_bars("EURUSD", "M3", 10)


def _deals_mt5(raw):
    calls = []
    mt5 = types.SimpleNamespace(DEAL_ENTRY_IN=0, DEAL_ENTRY_OUT=1, DEAL_ENTRY_INOUT=2)

    def history_deals_get(date_from, date_to):
        calls.append((date_from, date_to))
        return raw

    mt5.history_deals_get = history_deals_get
    return mt5, calls


def test_deals_since_pads_date_to_for_broker_server_time():
    from datetime import UTC, datetime, timedelta

    deal = lambda t, entry, profit: types.SimpleNamespace(  # noqa: E731
        time=t, magic=990_001, entry=entry, profit=profit, commission=-0.5, swap=0.0)
    raw = [deal(1700000000, 0, 0.0), deal(1700003600, 1, 12.5), deal(1700007200, 2, -3.0)]
    src = MT5Source()
    src._mt5, calls = _deals_mt5(raw)
    src._connected = True
    start = datetime.now(UTC) - timedelta(days=3)
    before = datetime.now(UTC)
    df = src.deals_since(start)
    (date_from, date_to), = calls
    assert date_from == start
    # MT5 deal times are broker server time (UTC+2/+3): look a day ahead
    assert date_to >= before + timedelta(hours=23)
    assert list(df["entry"]) == ["in", "out", "out"]
    assert list(df["profit"]) == [0.0, 12.5, -3.0]
    assert df["time"].iloc[0] == datetime.fromtimestamp(1700000000, UTC)


@pytest.mark.parametrize("raw", [None, ()])
def test_deals_since_empty_result_has_columns(raw):
    from datetime import UTC, datetime

    src = MT5Source()
    src._mt5, _ = _deals_mt5(raw)
    src._connected = True
    df = src.deals_since(datetime.now(UTC))
    assert df.empty
    assert list(df.columns) == ["time", "magic", "entry", "profit", "commission", "swap"]
