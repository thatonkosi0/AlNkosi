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
