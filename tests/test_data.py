import numpy as np
import pandas as pd
import pytest

from alglory.data.cache import BarCache, InsufficientDataError, check_sufficiency
from alglory.data.sample import generate_bars

EXPECTED_COLUMNS = ["time", "open", "high", "low", "close", "volume", "spread"]


def test_generate_bars_is_deterministic():
    a = generate_bars("EURUSD", "H1", 500, seed=42)
    b = generate_bars("EURUSD", "H1", 500, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_generate_bars_different_seeds_differ():
    a = generate_bars("EURUSD", "H1", 500, seed=1)
    b = generate_bars("EURUSD", "H1", 500, seed=2)
    assert not a["close"].equals(b["close"])


def test_generate_bars_contract(bars_h1):
    df = bars_h1
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 5000
    assert df["time"].is_monotonic_increasing
    assert str(df["time"].dtype).startswith("datetime64")
    for col in ["open", "high", "low", "close", "volume", "spread"]:
        assert df[col].dtype == np.float64
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["spread"] > 0).all()
    assert (df["close"] > 0).all()


def test_generate_bars_timeframe_spacing():
    h1 = generate_bars("EURUSD", "H1", 10, seed=1)
    m15 = generate_bars("EURUSD", "M15", 10, seed=1)
    assert (h1["time"].diff().dropna() == pd.Timedelta(hours=1)).all()
    assert (m15["time"].diff().dropna() == pd.Timedelta(minutes=15)).all()


def test_generate_bars_rejects_unknown_timeframe():
    with pytest.raises(ValueError, match="timeframe"):
        generate_bars("EURUSD", "M3", 10)


def test_cache_round_trip(tmp_path, bars_h1):
    cache = BarCache(tmp_path)
    assert cache.load("EURUSD", "H1") is None
    cache.save("EURUSD", "H1", bars_h1)
    loaded = cache.load("EURUSD", "H1")
    pd.testing.assert_frame_equal(loaded, bars_h1)
    start, end = cache.coverage("EURUSD", "H1")
    assert start == bars_h1["time"].iloc[0]
    assert end == bars_h1["time"].iloc[-1]


def test_coverage_none_when_missing(tmp_path):
    cache = BarCache(tmp_path)
    assert cache.coverage("GBPUSD", "H1") is None


def test_check_sufficiency_passes(bars_h1):
    check_sufficiency(bars_h1, min_bars=1000)


def test_check_sufficiency_raises_with_counts():
    short = generate_bars("EURUSD", "H1", 50, seed=1)
    with pytest.raises(InsufficientDataError, match="50.*1000|1000.*50"):
        check_sufficiency(short, min_bars=1000)


def test_check_sufficiency_raises_on_none():
    with pytest.raises(InsufficientDataError):
        check_sufficiency(None, min_bars=10)
