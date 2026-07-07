import numpy as np
import pytest

from alglory.indicators import (
    atr,
    bollinger,
    donchian,
    ema,
    macd,
    momentum,
    rsi,
    sma,
)


def test_sma_hand_computed():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(a, 3)
    assert np.isnan(out[:2]).all()
    assert out[2:] == pytest.approx([2.0, 3.0, 4.0])


def test_ema_first_value_is_seeded_by_sma():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ema(a, 3)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(2.0)  # seeded with SMA(3)
    k = 2 / (3 + 1)
    assert out[3] == pytest.approx(2.0 + k * (4.0 - 2.0))


def test_rsi_all_gains_is_100():
    a = np.linspace(1.0, 50.0, 50)
    out = rsi(a, 14)
    assert np.isnan(out[:14]).all()
    assert out[-1] == pytest.approx(100.0)


def test_rsi_all_losses_is_0():
    a = np.linspace(50.0, 1.0, 50)
    out = rsi(a, 14)
    assert out[-1] == pytest.approx(0.0, abs=1e-9)


def test_atr_constant_range():
    n = 30
    high = np.full(n, 11.0)
    low = np.full(n, 10.0)
    close = np.full(n, 10.5)
    out = atr(high, low, close, 14)
    assert np.isnan(out[:14]).all()
    assert out[-1] == pytest.approx(1.0)


def test_momentum_hand_computed():
    a = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    out = momentum(a, 2)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(102.0 / 100.0 - 1.0)
    assert out[4] == pytest.approx(104.0 / 102.0 - 1.0)


def test_macd_shapes_and_relationship():
    a = np.cumsum(np.ones(300)) + 100.0
    line, signal = macd(a, fast=12, slow=26, signal=9)
    assert line.shape == a.shape and signal.shape == a.shape
    # On a linear ramp macd line converges to a positive constant
    assert line[-1] > 0
    assert signal[-1] == pytest.approx(line[-1], rel=0.05)


def test_bollinger_bands_bracket_mid():
    rng = np.random.default_rng(0)
    a = 100 + np.cumsum(rng.normal(0, 1, 200))
    up, mid, lo = bollinger(a, 20, 2.0)
    valid = ~np.isnan(mid)
    assert (up[valid] >= mid[valid]).all()
    assert (lo[valid] <= mid[valid]).all()
    assert mid[25] == pytest.approx(a[6:26].mean())


def test_donchian_is_causal():
    high = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    low = high - 0.5
    up, lo = donchian(high, low, 2)
    # value at bar 3 is the max/min of bars 1..2 (prior n bars, excluding current)
    assert up[3] == pytest.approx(3.0)
    assert lo[3] == pytest.approx(1.5)
    assert np.isnan(up[:2]).all()
