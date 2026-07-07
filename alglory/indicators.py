"""Pure-numpy technical indicators.

All functions return arrays the same length as the input, with NaN in the
leading positions until the indicator's window is filled. All are causal:
the value at index i uses only data at indices <= i (donchian excludes the
current bar so breakout comparisons stay causal).
"""

from __future__ import annotations

import numpy as np


def _rolling_mean(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(a.shape, np.nan)
    if n <= 0 or len(a) < n:
        return out
    c = np.cumsum(np.insert(a, 0, 0.0))
    out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def sma(a: np.ndarray, n: int) -> np.ndarray:
    return _rolling_mean(np.asarray(a, dtype=float), n)


def ema(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = np.full(a.shape, np.nan)
    if len(a) < n:
        return out
    k = 2.0 / (n + 1.0)
    out[n - 1] = a[:n].mean()  # seed with SMA
    for i in range(n, len(a)):
        out[i] = out[i - 1] + k * (a[i] - out[i - 1])
    return out


def _wilder(a: np.ndarray, n: int, seed: float, start: int) -> np.ndarray:
    """Wilder smoothing: out[i] = (out[i-1]*(n-1) + a[i]) / n."""
    out = np.full(a.shape, np.nan)
    out[start] = seed
    for i in range(start + 1, len(a)):
        out[i] = (out[i - 1] * (n - 1) + a[i]) / n
    return out


def rsi(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = np.full(a.shape, np.nan)
    if len(a) <= n:
        return out
    delta = np.diff(a)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = _wilder(gains, n, gains[:n].mean(), n - 1)
    avg_loss = _wilder(losses, n, losses[:n].mean(), n - 1)
    for i in range(n - 1, len(delta)):
        g, l = avg_gain[i], avg_loss[i]
        if l == 0.0:
            out[i + 1] = 100.0 if g > 0 else 50.0
        else:
            rs = g / l
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    out = np.full(high.shape, np.nan)
    if len(high) <= n:
        return out
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )
    smoothed = _wilder(tr, n, tr[1 : n + 1].mean(), n)
    out[n:] = smoothed[n:]
    return out


def momentum(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = np.full(a.shape, np.nan)
    if len(a) <= n:
        return out
    out[n:] = a[n:] / a[:-n] - 1.0
    return out


def macd(
    a: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    line = ema(a, fast) - ema(a, slow)
    sig = np.full(a.shape, np.nan)
    valid = np.where(~np.isnan(line))[0]
    if len(valid) >= signal:
        seg = ema(line[valid], signal)
        sig[valid] = seg
    return line, sig


def bollinger(
    a: np.ndarray, n: int = 20, k: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    mid = _rolling_mean(a, n)
    std = np.full(a.shape, np.nan)
    if len(a) >= n:
        windows = np.lib.stride_tricks.sliding_window_view(a, n)
        std[n - 1 :] = windows.std(axis=1)
    return mid + k * std, mid, mid - k * std


def donchian(high: np.ndarray, low: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling max/min of the *prior* n bars (current bar excluded)."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    up = np.full(high.shape, np.nan)
    lo = np.full(low.shape, np.nan)
    if len(high) <= n:
        return up, lo
    up_windows = np.lib.stride_tricks.sliding_window_view(high, n).max(axis=1)
    lo_windows = np.lib.stride_tricks.sliding_window_view(low, n).min(axis=1)
    up[n:] = up_windows[:-1]
    lo[n:] = lo_windows[:-1]
    return up, lo
