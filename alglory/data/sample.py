"""Deterministic synthetic OHLCV bars for tests and first-run demos.

A seeded regime-switching random walk: drift flips between trending and
mean-reverting segments so that different strategy tribes can find (and
lose) edges, which keeps GA runs interesting without real market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMEFRAME_DELTAS = {
    "M15": pd.Timedelta(minutes=15),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}

_START = pd.Timestamp("2020-01-06 00:00:00", tz="UTC")
_REGIME_LEN = 250
_BASE_PRICE = 1.10
_BASE_VOL = 0.0009


def generate_bars(symbol: str, timeframe: str, n: int, seed: int = 7) -> pd.DataFrame:
    if timeframe not in TIMEFRAME_DELTAS:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}; expected one of {sorted(TIMEFRAME_DELTAS)}"
        )
    # Mix the symbol into the seed so different symbols get different paths.
    mixed = np.random.SeedSequence([seed, abs(hash(symbol)) % (2**32)])
    rng = np.random.default_rng(mixed)

    n_regimes = n // _REGIME_LEN + 1
    drifts = rng.choice(
        [-0.00035, -0.0001, 0.0, 0.0001, 0.00035], size=n_regimes
    )
    vols = rng.uniform(0.6, 1.6, size=n_regimes) * _BASE_VOL
    drift = np.repeat(drifts, _REGIME_LEN)[:n]
    vol = np.repeat(vols, _REGIME_LEN)[:n]

    rets = drift + rng.normal(0.0, 1.0, n) * vol
    close = _BASE_PRICE * np.exp(np.cumsum(rets))
    open_ = np.concatenate(([_BASE_PRICE], close[:-1]))
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick = np.abs(rng.normal(0.0, 0.5, n)) * vol * close
    high = body_hi + wick
    low = body_lo - np.abs(rng.normal(0.0, 0.5, n)) * vol * close

    volume = rng.uniform(50.0, 500.0, n)
    spread = np.full(n, 0.00012) + rng.uniform(0.0, 0.00004, n)

    time = pd.Series(_START + TIMEFRAME_DELTAS[timeframe] * np.arange(n), name="time")
    return pd.DataFrame(
        {
            "time": time.dt.tz_localize(None),
            "open": open_.astype(np.float64),
            "high": high.astype(np.float64),
            "low": low.astype(np.float64),
            "close": close.astype(np.float64),
            "volume": volume.astype(np.float64),
            "spread": spread.astype(np.float64),
        }
    )
