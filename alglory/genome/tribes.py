"""Tribe definitions: which signal kinds each strategy family may use,
and the legal parameter space per signal kind."""

from __future__ import annotations

TRIBES: dict[str, list[str]] = {
    "trend": ["ma_cross", "macd_trend"],
    "breakout": ["donchian_breakout"],
    "mean_reversion": ["rsi_reversion", "bollinger_fade"],
    "momentum": ["momentum_burst"],
}

# param -> (lo, hi, is_int)
PARAM_SPACE: dict[str, dict[str, tuple[float, float, bool]]] = {
    "ma_cross": {"fast": (5, 50, True), "slow": (20, 200, True)},
    "macd_trend": {"fast": (8, 15, True), "slow": (20, 35, True), "signal": (5, 12, True)},
    "donchian_breakout": {"period": (10, 100, True)},
    "rsi_reversion": {
        "period": (5, 30, True),
        "buy_below": (15, 40, True),
        "sell_above": (60, 85, True),
    },
    "bollinger_fade": {"period": (10, 40, True), "k": (1.5, 3.0, False)},
    "momentum_burst": {"period": (5, 40, True), "threshold": (0.002, 0.03, False)},
}

# Management/risk bounds shared by all tribes.
MANAGEMENT_SPACE = {
    "sl_atr": (0.5, 5.0, False),
    "tp_atr": (0.5, 8.0, False),
    "trail_atr": (0.5, 4.0, False),  # optional gene
    "max_bars": (10, 200, True),  # optional gene
    "breakeven_atr": (0.3, 2.0, False),  # optional gene: move SL to entry after this run-up
}
RISK_SPACE = {"risk_pct": (0.003, 0.02, False)}
TREND_MA_SPACE = (50, 200, True)
