import numpy as np
import pandas as pd
import pytest

from alglory.analysis.sizing import SymbolSpec
from alglory.backtest.engine import Costs
from alglory.data.cache import BarCache
from alglory.evolve import campaign
from alglory.evolve.campaign import CampaignConfig, _load_bars
from alglory.evolve.costs import CostProfile, commission_price, cost_profile, resolve_profile

TINY = {"symbols": ["EURUSD"], "tribes": ["trend"]}


def _bars(n):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0,
         "spread": 2e-5}, index=idx,
    )


def test_load_bars_halves_request_on_terminal_failure(monkeypatch, tmp_path):
    requested = []

    class FakeSource:
        def connect(self):
            return type("S", (), {"ok": True, "account": "demo", "message": ""})()

        def fetch_bars(self, symbol, timeframe, count):
            requested.append(count)
            if count > 12_500:
                raise RuntimeError("Terminal: Call failed")
            return _bars(count)

        def symbol_spec(self, symbol):
            return SymbolSpec(tick_value=1.0, tick_size=1e-5)

        def symbol_spread(self, symbol):
            return 0.0

    monkeypatch.setattr("alglory.data.mt5source.MT5Source", FakeSource)
    cfg = CampaignConfig(**TINY, source="mt5")
    bars, costs = _load_bars(cfg, "EURUSD", "H1", BarCache(tmp_path), lambda e: None)
    assert requested == [50_000, 25_000, 12_500]
    assert len(bars) == 12_500
    assert costs.spread == pytest.approx(1e-4)
    assert costs.commission == pytest.approx(7e-5)


def test_profiles_by_market():
    assert cost_profile("EURUSD") == CostProfile(0.0001, 7.0)
    assert cost_profile("USDJPY") == CostProfile(0.01, 7.0)
    assert cost_profile("XAUUSD") == CostProfile(0.25, 7.0)
    assert cost_profile("NAS100") == CostProfile(1.5, 0.0)
    assert cost_profile("USTEC") == CostProfile(1.5, 0.0)


def test_commission_to_price_units():
    spec = SymbolSpec(tick_value=1.0, tick_size=1e-5)
    assert commission_price(7.0, spec) == pytest.approx(7e-5)


def test_costs_round_trip_adds_price_units():
    assert Costs(spread=1e-4, commission=7e-5).round_trip == pytest.approx(1.7e-4)


def test_cost_overrides_win():
    cfg = CampaignConfig(**TINY, cost_overrides={"XAUUSD": {"min_spread": 0.4, "commission_per_lot": 0.0}})
    assert resolve_profile(cfg, "XAUUSD") == CostProfile(0.4, 0.0)
    assert resolve_profile(cfg, "EURUSD") == CostProfile(0.0001, 7.0)


def test_old_config_with_commission_fields_parses():
    cfg = CampaignConfig(**TINY, commission_per_lot=0.0001, min_spread_pips=2.0)
    assert not hasattr(cfg, "commission_per_lot")
