import numpy as np
import pytest

from alglory.deploy.mql5 import GUARDRAIL_PRESETS, Guardrails, generate_ea, write_ea
from alglory.genome import (
    TRIBES,
    FilterGene,
    Genome,
    ManagementGene,
    RiskGene,
    SignalGene,
    random_genome,
)

MGMT = ManagementGene(sl_atr=1.5, tp_atr=3.0, trail_atr=None, max_bars=None)
FILTERS = FilterGene(trend_ma=None, session=None, atr_regime=None)
GUARD = Guardrails(risk_pct=0.01, daily_loss_pct=0.05, max_dd_pct=0.10)


def _gen(genome):
    return generate_ea(genome, name="TestEA", symbol="EURUSD", timeframe="H1", guardrails=GUARD)


def test_ma_cross_ea_contains_exact_params():
    g = Genome(
        "trend", "both",
        SignalGene("ma_cross", {"fast": 12, "slow": 48}),
        FILTERS, MGMT, RiskGene(0.01),
    )
    code = _gen(g)
    assert "iMA" in code
    assert "12" in code and "48" in code
    assert "OnInit" in code and "OnTick" in code
    assert "InpSLAtr = 1.5" in code
    assert "InpTPAtr = 3.0" in code
    assert "InpDailyLossPct = 0.05" in code
    assert "InpMaxDDPct = 0.1" in code


def test_rsi_ea_contains_thresholds():
    g = Genome(
        "mean_reversion", "long",
        SignalGene("rsi_reversion", {"period": 14, "buy_below": 28, "sell_above": 72}),
        FILTERS, MGMT, RiskGene(0.01),
    )
    code = _gen(g)
    assert "iRSI" in code
    assert "28" in code and "72" in code
    # long-only genome must not open shorts
    assert "InpAllowShort = false" in code
    assert "InpAllowLong = true" in code


@pytest.mark.parametrize("tribe", list(TRIBES))
def test_every_tribe_generates_complete_ea(tribe):
    rng = np.random.default_rng(11)
    for _ in range(5):
        g = random_genome(tribe, rng)
        code = _gen(g)
        assert "OnInit" in code and "OnTick" in code and "OnDeinit" in code
        assert "{" + "}" not in code
        assert "{placeholder}" not in code
        assert "None" not in code  # python leak guard
        assert code.count("{") == code.count("}")


def test_session_filter_compiled_in():
    g = Genome(
        "momentum", "both",
        SignalGene("momentum_burst", {"period": 10, "threshold": 0.01}),
        FilterGene(trend_ma=None, session=(8, 17), atr_regime=None),
        MGMT, RiskGene(0.01),
    )
    code = _gen(g)
    assert "InpSessionStart = 8" in code
    assert "InpSessionEnd = 17" in code


def test_guardrail_presets_exist():
    assert set(GUARDRAIL_PRESETS) == {"personal", "prop_conservative", "prop_aggressive"}
    prop = GUARDRAIL_PRESETS["prop_conservative"]
    assert prop.daily_loss_pct < GUARDRAIL_PRESETS["personal"].daily_loss_pct


def test_write_ea(tmp_path):
    g = Genome(
        "trend", "both",
        SignalGene("ma_cross", {"fast": 10, "slow": 40}),
        FILTERS, MGMT, RiskGene(0.01),
    )
    path = write_ea(_gen(g), "MyBot", tmp_path)
    assert path.name == "MyBot.mq5"
    assert "OnTick" in path.read_text(encoding="utf-8")
