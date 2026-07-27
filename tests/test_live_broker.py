"""Stage 1 of the live executor: MT5Source trading methods + FakeMT5Source.

The real MT5Source wraps the MetaTrader5 module, which isn't available/live in
CI, so its trading methods are exercised by injecting a tiny in-memory *module*
double (``_FakeModule``) — enough to prove request construction and retcode
interpretation. FakeMT5Source is the higher-level, in-memory broker used by the
executor (stage 2) and is tested directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alglory.analysis.sizing import lots_for_risk
from alglory.data.fake_mt5 import FakeMT5Source
from alglory.data.mt5source import MT5Source

DONE = 10009  # TRADE_RETCODE_DONE
REJECT = 10004


class _FakeModule:
    """Minimal stand-in for the MetaTrader5 module (records the last request)."""

    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 2
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = DONE
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ACCOUNT_TRADE_MODE_DEMO = 0

    def __init__(self, *, retcode=DONE, positions=(), tick=(1.10123, 1.10100)):
        self.retcode = retcode
        self._positions = list(positions)
        self._ask, self._bid = tick
        self.last_request = None

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(ask=self._ask, bid=self._bid)

    def order_send(self, request):
        self.last_request = request
        return SimpleNamespace(retcode=self.retcode, order=777)

    def positions_get(self):
        return self._positions

    def account_info(self):
        return SimpleNamespace(
            login=42, balance=1000.0, equity=1010.0, currency="USD", trade_mode=0
        )

    def symbol_info(self, symbol):
        return SimpleNamespace(
            trade_tick_value=1.0, trade_tick_size=0.00001,
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
        )

    def last_error(self):
        return (-1, "no terminal")


def _connected(module) -> MT5Source:
    s = MT5Source()
    s._mt5 = module
    s._connected = True
    return s


# ---- real MT5Source trading methods (injected module) -----------------


def test_place_market_buy_builds_deal_at_ask():
    mod = _FakeModule()
    res = _connected(mod).place_market("EURUSD", 1, 0.05, sl=1.09, tp=1.12, magic=990117)
    r = mod.last_request
    assert r["action"] == _FakeModule.TRADE_ACTION_DEAL
    assert r["type"] == _FakeModule.ORDER_TYPE_BUY
    assert r["price"] == 1.10123  # ask for a buy
    assert r["volume"] == 0.05 and r["magic"] == 990117
    assert r["sl"] == 1.09 and r["tp"] == 1.12
    assert res.ok and res.retcode == DONE


def test_place_market_sell_uses_bid():
    mod = _FakeModule()
    _connected(mod).place_market("EURUSD", -1, 0.02, magic=1)
    assert mod.last_request["type"] == _FakeModule.ORDER_TYPE_SELL
    assert mod.last_request["price"] == 1.10100  # bid for a sell


def test_place_market_rejected_retcode_is_not_ok():
    res = _connected(_FakeModule(retcode=REJECT)).place_market("EURUSD", 1, 0.01, magic=1)
    assert res.ok is False and res.retcode == REJECT


def test_positions_maps_direction_and_filters_magic():
    positions = [
        SimpleNamespace(ticket=1, symbol="EURUSD", type=0, volume=0.1, price_open=1.1,
                        sl=1.0, tp=1.2, magic=7, profit=3.0),
        SimpleNamespace(ticket=2, symbol="EURUSD", type=1, volume=0.2, price_open=1.1,
                        sl=1.2, tp=1.0, magic=9, profit=-1.0),
    ]
    src = _connected(_FakeModule(positions=positions))
    all_pos = src.positions()
    assert [p.direction for p in all_pos] == [1, -1]
    assert [p.magic for p in src.positions(magic=9)] == [9]


def test_account_maps_demo_flag_and_symbol_spec():
    src = _connected(_FakeModule())
    acct = src.account()
    assert acct.is_demo is True and acct.currency == "USD" and acct.equity == 1010.0
    spec = src.symbol_spec("EURUSD")
    assert spec.min_lot == 0.01 and spec.tick_value == 1.0


def test_close_position_sends_opposite_side():
    pos = [SimpleNamespace(ticket=5, symbol="EURUSD", type=0, volume=0.1, price_open=1.1,
                           sl=1.0, tp=1.2, magic=7, profit=0.0)]
    mod = _FakeModule(positions=pos)
    assert _connected(mod).close_position("EURUSD", 7) is True
    # closing a long sends a SELL for the position ticket
    assert mod.last_request["type"] == _FakeModule.ORDER_TYPE_SELL
    assert mod.last_request["position"] == 5


def test_trading_methods_require_connection():
    with pytest.raises(Exception):
        MT5Source().account()


# ---- FakeMT5Source (in-memory broker) ---------------------------------


def test_fake_place_creates_position_with_magic():
    fake = FakeMT5Source(balance=5000, price=1.10)
    res = fake.place_market("EURUSD", 1, 0.03, sl=1.09, tp=1.12, magic=42)
    assert res.ok
    pos = fake.positions(magic=42)
    assert len(pos) == 1
    assert pos[0].direction == 1 and pos[0].volume == 0.03 and pos[0].price_open == 1.10
    assert fake.positions(magic=999) == []


def test_fake_close_and_modify():
    fake = FakeMT5Source()
    fake.place_market("EURUSD", -1, 0.01, magic=7)
    assert fake.modify_position("EURUSD", 7, sl=1.2, tp=1.0) is True
    assert fake.positions(magic=7)[0].sl == 1.2
    assert fake.close_position("EURUSD", 7) is True
    assert fake.positions(magic=7) == []
    assert fake.close_position("EURUSD", 7) is False  # already gone


def test_fake_zero_volume_places_nothing():
    fake = FakeMT5Source()
    res = fake.place_market("EURUSD", 1, 0.0, magic=1)
    assert res.ok is False
    assert fake.positions() == []


def test_fake_account_and_spec_feed_sizing():
    fake = FakeMT5Source(balance=1000, currency="USD", is_demo=True)
    acct = fake.account()
    assert acct.balance == 1000 and acct.is_demo is True
    spec = fake.symbol_spec("EURUSD")
    # the spec must be directly usable by the sizing rule
    lots = lots_for_risk(1000, 0.01, 0.0052, spec)
    assert lots > 0


def test_fake_records_order_calls():
    fake = FakeMT5Source()
    fake.place_market("EURUSD", 1, 0.02, magic=1)
    fake.close_position("EURUSD", 1)
    assert [o[0] for o in fake.orders] == ["open", "close"]
