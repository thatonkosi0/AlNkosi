"""MQL5 Expert Advisor code generation.

Each genome compiles to a complete, self-contained .mq5 source file: the
signal logic is generated per signal kind, management and guardrails are
baked in as input variables, and sizing is risk-based off the account
balance. The user compiles it once in MetaEditor (F7) and attaches it to
the matching chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alglory.genome import Genome


@dataclass(frozen=True)
class Guardrails:
    risk_pct: float
    daily_loss_pct: float
    max_dd_pct: float


GUARDRAIL_PRESETS: dict[str, Guardrails] = {
    "personal": Guardrails(risk_pct=0.01, daily_loss_pct=0.08, max_dd_pct=0.25),
    "prop_conservative": Guardrails(risk_pct=0.005, daily_loss_pct=0.03, max_dd_pct=0.08),
    "prop_aggressive": Guardrails(risk_pct=0.01, daily_loss_pct=0.04, max_dd_pct=0.09),
}


def _flt(x: float) -> str:
    return str(float(x))


def _signal_globals(kind: str, p: dict) -> str:
    if kind == "ma_cross":
        return "int hFast; int hSlow;"
    if kind == "macd_trend":
        return "int hMacd;"
    if kind == "donchian_breakout":
        return ""
    if kind == "rsi_reversion":
        return "int hRsi;"
    if kind == "bollinger_fade":
        return "int hBands;"
    if kind == "momentum_burst":
        return ""
    raise ValueError(f"Unknown signal kind {kind!r}")


def _signal_init(kind: str, p: dict) -> str:
    if kind == "ma_cross":
        return (
            f"   hFast = iMA(_Symbol, _Period, {int(p['fast'])}, 0, MODE_SMA, PRICE_CLOSE);\n"
            f"   hSlow = iMA(_Symbol, _Period, {int(p['slow'])}, 0, MODE_SMA, PRICE_CLOSE);\n"
            "   if(hFast == INVALID_HANDLE || hSlow == INVALID_HANDLE) return INIT_FAILED;"
        )
    if kind == "macd_trend":
        return (
            f"   hMacd = iMACD(_Symbol, _Period, {int(p['fast'])}, {int(p['slow'])}, "
            f"{int(p['signal'])}, PRICE_CLOSE);\n"
            "   if(hMacd == INVALID_HANDLE) return INIT_FAILED;"
        )
    if kind == "rsi_reversion":
        return (
            f"   hRsi = iRSI(_Symbol, _Period, {int(p['period'])}, PRICE_CLOSE);\n"
            "   if(hRsi == INVALID_HANDLE) return INIT_FAILED;"
        )
    if kind == "bollinger_fade":
        return (
            f"   hBands = iBands(_Symbol, _Period, {int(p['period'])}, 0, "
            f"{_flt(p['k'])}, PRICE_CLOSE);\n"
            "   if(hBands == INVALID_HANDLE) return INIT_FAILED;"
        )
    return "   // no indicator handles required for this signal"


def _signal_body(kind: str, p: dict) -> str:
    if kind == "ma_cross":
        return """   double fast[], slow[];
   // CopyBuffer fills oldest-first; flip to series so [1] is the last
   // closed bar and [2] the bar before it (matching the backtester).
   ArraySetAsSeries(fast, true);
   ArraySetAsSeries(slow, true);
   if(CopyBuffer(hFast, 0, 0, 3, fast) < 3) return 0;
   if(CopyBuffer(hSlow, 0, 0, 3, slow) < 3) return 0;
   if(fast[1] > slow[1] && fast[2] <= slow[2]) return 1;
   if(fast[1] < slow[1] && fast[2] >= slow[2]) return -1;
   return 0;"""
    if kind == "macd_trend":
        return """   double main[], sig[];
   ArraySetAsSeries(main, true);
   ArraySetAsSeries(sig, true);
   if(CopyBuffer(hMacd, 0, 0, 3, main) < 3) return 0;
   if(CopyBuffer(hMacd, 1, 0, 3, sig) < 3) return 0;
   if(main[1] > sig[1] && main[2] <= sig[2]) return 1;
   if(main[1] < sig[1] && main[2] >= sig[2]) return -1;
   return 0;"""
    if kind == "donchian_breakout":
        period = int(p["period"])
        return f"""   int hi = iHighest(_Symbol, _Period, MODE_HIGH, {period}, 2);
   int lo = iLowest(_Symbol, _Period, MODE_LOW, {period}, 2);
   if(hi < 0 || lo < 0) return 0;
   double upper = iHigh(_Symbol, _Period, hi);
   double lower = iLow(_Symbol, _Period, lo);
   double c1 = iClose(_Symbol, _Period, 1);
   if(c1 > upper) return 1;
   if(c1 < lower) return -1;
   return 0;"""
    if kind == "rsi_reversion":
        return f"""   double r[];
   ArraySetAsSeries(r, true);
   if(CopyBuffer(hRsi, 0, 0, 3, r) < 3) return 0;
   if(r[1] < {int(p['buy_below'])} && r[2] >= {int(p['buy_below'])}) return 1;
   if(r[1] > {int(p['sell_above'])} && r[2] <= {int(p['sell_above'])}) return -1;
   return 0;"""
    if kind == "bollinger_fade":
        return """   double upper[], lower[];
   ArraySetAsSeries(upper, true);
   ArraySetAsSeries(lower, true);
   if(CopyBuffer(hBands, 1, 0, 2, upper) < 2) return 0;
   if(CopyBuffer(hBands, 2, 0, 2, lower) < 2) return 0;
   double c1 = iClose(_Symbol, _Period, 1);
   if(c1 < lower[1]) return 1;
   if(c1 > upper[1]) return -1;
   return 0;"""
    if kind == "momentum_burst":
        period = int(p["period"])
        return f"""   double now = iClose(_Symbol, _Period, 1);
   double before = iClose(_Symbol, _Period, {period + 1});
   if(before <= 0) return 0;
   double mom = now / before - 1.0;
   if(mom > {_flt(p['threshold'])}) return 1;
   if(mom < -{_flt(p['threshold'])}) return -1;
   return 0;"""
    raise ValueError(f"Unknown signal kind {kind!r}")


def generate_ea(
    genome: Genome,
    *,
    name: str,
    symbol: str,
    timeframe: str,
    guardrails: Guardrails,
) -> str:
    g = genome
    p = g.signal.params
    use_session = g.filters.session is not None
    session_start, session_end = g.filters.session if use_session else (0, 24)
    use_trend = g.filters.trend_ma is not None
    trend_period = int(g.filters.trend_ma) if use_trend else 100
    use_trail = g.management.trail_atr is not None
    trail_atr = g.management.trail_atr if use_trail else 0.0
    use_breakeven = g.management.breakeven_atr is not None
    breakeven_atr = g.management.breakeven_atr if use_breakeven else 0.0
    max_bars = int(g.management.max_bars) if g.management.max_bars is not None else 0
    allow_long = "true" if g.direction in ("long", "both") else "false"
    allow_short = "true" if g.direction in ("short", "both") else "false"
    atr_regime = {None: 0, "high": 1, "low": -1}[g.filters.atr_regime]

    trend_globals = "int hTrend;" if use_trend else ""
    trend_init = (
        f"   hTrend = iMA(_Symbol, _Period, {trend_period}, 0, MODE_SMA, PRICE_CLOSE);\n"
        "   if(hTrend == INVALID_HANDLE) return INIT_FAILED;"
        if use_trend
        else "   // trend filter disabled"
    )
    trend_check = (
        """   double tr[];
   ArraySetAsSeries(tr, true);
   if(CopyBuffer(hTrend, 0, 0, 3, tr) < 3) return 0;
   if(dir == 1 && tr[1] <= tr[2]) return 0;
   if(dir == -1 && tr[1] >= tr[2]) return 0;"""
        if use_trend
        else "   // trend filter disabled"
    )

    return f"""//+------------------------------------------------------------------+
//| {name}.mq5                                                        |
//| Generated by ALGLORY - genetic bot factory                        |
//| Bred for {symbol} {timeframe} | tribe: {g.tribe} | signal: {g.signal.kind}
//| RISK WARNING: algorithmic trading software, not financial advice. |
//| Backtested performance does not guarantee future results.         |
//+------------------------------------------------------------------+
#property copyright "ALGLORY"
#property version   "1.00"
#include <Trade/Trade.mqh>

input long   InpMagic = 990117;
input double InpRiskPct = {_flt(guardrails.risk_pct)};      // risk per trade (fraction of balance)
input double InpDailyLossPct = {_flt(guardrails.daily_loss_pct)}; // daily loss halt (fraction)
input double InpMaxDDPct = {_flt(guardrails.max_dd_pct)};    // max drawdown permanent halt (fraction)
input bool   InpMinLotFallback = true;  // if risk-size rounds below the broker min lot, trade the min lot so the strategy still fires (bounded by InpMaxRiskPct)
input double InpMaxRiskPct = 0.05;      // per-trade risk ceiling for the min-lot fallback; below this the account is refused loudly, not blown up
input double InpSLAtr = {_flt(g.management.sl_atr)};       // stop loss in ATR multiples
input double InpTPAtr = {_flt(g.management.tp_atr)};       // take profit in ATR multiples
input bool   InpUseTrail = {"true" if use_trail else "false"};
input double InpTrailAtr = {_flt(trail_atr)};   // trail distance in CURRENT ATR multiples (adaptive)
input bool   InpUseBreakeven = {"true" if use_breakeven else "false"};
input double InpBreakevenAtr = {_flt(breakeven_atr)}; // move SL to entry after this run-up (entry-ATR multiples)
input int    InpMaxBars = {max_bars};        // 0 disables the time exit
input bool   InpUseSession = {"true" if use_session else "false"};
input int    InpSessionStart = {int(session_start)};
input int    InpSessionEnd = {int(session_end)};
input int    InpAtrRegime = {atr_regime};     // 1 trade high-vol only, -1 low-vol only, 0 off
input bool   InpAllowLong = {allow_long};
input bool   InpAllowShort = {allow_short};

CTrade trade;
int hAtr;
{_signal_globals(g.signal.kind, p)}
{trend_globals}
double gPeakEquity = 0.0;
double gDayStartEquity = 0.0;
int gDayOfYear = -1;
bool gHalted = false;
datetime gLastBar = 0;
datetime gEntryBar = 0;
double gEntryAtr = 0.0;
bool gBeArmed = false;
bool gMinLotWarned = false;

int OnInit()
{{
   trade.SetExpertMagicNumber(InpMagic);
   hAtr = iATR(_Symbol, _Period, 14);
   if(hAtr == INVALID_HANDLE) return INIT_FAILED;
{_signal_init(g.signal.kind, p)}
{trend_init}
   if(_Symbol != "{symbol}")
      Print("ALGLORY warning: bred for {symbol} but attached to ", _Symbol,
            " — this genome was not validated on this symbol.");
   if(_Period != PERIOD_{timeframe})
      Print("ALGLORY warning: bred for {timeframe} but attached to a different timeframe chart.");
   gPeakEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   gDayStartEquity = gPeakEquity;
   return INIT_SUCCEEDED;
}}

double Norm(double price)
{{
   return NormalizeDouble(price, _Digits);
}}

double AtrMedian()
{{
   // median ATR of the last 100 completed bars (mirrors the backtester's
   // rolling-median volatility regime, min 20 observations)
   double buf[];
   int got = CopyBuffer(hAtr, 0, 1, 100, buf);
   if(got < 20) return 0.0;
   ArraySort(buf);
   if(got % 2 == 1) return buf[got / 2];
   return 0.5 * (buf[got / 2 - 1] + buf[got / 2]);
}}

void OnDeinit(const int reason)
{{
   IndicatorRelease(hAtr);
}}

int RawSignal()
{{
{_signal_body(g.signal.kind, p)}
}}

int FilteredSignal()
{{
   int dir = RawSignal();
   if(dir == 0) return 0;
   if(dir == 1 && !InpAllowLong) return 0;
   if(dir == -1 && !InpAllowShort) return 0;
   if(InpUseSession)
   {{
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      if(dt.hour < InpSessionStart || dt.hour >= InpSessionEnd) return 0;
   }}
{trend_check}
   if(InpAtrRegime != 0)
   {{
      double med = AtrMedian();
      if(med <= 0.0) return 0;
      bool volHigh = CurrentAtr() > med;
      if(InpAtrRegime == 1 && !volHigh) return 0;
      if(InpAtrRegime == -1 && volHigh) return 0;
   }}
   return dir;
}}

double CurrentAtr()
{{
   double a[];
   ArraySetAsSeries(a, true); // a[1] = ATR of the last completed bar
   if(CopyBuffer(hAtr, 0, 0, 2, a) < 2) return 0.0;
   return a[1];
}}

bool GuardrailsAllowTrading()
{{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity > gPeakEquity) gPeakEquity = equity;
   if(gHalted) return false;
   if(equity < gPeakEquity * (1.0 - InpMaxDDPct))
   {{
      gHalted = true;
      Print("ALGLORY guardrail: max drawdown breached, EA halted permanently.");
      return false;
   }}
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(dt.day_of_year != gDayOfYear)
   {{
      gDayOfYear = dt.day_of_year;
      gDayStartEquity = equity;
   }}
   if(equity < gDayStartEquity * (1.0 - InpDailyLossPct))
   {{
      return false; // done for the day
   }}
   return true;
}}

double LotsForRisk(double slDistance)
{{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickValue <= 0 || tickSize <= 0 || slDistance <= 0) return 0.0;
   double lossPerLot = slDistance / tickSize * tickValue;
   double lots = balance * InpRiskPct / lossPerLot;
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   lots = MathFloor(lots / step) * step;
   if(lots < minLot)
   {{
      // Risk-based size is below the broker minimum. Rather than silently never
      // trading on a small account, optionally fall back to the minimum lot so
      // the strategy still fires — but only while that minimum-lot trade stays
      // within InpMaxRiskPct, so a too-small account is refused loudly, not
      // quietly blown up.
      if(InpMinLotFallback)
      {{
         double minLotRisk = minLot * lossPerLot / balance;
         if(minLotRisk <= InpMaxRiskPct)
         {{
            if(!gMinLotWarned)
            {{
               PrintFormat("ALGLORY: risk-based size below broker min; trading min lot %.2f (%.1f%% risk this trade).",
                           minLot, minLotRisk * 100.0);
               gMinLotWarned = true;
            }}
            return minLot;
         }}
         PrintFormat("ALGLORY: account too small — even min lot %.2f risks %.1f%% (cap %.1f%%). Fund to about %.0f %s to trade this strategy.",
                     minLot, minLotRisk * 100.0, InpMaxRiskPct * 100.0,
                     minLot * lossPerLot / InpMaxRiskPct, AccountInfoString(ACCOUNT_CURRENCY));
         return 0.0;
      }}
      Print("ALGLORY: account too small for the chosen risk (enable InpMinLotFallback to trade the minimum lot).");
      return 0.0;
   }}
   return MathMin(lots, maxLot);
}}

void ManagePosition()
{{
   if(!PositionSelect(_Symbol)) return;
   if(PositionGetInteger(POSITION_MAGIC) != InpMagic) return;
   long type = PositionGetInteger(POSITION_TYPE);
   double atrNow = CurrentAtr();
   if(InpMaxBars > 0 && gEntryBar > 0)
   {{
      int held = Bars(_Symbol, _Period, gEntryBar, TimeCurrent()) - 1;
      if(held >= InpMaxBars)
      {{
         trade.PositionClose(_Symbol);
         return;
      }}
   }}
   double sl = PositionGetDouble(POSITION_SL);
   double tp = PositionGetDouble(POSITION_TP);
   double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(InpUseBreakeven && !gBeArmed && gEntryAtr > 0)
   {{
      double run = InpBreakevenAtr * gEntryAtr;
      bool reached = (type == POSITION_TYPE_BUY) ? (bid >= openPrice + run)
                                                 : (ask <= openPrice - run);
      if(reached)
      {{
         gBeArmed = true;
         bool improves = (type == POSITION_TYPE_BUY) ? (sl < openPrice)
                                                     : (sl == 0 || sl > openPrice);
         if(improves && trade.PositionModify(_Symbol, Norm(openPrice), tp))
            sl = Norm(openPrice);
      }}
   }}
   if(InpUseTrail && atrNow > 0)
   {{
      // trail distance follows the CURRENT ATR (adaptive to volatility);
      // the TP shifts outward in lockstep so winners keep running
      if(type == POSITION_TYPE_BUY)
      {{
         double candidate = bid - InpTrailAtr * atrNow;
         if(candidate > sl)
         {{
            double newTp = (tp > 0 && sl > 0) ? tp + (candidate - sl) : tp;
            trade.PositionModify(_Symbol, Norm(candidate), Norm(newTp));
         }}
      }}
      else
      {{
         double candidate = ask + InpTrailAtr * atrNow;
         if(sl == 0 || candidate < sl)
         {{
            double newTp = (tp > 0 && sl > 0) ? tp - (sl - candidate) : tp;
            trade.PositionModify(_Symbol, Norm(candidate), Norm(newTp));
         }}
      }}
   }}
}}

void OnTick()
{{
   datetime barTime = iTime(_Symbol, _Period, 0);
   bool newBar = (barTime != gLastBar);
   if(newBar) gLastBar = barTime;

   ManagePosition();

   if(!newBar) return;
   if(!GuardrailsAllowTrading()) return;
   if(PositionSelect(_Symbol)) return; // one position at a time

   int dir = FilteredSignal();
   if(dir == 0) return;

   double atrNow = CurrentAtr();
   if(atrNow <= 0) return;
   double slDist = InpSLAtr * atrNow;
   double tpDist = InpTPAtr * atrNow;
   double lots = LotsForRisk(slDist);
   if(lots <= 0) return;

   if(dir == 1)
   {{
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      if(trade.Buy(lots, _Symbol, 0.0, Norm(ask - slDist), Norm(ask + tpDist), "ALGLORY"))
      {{
         gEntryBar = barTime;
         gEntryAtr = atrNow;
         gBeArmed = false;
      }}
      else
         Print("ALGLORY: buy rejected — ", trade.ResultRetcode(), " ",
               trade.ResultRetcodeDescription());
   }}
   else
   {{
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      if(trade.Sell(lots, _Symbol, 0.0, Norm(bid + slDist), Norm(bid - tpDist), "ALGLORY"))
      {{
         gEntryBar = barTime;
         gEntryAtr = atrNow;
         gBeArmed = false;
      }}
      else
         Print("ALGLORY: sell rejected — ", trade.ResultRetcode(), " ",
               trade.ResultRetcodeDescription());
   }}
}}
"""


def write_ea(code: str, name: str, experts_dir: Path) -> Path:
    experts_dir = Path(experts_dir)
    experts_dir.mkdir(parents=True, exist_ok=True)
    path = experts_dir / f"{name}.mq5"
    path.write_text(code, encoding="utf-8")
    return path
