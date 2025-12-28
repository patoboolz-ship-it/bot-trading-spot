#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BOT SPOT Binance - RSI + MACD + Consecutivas (con HA opcional) + GUI + Excel
- Solo LONG: compra y cierra (venta para cerrar)
- Señales SOLO con vela cerrada (evita repaint intrabar)
- Compra con 98% USDT libre, aunque tengas "polvo" SOL
- Verifica compra: gasto >= 90% de lo objetivo; si no, reintenta
- Exporta Excel: operaciones abiertas y cerradas

Requisitos:
  pip install python-binance pandas openpyxl

Seguridad:
- Parte en DRY_RUN=True
- Usa API con permisos SOLO SPOT (sin retiros)
"""

import os
import time
import math
import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# =========================
# CONFIG
# =========================

API_KEY_PATH    = r"C:\Users\EQCONF\Documents\programas python\claves\bot trading\Clave API.txt"
API_SECRET_PATH = r"C:\Users\EQCONF\Documents\programas python\claves\bot trading\Clave secreta.txt"

SYMBOL = "SOLUSDT"
BASE_ASSET = "SOL"
QUOTE_ASSET = "USDT"

# Timeframe de señales (TradingView)
# Binance: '1m','3m','5m','15m','30m','1h','2h','4h','1d'...
INTERVAL = "1h"

# Backtest/Señales: cuántas velas cargar
LOOKBACK = 800  # para indicadores, 800 suele sobrar (MACD lento 83)

# Ordenes
USE_98_PERCENT = True
ORDER_PCT = 0.98  # 98% del USDT libre
MIN_FILL_RATIO = 0.90  # verificación: gastado >= 90% de objetivo
MAX_RETRY_BUY = 2       # reintentos si quedó muy parcial
RETRY_SLEEP_SEC = 2

# Ejecutar real o simulación
DRY_RUN = True

# Logs / journal
JOURNAL_CSV = "bot_journal.csv"


# =========================
# GEN 235 (defaults)
# =========================
DEFAULT_GEN = {
  "use_ha": 1,
  "rsi_period": 10,
  "rsi_oversold": 40.0,
  "rsi_overbought": 58.0,
  "macd_fast": 40,
  "macd_slow": 83,
  "macd_signal": 46,
  "consec_red": 3,
  "consec_green": 4,
  "w_buy_rsi": 0.51,
  "w_buy_macd": 0.49,
  "w_buy_consec": 0.00,
  "buy_th": 0.74,
  "w_sell_rsi": 0.21,
  "w_sell_macd": 0.46,
  "w_sell_consec": 0.33,
  "sell_th": 0.62,
  "take_profit": 0.109,
  "stop_loss": 0.012,
  "cooldown": 1,
  "edge_trigger": 0
}


# =========================
# Helpers: read keys
# =========================

def read_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# =========================
# Dataclasses
# =========================

@dataclass
class Trade:
    trade_id: str
    symbol: str
    side: str             # BUY / SELL
    qty: float
    price: float          # avg price
    quote_spent: float    # USDT
    time_utc: str
    reason: str           # SIGNAL / TP / SL / CLOSE
    order_id: str = ""
    raw: str = ""


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time_utc: str
    tp_price: float
    sl_price: float


# =========================
# Indicators
# =========================

def ema(series, length: int):
    """Exponential moving average."""
    alpha = 2.0 / (length + 1.0)
    out = []
    prev = series[0]
    out.append(prev)
    for x in series[1:]:
        prev = alpha * x + (1 - alpha) * prev
        out.append(prev)
    return out

def rsi(close, period: int):
    """Classic RSI."""
    if period < 2:
        period = 2
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(close)):
        ch = close[i] - close[i-1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    # Wilder smoothing
    avg_gain = sum(gains[1:period+1]) / period
    avg_loss = sum(losses[1:period+1]) / period
    rsis = [50.0] * len(close)
    if avg_loss == 0:
        rs = float('inf')
    else:
        rs = avg_gain / avg_loss
    rsis[period] = 100 - (100 / (1 + rs if rs != float('inf') else 1e9))
    for i in range(period+1, len(close)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsis[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsis[i] = 100 - (100 / (1 + rs))
    return rsis

def heikin_ashi(ohlc):
    """Compute Heikin Ashi candles from regular ohlc arrays."""
    # ohlc: list of dicts with open/high/low/close
    ha = []
    ha_open = None
    ha_close_prev = None
    for i, c in enumerate(ohlc):
        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
        ha_close = (o + h + l + cl) / 4.0
        if i == 0:
            ha_open = (o + cl) / 2.0
        else:
            ha_open = (ha_open + ha_close_prev) / 2.0
        ha_high = max(h, ha_open, ha_close)
        ha_low  = min(l, ha_open, ha_close)
        ha.append({"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close})
        ha_close_prev = ha_close
    return ha

def macd_hist(close, fast: int, slow: int, signal: int):
    fast = max(2, int(fast))
    slow = max(fast + 1, int(slow))
    signal = max(2, int(signal))

    efast = ema(close, fast)
    eslow = ema(close, slow)
    macd_line = [a - b for a, b in zip(efast, eslow)]
    macd_sig = ema(macd_line, signal)
    hist = [a - b for a, b in zip(macd_line, macd_sig)]
    return hist


# =========================
# Strategy scoring
# =========================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def consec_up(close, n: int) -> float:
    n = int(n)
    if n <= 0:
        return 0.0
    if len(close) < n + 1:
        return 0.0
    ok = True
    # último n: close[-1] >= close[-2] >= ... >= close[-(n+1)]
    for i in range(n):
        if not (close[-1 - i] >= close[-2 - i]):
            ok = False
            break
    return 1.0 if ok else 0.0

def consec_down(close, n: int) -> float:
    n = int(n)
    if n <= 0:
        return 0.0
    if len(close) < n + 1:
        return 0.0
    ok = True
    for i in range(n):
        if not (close[-1 - i] <= close[-2 - i]):
            ok = False
            break
    return 1.0 if ok else 0.0

def buy_rsi_signal(rsi_val: float, os: float, ob: float) -> float:
    # r <= os => 1 ; r >= ob => 0 ; linear
    os2 = min(os, ob - 1e-6)
    ob2 = max(ob, os2 + 1e-6)
    r = rsi_val
    if r <= os2:
        return 1.0
    if r >= ob2:
        return 0.0
    return clamp01((ob2 - r) / (ob2 - os2))

def sell_rsi_signal(rsi_val: float, os: float, ob: float) -> float:
    # r >= ob => 1 ; r <= os => 0 ; linear
    os2 = min(os, ob - 1e-6)
    ob2 = max(ob, os2 + 1e-6)
    r = rsi_val
    if r >= ob2:
        return 1.0
    if r <= os2:
        return 0.0
    return clamp01((r - os2) / (ob2 - os2))

def buy_macd_signal(hist_now: float, hist_prev: float, edge_trigger: int) -> float:
    if edge_trigger:
        return 1.0 if (hist_prev <= 0 and hist_now > 0) else 0.0
    return 1.0 if hist_now > 0 else 0.0

def sell_macd_signal(hist_now: float, hist_prev: float, edge_trigger: int) -> float:
    if edge_trigger:
        return 1.0 if (hist_prev >= 0 and hist_now < 0) else 0.0
    return 1.0 if hist_now < 0 else 0.0

def normalize3(a,b,c):
    s = a+b+c
    if s <= 1e-12:
        return (1/3,1/3,1/3)
    return (a/s, b/s, c/s)

def compute_scores(params: dict, candles: list):
    """
    candles: list of dict open/high/low/close
    returns (buyScore, sellScore, last_close)
    """
    use_ha = int(params["use_ha"]) == 1
    edge = int(params["edge_trigger"])

    if use_ha:
        src = heikin_ashi(candles)
    else:
        src = candles

    close = [c["close"] for c in src]

    rsi_period = int(params["rsi_period"])
    rsi_vals = rsi(close, rsi_period)
    rsi_last = rsi_vals[-1]

    hist = macd_hist(close, int(params["macd_fast"]), int(params["macd_slow"]), int(params["macd_signal"]))
    hist_now = hist[-1]
    hist_prev = hist[-2] if len(hist) >= 2 else 0.0

    # weights
    wBR, wBM, wBC = normalize3(float(params["w_buy_rsi"]), float(params["w_buy_macd"]), float(params["w_buy_consec"]))
    wSR, wSM, wSC = normalize3(float(params["w_sell_rsi"]), float(params["w_sell_macd"]), float(params["w_sell_consec"]))

    b = (
        wBR * buy_rsi_signal(rsi_last, float(params["rsi_oversold"]), float(params["rsi_overbought"])) +
        wBM * buy_macd_signal(hist_now, hist_prev, edge) +
        wBC * consec_up(close, int(params["consec_green"]))
    )

    s = (
        wSR * sell_rsi_signal(rsi_last, float(params["rsi_oversold"]), float(params["rsi_overbought"])) +
        wSM * sell_macd_signal(hist_now, hist_prev, edge) +
        wSC * consec_down(close, int(params["consec_red"]))
    )

    return b, s, close[-1]


# =========================
# Binance helpers
# =========================

def round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step

def get_filters(client: Client, symbol: str):
    info = client.get_symbol_info(symbol)
    if not info:
        raise RuntimeError(f"No symbol info for {symbol}")
    tick = None
    step = None
    min_notional = None
    for f in info["filters"]:
        if f["filterType"] == "PRICE_FILTER":
            tick = float(f["tickSize"])
        elif f["filterType"] == "LOT_SIZE":
            step = float(f["stepSize"])
        elif f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
            # NOTIONAL aparece en algunos símbolos; usamos minNotional si existe
            min_notional = float(f.get("minNotional", f.get("notional", 0.0)))
    return tick, step, min_notional

def get_free_balance(client: Client, asset: str) -> float:
    bal = client.get_asset_balance(asset=asset)
    if not bal:
        return 0.0
    return float(bal["free"])

def fetch_klines_closed(client: Client, symbol: str, interval: str, limit: int):
    """
    Trae velas y devuelve SOLO velas cerradas (quita la vela actual en formación).
    """
    kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    # kline format: [open_time, open, high, low, close, volume, close_time, ...]
    candles = []
    for row in kl:
        candles.append({
            "open_time": int(row[0]),
            "close_time": int(row[6]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })
    # vela cerrada: close_time < now
    now_ms = int(time.time() * 1000)
    closed = [c for c in candles if c["close_time"] <= now_ms - 500]  # buffer
    # a veces Binance devuelve la última cerrada + la actual; nos quedamos con todas menos la última si parece "viva"
    if len(closed) >= 2 and closed[-1]["close_time"] > now_ms:
        closed = closed[:-1]
    return closed


# =========================
# Bot core
# =========================

class SpotBot:
    def __init__(self, client: Client, symbol: str, interval: str, params: dict, ui_cb=None):
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.params = params.copy()
        self.ui_cb = ui_cb  # callback para UI

        self.tick, self.step, self.min_notional = get_filters(client, symbol)

        self.running = False
        self.thread = None

        self.position: Position | None = None
        self.last_action_bar_close_time = None  # cooldown por vela cerrada

        self.closed_trades: list[Trade] = []
        self.open_trades: list[Trade] = []

        self._load_journal()

    def _emit(self, kind: str, payload: dict):
        if self.ui_cb:
            self.ui_cb(kind, payload)

    def _load_journal(self):
        if os.path.exists(JOURNAL_CSV):
            try:
                df = pd.read_csv(JOURNAL_CSV)
                # no reconstruimos todo perfecto, solo para vista; es opcional
            except Exception:
                pass

    def _append_journal(self, trade: Trade):
        row = asdict(trade)
        df = pd.DataFrame([row])
        if not os.path.exists(JOURNAL_CSV):
            df.to_csv(JOURNAL_CSV, index=False)
        else:
            df.to_csv(JOURNAL_CSV, mode="a", header=False, index=False)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def export_excel(self, path: str):
        open_rows = [asdict(t) for t in self.open_trades]
        closed_rows = [asdict(t) for t in self.closed_trades]
        df_open = pd.DataFrame(open_rows)
        df_closed = pd.DataFrame(closed_rows)
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            df_open.to_excel(w, index=False, sheet_name="abiertas")
            df_closed.to_excel(w, index=False, sheet_name="cerradas")

    def _now_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _cooldown_ok(self, last_closed_close_time: int) -> bool:
        cd = int(self.params["cooldown"])
        if cd <= 0:
            return True
        if self.last_action_bar_close_time is None:
            return True
        # cooldown en velas cerradas: si ya pasaron cd velas cerradas
        # usamos close_time para compararlo con índice en lista: más simple -> guardamos el close_time de la vela donde actuamos
        # y exigimos que hayan pasado cd cierres posteriores (aprox por tiempo)
        # Esto asume intervalos regulares.
        interval_sec = self._interval_seconds(self.interval)
        return (last_closed_close_time - self.last_action_bar_close_time) >= cd * interval_sec * 1000

    def _interval_seconds(self, interval: str) -> int:
        mapping = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400,
            "1d": 86400
        }
        return mapping.get(interval, 3600)

    def _place_market_buy_by_quote(self, quote_amount: float, reason: str):
        """
        Compra gastando quote_amount USDT (aprox).
        En SPOT Binance permite quoteOrderQty.
        Verificación: gasto real vs objetivo.
        """
        if quote_amount <= 0:
            return None

        if self.min_notional and quote_amount < self.min_notional:
            self._emit("log", {"msg": f"[BUY] quote {quote_amount:.4f} < minNotional {self.min_notional:.4f} -> no compro"})
            return None

        if DRY_RUN:
            # Simulación simple: asumimos ejecución a precio actual
            price = float(self.client.get_symbol_ticker(symbol=self.symbol)["price"])
            qty = quote_amount / price
            qty = round_step(qty, self.step)
            if qty <= 0:
                return None
            trade = Trade(
                trade_id=f"SIMBUY-{int(time.time())}",
                symbol=self.symbol,
                side="BUY",
                qty=qty,
                price=price,
                quote_spent=qty * price,
                time_utc=self._now_utc(),
                reason=reason,
                order_id="SIM",
                raw=""
            )
            return trade

        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quoteOrderQty=f"{quote_amount:.8f}"
            )
            fills = order.get("fills", [])
            if fills:
                qty_f = sum(float(f["qty"]) for f in fills)
                quote_f = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                avg = quote_f / qty_f if qty_f > 0 else 0.0
            else:
                qty_f = float(order.get("executedQty", 0.0))
                cumm = float(order.get("cummulativeQuoteQty", 0.0))
                avg = cumm / qty_f if qty_f > 0 else 0.0
                quote_f = cumm

            trade = Trade(
                trade_id=f"BUY-{order['orderId']}",
                symbol=self.symbol,
                side="BUY",
                qty=qty_f,
                price=avg,
                quote_spent=quote_f,
                time_utc=self._now_utc(),
                reason=reason,
                order_id=str(order["orderId"]),
                raw=json.dumps(order, ensure_ascii=False)
            )
            return trade

        except BinanceAPIException as e:
            self._emit("log", {"msg": f"[BUY] BinanceAPIException: {e}"})
            return None

    def _place_market_sell_qty(self, qty: float, reason: str):
        qty = round_step(qty, self.step)
        if qty <= 0:
            return None

        if DRY_RUN:
            price = float(self.client.get_symbol_ticker(symbol=self.symbol)["price"])
            trade = Trade(
                trade_id=f"SIMSELL-{int(time.time())}",
                symbol=self.symbol,
                side="SELL",
                qty=qty,
                price=price,
                quote_spent=qty * price,
                time_utc=self._now_utc(),
                reason=reason,
                order_id="SIM",
                raw=""
            )
            return trade

        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_MARKET,
                quantity=f"{qty:.8f}"
            )
            fills = order.get("fills", [])
            if fills:
                qty_f = sum(float(f["qty"]) for f in fills)
                quote_f = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                avg = quote_f / qty_f if qty_f > 0 else 0.0
            else:
                qty_f = float(order.get("executedQty", 0.0))
                cumm = float(order.get("cummulativeQuoteQty", 0.0))
                avg = cumm / qty_f if qty_f > 0 else 0.0
                quote_f = cumm

            trade = Trade(
                trade_id=f"SELL-{order['orderId']}",
                symbol=self.symbol,
                side="SELL",
                qty=qty_f,
                price=avg,
                quote_spent=quote_f,
                time_utc=self._now_utc(),
                reason=reason,
                order_id=str(order["orderId"]),
                raw=json.dumps(order, ensure_ascii=False)
            )
            return trade

        except BinanceAPIException as e:
            self._emit("log", {"msg": f"[SELL] BinanceAPIException: {e}"})
            return None

    def _open_position_from_trade(self, buy_trade: Trade, last_close_time_ms: int):
        ep = buy_trade.price
        qty = buy_trade.qty
        tp = ep * (1.0 + float(self.params["take_profit"]))
        sl = ep * (1.0 - float(self.params["stop_loss"]))
        self.position = Position(
            symbol=self.symbol,
            qty=qty,
            entry_price=ep,
            entry_time_utc=buy_trade.time_utc,
            tp_price=tp,
            sl_price=sl
        )
        self.last_action_bar_close_time = last_close_time_ms
        self.open_trades.append(buy_trade)
        self._append_journal(buy_trade)

        self._emit("position", {"pos": asdict(self.position)})

    def _close_position(self, reason: str, last_close_time_ms: int):
        if not self.position:
            return

        # vender todo el SOL libre (no uses position.qty a ciegas, puede variar por comisiones/redondeo)
        sol_free = get_free_balance(self.client, BASE_ASSET)
        qty = round_step(sol_free, self.step)
        if qty <= 0:
            self._emit("log", {"msg": "[CLOSE] No hay SOL libre para vender (solo dust)."})
            self.position = None
            return

        sell_trade = self._place_market_sell_qty(qty, reason=reason)
        if sell_trade:
            self.closed_trades.append(sell_trade)
            self._append_journal(sell_trade)

        self.position = None
        self.last_action_bar_close_time = last_close_time_ms
        self._emit("position", {"pos": None})

    def _loop(self):
        self._emit("log", {"msg": f"[BOT] Start {self.symbol} {self.interval} | DRY_RUN={DRY_RUN} | tick={self.tick} step={self.step} minNot={self.min_notional}"})

        last_seen_close_time = None

        while self.running:
            try:
                candles = fetch_klines_closed(self.client, self.symbol, self.interval, LOOKBACK)
                if len(candles) < 100:
                    time.sleep(2)
                    continue

                last_c = candles[-1]
                close_time = last_c["close_time"]

                # Procesa SOLO cuando aparece una nueva vela cerrada
                if last_seen_close_time is not None and close_time == last_seen_close_time:
                    time.sleep(2)
                    continue
                last_seen_close_time = close_time

                bScore, sScore, last_close = compute_scores(self.params, candles)

                in_pos = self.position is not None
                cool_ok = self._cooldown_ok(close_time)

                buy_cond  = (not in_pos) and cool_ok and (bScore >= float(self.params["buy_th"]))
                sell_cond = (in_pos) and cool_ok and (sScore >= float(self.params["sell_th"]))

                # TP/SL (si hay posición)
                tp_hit = sl_hit = False
                if in_pos:
                    if last_close >= self.position.tp_price:
                        tp_hit = True
                    if last_close <= self.position.sl_price:
                        sl_hit = True

                self._emit("tick", {
                    "time_utc": self._now_utc(),
                    "close": last_close,
                    "buyScore": bScore,
                    "sellScore": sScore,
                    "buyCond": buy_cond,
                    "sellCond": sell_cond,
                    "tp_hit": tp_hit,
                    "sl_hit": sl_hit
                })

                # Ejecutar
                if not in_pos and buy_cond:
                    usdt_free = get_free_balance(self.client, QUOTE_ASSET)
                    if usdt_free <= 0:
                        self._emit("log", {"msg": "[BUY] USDT libre = 0, no compro"})
                        continue

                    target = usdt_free * ORDER_PCT if USE_98_PERCENT else usdt_free
                    target = float(target)

                    # Verificación y reintento si llenó poco
                    spent_total = 0.0
                    last_trade = None
                    for attempt in range(MAX_RETRY_BUY + 1):
                        remaining_target = max(0.0, target - spent_total)
                        if remaining_target <= 0:
                            break

                        t = self._place_market_buy_by_quote(remaining_target, reason="SIGNAL_BUY")
                        if not t:
                            break
                        spent_total += float(t.quote_spent)
                        last_trade = t

                        fill_ratio = spent_total / target if target > 0 else 0.0
                        self._emit("log", {"msg": f"[BUY] attempt {attempt} spent={spent_total:.4f}/{target:.4f} ({fill_ratio*100:.1f}%) qty={t.qty:.6f} avg={t.price:.4f}"})

                        if fill_ratio >= MIN_FILL_RATIO:
                            break

                        time.sleep(RETRY_SLEEP_SEC)

                    if last_trade:
                        # Consolidar posición por “precio” usando último trade (simple). En real podrías promediar.
                        self._open_position_from_trade(last_trade, close_time)

                # Cierre por TP/SL primero
                if in_pos and tp_hit:
                    self._emit("log", {"msg": "[TP] Take Profit hit -> cierro"})
                    self._close_position(reason="TP", last_close_time_ms=close_time)
                    continue
                if in_pos and sl_hit:
                    self._emit("log", {"msg": "[SL] Stop Loss hit -> cierro"})
                    self._close_position(reason="SL", last_close_time_ms=close_time)
                    continue

                # Cierre por señal
                if in_pos and sell_cond:
                    self._emit("log", {"msg": "[SELL] Señal de cierre -> cierro"})
                    self._close_position(reason="SIGNAL_SELL", last_close_time_ms=close_time)
                    continue

            except Exception as e:
                self._emit("log", {"msg": f"[ERR] {e}"})
                time.sleep(2)

        self._emit("log", {"msg": "[BOT] Stop"})


# =========================
# GUI
# =========================

class BotGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bot SPOT (RSI+MACD+Consec) - Binance")

        # client
        self.client = None
        self.bot = None

        # Params (editable)
        self.params = DEFAULT_GEN.copy()

        # UI vars
        self.var_status = tk.StringVar(value="STOP")
        self.var_last = tk.StringVar(value="-")
        self.var_scores = tk.StringVar(value="bScore=- sScore=-")
        self.var_price = tk.StringVar(value="close=-")
        self.var_flags = tk.StringVar(value="buyCond=- sellCond=- tp=- sl=-")
        self.var_pos = tk.StringVar(value="pos: NONE")

        self._build()

    def _build(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)

        top = ttk.Frame(frm)
        top.pack(fill="x")

        ttk.Label(top, text="Estado:").grid(row=0, column=0, sticky="w")
        ttk.Label(top, textvariable=self.var_status).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(top, text="Último:").grid(row=1, column=0, sticky="w")
        ttk.Label(top, textvariable=self.var_last).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(top, textvariable=self.var_price).grid(row=0, column=2, sticky="w", padx=10)
        ttk.Label(top, textvariable=self.var_scores).grid(row=1, column=2, sticky="w", padx=10)
        ttk.Label(top, textvariable=self.var_flags).grid(row=2, column=2, sticky="w", padx=10)

        ttk.Label(top, textvariable=self.var_pos).grid(row=2, column=0, columnspan=2, sticky="w", pady=5)

        # Buttons
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=8)

        ttk.Button(btns, text="Conectar", command=self.connect).pack(side="left")
        ttk.Button(btns, text="Iniciar", command=self.start).pack(side="left", padx=5)
        ttk.Button(btns, text="Detener", command=self.stop).pack(side="left", padx=5)
        ttk.Button(btns, text="Exportar Excel", command=self.export_excel).pack(side="left", padx=5)
        ttk.Button(btns, text="Cargar GEN235", command=self.load_gen235).pack(side="right")

        # Params quick view
        p = ttk.LabelFrame(frm, text="Parámetros (GEN actual)")
        p.pack(fill="x", pady=8)

        self.txt_params = tk.Text(p, height=10, width=90)
        self.txt_params.pack(fill="x")
        self._refresh_params_box()

        # Trades table
        tfrm = ttk.LabelFrame(frm, text="Trades (cerradas)")
        tfrm.pack(fill="both", expand=True, pady=8)

        cols = ("time_utc", "side", "qty", "price", "quote_spent", "reason")
        self.tree = ttk.Treeview(tfrm, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="w")
        self.tree.pack(fill="both", expand=True)

        # Log
        lfrm = ttk.LabelFrame(frm, text="Log")
        lfrm.pack(fill="both", expand=True)

        self.txt_log = tk.Text(lfrm, height=8, width=90)
        self.txt_log.pack(fill="both", expand=True)

    def _refresh_params_box(self):
        self.txt_params.delete("1.0", "end")
        self.txt_params.insert("1.0", json.dumps(self.params, indent=2, ensure_ascii=False))

    def log(self, msg: str):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")

    def ui_callback(self, kind: str, payload: dict):
        # thread-safe update
        def _upd():
            if kind == "log":
                self.log(payload["msg"])
            elif kind == "tick":
                self.var_last.set(payload["time_utc"])
                self.var_price.set(f"close={payload['close']:.4f}")
                self.var_scores.set(f"bScore={payload['buyScore']:.3f} sScore={payload['sellScore']:.3f}")
                self.var_flags.set(f"buyCond={payload['buyCond']} sellCond={payload['sellCond']} tp={payload['tp_hit']} sl={payload['sl_hit']}")
            elif kind == "position":
                if payload["pos"] is None:
                    self.var_pos.set("pos: NONE")
                else:
                    pos = payload["pos"]
                    self.var_pos.set(f"pos: qty={pos['qty']:.6f} entry={pos['entry_price']:.4f} tp={pos['tp_price']:.4f} sl={pos['sl_price']:.4f}")
            # refresh table
            if self.bot:
                # render last 200 closed trades
                self.tree.delete(*self.tree.get_children())
                for tr in self.bot.closed_trades[-200:]:
                    self.tree.insert("", "end", values=(tr.time_utc, tr.side, f"{tr.qty:.6f}", f"{tr.price:.4f}", f"{tr.quote_spent:.4f}", tr.reason))

        self.root.after(0, _upd)

    def connect(self):
        try:
            key = read_key(API_KEY_PATH)
            sec = read_key(API_SECRET_PATH)
            self.client = Client(key, sec)
            self.client.ping()
            self.log("[OK] Conectado a Binance SPOT (ping ok).")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_gen235(self):
        self.params = DEFAULT_GEN.copy()
        self._refresh_params_box()
        self.log("[OK] GEN235 cargado en parámetros.")

    def start(self):
        if not self.client:
            messagebox.showwarning("Aviso", "Primero conecta.")
            return

        # leer params desde caja (por si editaste)
        try:
            txt = self.txt_params.get("1.0", "end").strip()
            self.params = json.loads(txt)
        except Exception as e:
            messagebox.showerror("Parámetros", f"JSON inválido: {e}")
            return

        self.bot = SpotBot(self.client, SYMBOL, INTERVAL, self.params, ui_cb=self.ui_callback)
        self.bot.start()
        self.var_status.set("RUN")
        self.log("[BOT] Iniciado.")

    def stop(self):
        if self.bot:
            self.bot.stop()
            self.var_status.set("STOP")
            self.log("[BOT] Detenido.")

    def export_excel(self):
        if not self.bot:
            messagebox.showwarning("Aviso", "No hay bot corriendo/creado.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")]
        )
        if not path:
            return
        try:
            self.bot.export_excel(path)
            self.log(f"[OK] Excel exportado: {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    gui = BotGUI(root)
    gui.log(f"DRY_RUN={DRY_RUN} | SYMBOL={SYMBOL} | INTERVAL={INTERVAL}")
    gui.log("TIP: Parte con DRY_RUN=True. Cuando estés seguro, cambia DRY_RUN=False.")
    root.mainloop()


if __name__ == "__main__":
    main()
