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
from typing import Optional
import requests


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
    def __init__(self, parent: tk.Misc):
        self.root = parent

        # Si parent es un Frame, igual podemos setear el título del toplevel (la ventana)
        toplevel = parent.winfo_toplevel()
        try:
            toplevel.title("Bot SPOT (RSI+MACD+Consec) - Binance")
        except Exception:
            pass

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

    def set_params(self, params: dict, log_msg: Optional[str] = None):
        self.params = params.copy()
        self._refresh_params_box()
        if log_msg:
            self.log(log_msg)

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


# =========================
# Optimizer / GA
# =========================

BINANCE_BASE = "https://api.binance.com"
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "45m": 2_700_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def fetch_klines_public(symbol: str, interval: str, limit: int):
    if interval not in INTERVAL_MS:
        raise ValueError(f"Intervalo no soportado: {interval}")

    out = []
    remaining = int(limit)
    end_time = None

    while remaining > 0:
        batch = min(1000, remaining)
        params = {"symbol": symbol.upper(), "interval": interval, "limit": batch}
        if end_time is not None:
            params["endTime"] = end_time

        r = requests.get(BINANCE_BASE + "/api/v3/klines", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        chunk = []
        for k in data:
            chunk.append({
                "t": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "v": float(k[5]),
            })

        out = chunk + out
        remaining -= len(chunk)

        first_t = chunk[0]["t"]
        end_time = first_t - 1

        if len(chunk) < batch:
            break

        time.sleep(0.12)

    if len(out) > limit:
        out = out[-limit:]
    return out


@dataclass
class Genome:
    use_ha: int
    rsi_period: int
    rsi_oversold: float
    rsi_overbought: float
    macd_fast: int
    macd_slow: int
    macd_signal: int
    consec_red: int
    consec_green: int

    w_buy_rsi: float
    w_buy_macd: float
    w_buy_consec: float
    buy_th: float

    w_sell_rsi: float
    w_sell_macd: float
    w_sell_consec: float
    sell_th: float

    take_profit: float
    stop_loss: float
    cooldown: int
    edge_trigger: int


class ParamSpace:
    def __init__(self):
        self.spec = {}

    def add(self, name: str, lo, hi, step, ptype: str):
        self.spec[name] = {"min": lo, "max": hi, "step": step, "type": ptype}

    def quantize(self, name: str, x):
        s = self.spec[name]
        lo, hi, st, tp = s["min"], s["max"], s["step"], s["type"]
        x = max(lo, min(x, hi))
        if tp == "int":
            return int(round(x))
        if st <= 0:
            return float(x)
        q = round((x - lo) / st) * st + lo
        q = float(f"{q:.10f}")
        return max(lo, min(q, hi))

    def sample(self):
        g = {}
        for k, s in self.spec.items():
            lo, hi, tp = s["min"], s["max"], s["type"]
            if tp == "int":
                g[k] = random.randint(int(lo), int(hi))
            else:
                g[k] = lo + (hi - lo) * random.random()
                g[k] = self.quantize(k, g[k])

        g["w_buy_rsi"], g["w_buy_macd"], g["w_buy_consec"] = normalize3(
            g["w_buy_rsi"], g["w_buy_macd"], g["w_buy_consec"]
        )
        g["w_sell_rsi"], g["w_sell_macd"], g["w_sell_consec"] = normalize3(
            g["w_sell_rsi"], g["w_sell_macd"], g["w_sell_consec"]
        )
        if g["macd_slow"] <= g["macd_fast"]:
            g["macd_slow"] = min(self.spec["macd_slow"]["max"], g["macd_fast"] + 5)
        return Genome(**g)

    def clamp_genome(self, ge: Genome):
        d = asdict(ge)
        for k in d:
            d[k] = self.quantize(k, d[k])

        d["w_buy_rsi"], d["w_buy_macd"], d["w_buy_consec"] = normalize3(
            d["w_buy_rsi"], d["w_buy_macd"], d["w_buy_consec"]
        )
        d["w_sell_rsi"], d["w_sell_macd"], d["w_sell_consec"] = normalize3(
            d["w_sell_rsi"], d["w_sell_macd"], d["w_sell_consec"]
        )
        if d["macd_slow"] <= d["macd_fast"]:
            d["macd_slow"] = min(self.spec["macd_slow"]["max"], d["macd_fast"] + 5)
        return Genome(**d)


@dataclass
class Metrics:
    score: float
    net: float
    max_dd: float
    dd_pct: float
    trades: int
    wins: int
    losses: int
    winrate: float
    pf: float


def simulate_spot(candles, ge: Genome, fee_per_side: float, slip_per_side: float):
    if len(candles) < 200:
        return Metrics(score=-1e9, net=-1.0, max_dd=1.0, dd_pct=100.0,
                       trades=0, wins=0, losses=0, winrate=0.0, pf=0.0)

    data = heikin_ashi(candles) if ge.use_ha == 1 else candles
    close = [c["close"] for c in data]

    rsi_arr = rsi(close, ge.rsi_period)
    mh_arr = macd_hist(close, ge.macd_fast, ge.macd_slow, ge.macd_signal)

    def buy_score(i: int) -> float:
        rsi_val = rsi_arr[i]
        mh = mh_arr[i]
        mh_prev = mh_arr[i - 1] if i > 0 else mh
        rsi_sig = buy_rsi_signal(rsi_val, ge.rsi_oversold, ge.rsi_overbought)
        macd_sig = buy_macd_signal(mh, mh_prev, ge.edge_trigger)
        cons_sig = consec_up(close[: i + 1], ge.consec_green)
        wbr, wbm, wbc = normalize3(ge.w_buy_rsi, ge.w_buy_macd, ge.w_buy_consec)
        return wbr * rsi_sig + wbm * macd_sig + wbc * cons_sig

    def sell_score(i: int) -> float:
        rsi_val = rsi_arr[i]
        mh = mh_arr[i]
        mh_prev = mh_arr[i - 1] if i > 0 else mh
        rsi_sig = sell_rsi_signal(rsi_val, ge.rsi_oversold, ge.rsi_overbought)
        macd_sig = sell_macd_signal(mh, mh_prev, ge.edge_trigger)
        cons_sig = consec_down(close[: i + 1], ge.consec_red)
        wsr, wsm, wsc = normalize3(ge.w_sell_rsi, ge.w_sell_macd, ge.w_sell_consec)
        return wsr * rsi_sig + wsm * macd_sig + wsc * cons_sig

    equity = 1.0
    peak = equity
    max_dd = 0.0

    in_pos = False
    entry = 0.0
    entry_i = -10_000
    last_trade_i = -10_000

    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    trades = 0

    for i in range(2, len(close)):
        peak = max(peak, equity)
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

        sig_i = i - 1
        exec_px = close[i]

        bs = buy_score(sig_i)
        ss = sell_score(sig_i)

        if i - last_trade_i < int(ge.cooldown):
            continue

        if not in_pos:
            if bs >= ge.buy_th:
                buy_price = exec_px * (1.0 + slip_per_side)
                equity *= (1.0 - fee_per_side)
                entry = buy_price
                in_pos = True
                entry_i = i
                last_trade_i = i
        else:
            if i <= entry_i:
                continue

            px = exec_px
            tp_px = entry * (1.0 + float(ge.take_profit))
            sl_px = entry * (1.0 - float(ge.stop_loss))

            exit_reason = None
            if px >= tp_px:
                exit_reason = "TP"
            elif px <= sl_px:
                exit_reason = "SL"
            elif ss >= ge.sell_th:
                exit_reason = "SIG"

            if exit_reason is not None:
                sell_price = px * (1.0 - slip_per_side)
                gross_ratio = (sell_price / (entry + 1e-12))
                trade_ratio = (1.0 - fee_per_side) * gross_ratio
                trade_ret = trade_ratio - 1.0

                equity *= trade_ratio

                trades += 1
                if trade_ret >= 0:
                    wins += 1
                    gross_profit += trade_ret
                else:
                    losses += 1
                    gross_loss += (-trade_ret)

                in_pos = False
                entry = 0.0
                entry_i = -10_000
                last_trade_i = i

    net = equity - 1.0
    dd_pct = (max_dd / (peak + 1e-12)) * 100.0

    pf_raw = gross_profit / (gross_loss + 1e-6)
    pf = max(0.0, min(pf_raw, 10.0))
    winrate = (wins / trades * 100.0) if trades > 0 else 0.0

    score = math.log1p(max(0.0, pf)) + net - (max_dd * 1.0)

    return Metrics(score=score, net=net, max_dd=max_dd, dd_pct=dd_pct,
                   trades=trades, wins=wins, losses=losses, winrate=winrate, pf=pf)


@dataclass
class GAConfig:
    population: int
    generations: int
    elite: int
    n_cons: int
    n_exp: int
    n_wild: int

    fee_side: float
    fee_mult: float
    slip_side: float

    dd_weight: float
    trade_floor: int
    pen_per_missing_trade: float


BLOCKS = {
    "RSI": ["rsi_period", "rsi_oversold", "rsi_overbought"],
    "MACD": ["macd_fast", "macd_slow", "macd_signal"],
    "CONSEC": ["consec_red", "consec_green"],
    "RISK": ["take_profit", "stop_loss", "cooldown", "edge_trigger", "use_ha"],
    "CONF": [
        "w_buy_rsi", "w_buy_macd", "w_buy_consec", "buy_th",
        "w_sell_rsi", "w_sell_macd", "w_sell_consec", "sell_th",
    ],
}


def mutate_block(child, space: ParamSpace, block_name: str, strength: float):
    keys = BLOCKS[block_name]
    for gk in keys:
        spec = space.spec[gk]
        lo, hi, tp = spec["min"], spec["max"], spec["type"]

        if tp == "int":
            if random.random() < (0.60 * strength + 0.15):
                child[gk] = random.randint(int(lo), int(hi))
            else:
                cur = int(child[gk])
                step = max(1, int(spec["step"]))
                delta = step * random.randint(1, max(1, int(1 + 4 * strength)))
                child[gk] = int(max(lo, min(cur + (delta if random.random() < 0.5 else -delta), hi)))
        else:
            span = (hi - lo)
            if random.random() < (0.60 * strength + 0.15):
                child[gk] = lo + (hi - lo) * random.random()
            else:
                cur = float(child[gk])
                delta = span * (0.03 + 0.18 * strength * random.random())
                child[gk] = cur + (delta if random.random() < 0.5 else -delta)
                child[gk] = space.quantize(gk, child[gk])

        child[gk] = space.quantize(gk, child[gk])

    if block_name == "CONF":
        child["w_buy_rsi"], child["w_buy_macd"], child["w_buy_consec"] = normalize3(
            child["w_buy_rsi"], child["w_buy_macd"], child["w_buy_consec"]
        )
        child["w_sell_rsi"], child["w_sell_macd"], child["w_sell_consec"] = normalize3(
            child["w_sell_rsi"], child["w_sell_macd"], child["w_sell_consec"]
        )

    if block_name == "MACD":
        if child["macd_slow"] <= child["macd_fast"]:
            child["macd_slow"] = min(space.spec["macd_slow"]["max"], child["macd_fast"] + 5)
            child["macd_slow"] = space.quantize("macd_slow", child["macd_slow"])


def make_child_blocky(
    space: ParamSpace,
    p1: Genome,
    p2: Genome,
    mut_rate: float,
    max_blocks_per_child: int,
    forced_block: Optional[str] = None,
    strength: float = 0.8,
):
    d1 = asdict(p1)
    d2 = asdict(p2)
    child = {}

    for k in d1:
        child[k] = d1[k] if random.random() < 0.5 else d2[k]

    if random.random() < mut_rate:
        block_names = list(BLOCKS.keys())
        random.shuffle(block_names)

        chosen = []
        if forced_block and forced_block in BLOCKS:
            chosen.append(forced_block)

        need = random.randint(1, max_blocks_per_child)
        for bn in block_names:
            if bn in chosen:
                continue
            chosen.append(bn)
            if len(chosen) >= need:
                break

        for bn in chosen:
            mutate_block(child, space, bn, strength=strength)

    ge = Genome(**child)
    return space.clamp_genome(ge)


def force_coverage(space: ParamSpace, n: int):
    keys = list(space.spec.keys())
    out = []
    for i in range(n):
        g = space.sample()
        d = asdict(g)
        k = keys[i % len(keys)]
        if space.spec[k]["type"] == "int":
            d[k] = int(space.spec[k]["min"] if (i // len(keys)) % 2 == 0 else space.spec[k]["max"])
        else:
            d[k] = float(space.spec[k]["min"] if (i // len(keys)) % 2 == 0 else space.spec[k]["max"])
        ge = space.clamp_genome(Genome(**d))
        out.append(ge)
    return out


def run_ga(candles, space: ParamSpace, cfg: GAConfig, stop_flag, log_fn):
    fee_per_side = cfg.fee_side * cfg.fee_mult
    slip_per_side = cfg.slip_side

    log_fn(f"[COSTOS] SPOT | fee_lado={fee_per_side:.6f} (incluye mult) | slip_lado={slip_per_side:.6f}")

    pop = [space.sample() for _ in range(cfg.population)]
    cov = force_coverage(space, max(12, cfg.population // 6))
    pop[:len(cov)] = cov
    log_fn(f"[COBERTURA] forcé {len(cov)} individuos para cubrir rangos completos (inicio)")

    best_global = None
    best_metrics = None
    stuck = 0

    for gen in range(1, cfg.generations + 1):
        if stop_flag():
            log_fn("[STOP] detenido por el usuario.")
            return best_global, best_metrics

        scored = []
        for ge in pop:
            m = simulate_spot(candles, ge, fee_per_side, slip_per_side)

            missing = max(0, cfg.trade_floor - m.trades)
            penalty_trades = missing * cfg.pen_per_missing_trade
            score = (math.log1p(max(0.0, m.pf)) + m.net) - (cfg.dd_weight * m.max_dd) - penalty_trades

            scored.append((score, ge, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_ge, best_m = scored[0]

        if best_global is None or best_score > best_global[0] + 1e-9:
            best_global = (best_score, best_ge)
            best_metrics = best_m
            stuck = 0
        else:
            stuck += 1

        log_fn(
            f"[GEN {gen}] score={best_score:.4f} | net={best_m.net:.4f} | "
            f"DD={best_m.max_dd:.4f} ({best_m.dd_pct:.2f}%) | trades={best_m.trades} "
            f"wr={best_m.winrate:.1f}% PF={best_m.pf:.2f} | "
            f"HA={best_ge.use_ha} RSI(p={best_ge.rsi_period},os={best_ge.rsi_oversold:.0f},ob={best_ge.rsi_overbought:.0f}) | "
            f"MACD({best_ge.macd_fast},{best_ge.macd_slow},{best_ge.macd_signal}) | "
            f"consec(R={best_ge.consec_red},G={best_ge.consec_green}) | "
            f"TP={best_ge.take_profit:.3f} SL={best_ge.stop_loss:.3f} cd={best_ge.cooldown} edge={best_ge.edge_trigger} | "
            f"buy_th={best_ge.buy_th:.2f} sell_th={best_ge.sell_th:.2f} | "
            f"Wbuy({best_ge.w_buy_rsi:.2f},{best_ge.w_buy_macd:.2f},{best_ge.w_buy_consec:.2f}) "
            f"Wsell({best_ge.w_sell_rsi:.2f},{best_ge.w_sell_macd:.2f},{best_ge.w_sell_consec:.2f})"
        )

        if stuck >= 15:
            stuck = 0
            log_fn("[ANTI-PEGADO] se pegó -> meto más wild + más cobertura")
            cfg.n_wild = min(cfg.population - cfg.elite, cfg.n_wild + 20)
            extra_cov = force_coverage(space, max(10, cfg.population // 8))
        else:
            extra_cov = []

        elite = [x[1] for x in scored[:cfg.elite]]
        pool = [x[1] for x in scored[:max(cfg.elite * 6, cfg.population // 2)]]

        children = []

        block_cycle = ["RSI", "MACD", "CONSEC", "RISK", "CONF"]
        bc_i = 0

        for _ in range(cfg.n_cons):
            p1 = random.choice(pool)
            p2 = random.choice(pool)
            forced = block_cycle[bc_i % len(block_cycle)]
            bc_i += 1
            children.append(make_child_blocky(space, p1, p2, mut_rate=0.28, max_blocks_per_child=1,
                                              forced_block=forced, strength=0.55))

        for _ in range(cfg.n_exp):
            p1 = random.choice(pool)
            p2 = random.choice(pool)
            forced = block_cycle[bc_i % len(block_cycle)]
            bc_i += 1
            children.append(make_child_blocky(space, p1, p2, mut_rate=0.60, max_blocks_per_child=2,
                                              forced_block=forced, strength=0.80))

        for _ in range(cfg.n_wild):
            if random.random() < 0.35:
                children.append(space.sample())
            else:
                p1 = random.choice(pool)
                p2 = random.choice(pool)
                forced = block_cycle[bc_i % len(block_cycle)]
                bc_i += 1
                children.append(make_child_blocky(space, p1, p2, mut_rate=0.90, max_blocks_per_child=3,
                                                  forced_block=forced, strength=1.00))

        new_pop = elite + children
        if extra_cov:
            k = min(len(extra_cov), max(5, cfg.population // 10))
            new_pop[-k:] = extra_cov[:k]

        if len(new_pop) > cfg.population:
            new_pop = new_pop[:cfg.population]
        while len(new_pop) < cfg.population:
            new_pop.append(space.sample())

        pop = new_pop

    return best_global, best_metrics


class OptimizerGUI:
    def __init__(self, root: tk.Widget, apply_callback):
        self.root = root
        self.apply_callback = apply_callback

        self.worker = None
        self._stop = False
        self.best_genome = None
        self.best_metrics = None
        self.cached_candles = None

        self.var_symbol = tk.StringVar(value=SYMBOL)
        self.var_tf = tk.StringVar(value=INTERVAL)
        self.var_candles = tk.IntVar(value=3000)

        self.var_pop = tk.IntVar(value=160)
        self.var_gens = tk.IntVar(value=120)
        self.var_elite = tk.IntVar(value=6)

        self.var_cons = tk.IntVar(value=50)
        self.var_exp = tk.IntVar(value=50)
        self.var_wild = tk.IntVar(value=54)

        self.var_fee = tk.DoubleVar(value=0.001)
        self.var_fee_mult = tk.DoubleVar(value=1.1)
        self.var_slip = tk.DoubleVar(value=0.0002)

        self.var_dd_w = tk.DoubleVar(value=2.5)
        self.var_trade_floor = tk.IntVar(value=120)
        self.var_pen_missing = tk.DoubleVar(value=0.35)

        self.space = self.build_space_defaults()
        self.build_ui()

    def build_space_defaults(self):
        sp = ParamSpace()
        sp.add("use_ha", 0, 1, 1, "int")

        sp.add("rsi_period", 6, 30, 1, "int")
        sp.add("rsi_oversold", 20.0, 40.0, 1.0, "float")
        sp.add("rsi_overbought", 60.0, 90.0, 1.0, "float")

        sp.add("macd_fast", 3, 25, 1, "int")
        sp.add("macd_slow", 10, 120, 1, "int")
        sp.add("macd_signal", 3, 50, 1, "int")

        sp.add("consec_red", 1, 8, 1, "int")
        sp.add("consec_green", 1, 8, 1, "int")

        sp.add("w_buy_rsi", 0.0, 1.0, 0.01, "float")
        sp.add("w_buy_macd", 0.0, 1.0, 0.01, "float")
        sp.add("w_buy_consec", 0.0, 1.0, 0.01, "float")
        sp.add("buy_th", 0.45, 0.85, 0.01, "float")

        sp.add("w_sell_rsi", 0.0, 1.0, 0.01, "float")
        sp.add("w_sell_macd", 0.0, 1.0, 0.01, "float")
        sp.add("w_sell_consec", 0.0, 1.0, 0.01, "float")
        sp.add("sell_th", 0.45, 0.85, 0.01, "float")

        sp.add("take_profit", 0.01, 0.08, 0.001, "float")
        sp.add("stop_loss", 0.005, 0.06, 0.001, "float")

        sp.add("cooldown", 0, 12, 1, "int")
        sp.add("edge_trigger", 0, 1, 1, "int")
        return sp

    def build_ui(self):
        pad = 6
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)

        top = ttk.Frame(frm)
        top.pack(fill="x", padx=pad, pady=pad)

        def add_labeled(entry_parent, label, var, w=10):
            f = ttk.Frame(entry_parent)
            ttk.Label(f, text=label).pack(side="left")
            e = ttk.Entry(f, textvariable=var, width=w)
            e.pack(side="left", padx=4)
            f.pack(side="left", padx=8)
            return e

        add_labeled(top, "Símbolo:", self.var_symbol, 10)
        ttk.Label(top, text="Temporalidad:").pack(side="left", padx=4)
        cb = ttk.Combobox(top, textvariable=self.var_tf, values=list(INTERVAL_MS.keys()), width=6, state="readonly")
        cb.pack(side="left", padx=4)
        add_labeled(top, "Velas:", self.var_candles, 8)

        mid = ttk.Frame(frm)
        mid.pack(fill="x", padx=pad, pady=pad)
        add_labeled(mid, "Población:", self.var_pop, 8)
        add_labeled(mid, "Generaciones:", self.var_gens, 8)
        add_labeled(mid, "Elite:", self.var_elite, 6)

        kids = ttk.Frame(frm)
        kids.pack(fill="x", padx=pad, pady=pad)
        add_labeled(kids, "Hijos cons:", self.var_cons, 6)
        add_labeled(kids, "exp:", self.var_exp, 6)
        add_labeled(kids, "wild:", self.var_wild, 6)

        costs = ttk.Frame(frm)
        costs.pack(fill="x", padx=pad, pady=pad)
        add_labeled(costs, "fee/lado:", self.var_fee, 8)
        add_labeled(costs, "mult fee:", self.var_fee_mult, 6)
        add_labeled(costs, "slip/lado:", self.var_slip, 8)

        fit = ttk.Frame(frm)
        fit.pack(fill="x", padx=pad, pady=pad)
        add_labeled(fit, "DD weight:", self.var_dd_w, 6)
        add_labeled(fit, "Trade floor:", self.var_trade_floor, 8)
        add_labeled(fit, "Pen/trade falt.:", self.var_pen_missing, 8)

        ranges = ttk.LabelFrame(frm, text="Rangos (min / max / step)")
        ranges.pack(fill="both", expand=False, padx=pad, pady=pad)

        self.range_entries = {}
        headers = ("Parámetro", "Min", "Max", "Step", "Tipo")
        for j, h in enumerate(headers):
            ttk.Label(ranges, text=h, font=("Segoe UI", 9, "bold")).grid(row=0, column=j, padx=6, pady=4, sticky="w")
            ttk.Label(ranges, text=h, font=("Segoe UI", 9, "bold")).grid(row=0, column=j + 6, padx=6, pady=4, sticky="w")

        items = list(self.space.spec.items())
        per_col = math.ceil(len(items) / 2)
        separator = ttk.Separator(ranges, orient="vertical")
        separator.grid(row=0, column=5, rowspan=per_col + 1, sticky="ns", padx=4)

        for idx, (name, s) in enumerate(items):
            col_block = 0 if idx < per_col else 1
            row = 1 + (idx if idx < per_col else idx - per_col)
            base_col = col_block * 6

            ttk.Label(ranges, text=self.pretty_name(name)).grid(row=row, column=base_col + 0, padx=6, pady=2, sticky="w")
            vmin = tk.StringVar(value=str(s["min"]))
            vmax = tk.StringVar(value=str(s["max"]))
            vstep = tk.StringVar(value=str(s["step"]))
            ttk.Entry(ranges, textvariable=vmin, width=10).grid(row=row, column=base_col + 1, padx=6, pady=2)
            ttk.Entry(ranges, textvariable=vmax, width=10).grid(row=row, column=base_col + 2, padx=6, pady=2)
            ttk.Entry(ranges, textvariable=vstep, width=10).grid(row=row, column=base_col + 3, padx=6, pady=2)
            ttk.Label(ranges, text=s["type"]).grid(row=row, column=base_col + 4, padx=6, pady=2, sticky="w")
            self.range_entries[name] = (vmin, vmax, vstep)

        btns = ttk.Frame(frm)
        btns.pack(fill="x", padx=pad, pady=pad)

        ttk.Button(btns, text="Guardar parámetros", command=self.save_params).pack(side="left", padx=4)
        ttk.Button(btns, text="Cargar parámetros", command=self.load_params).pack(side="left", padx=4)
        ttk.Button(btns, text="Iniciar optimización", command=self.start).pack(side="left", padx=20)
        ttk.Button(btns, text="Detener", command=self.stop).pack(side="left")
        ttk.Button(btns, text="Backtest rápido", command=self.quick_backtest).pack(side="left", padx=8)
        ttk.Button(btns, text="Usar mejor en Bot", command=self.apply_best).pack(side="right")

        self.txt = tk.Text(frm, height=16, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=pad, pady=pad)

        self.log("Listo. Defaults cargados (recomendados).")

    def pretty_name(self, key: str) -> str:
        m = {
            "use_ha": "Usar Heikin Ashi (0/1)",
            "rsi_period": "RSI periodo",
            "rsi_oversold": "RSI sobrevendido",
            "rsi_overbought": "RSI sobrecomprado",
            "macd_fast": "MACD rápido",
            "macd_slow": "MACD lento",
            "macd_signal": "MACD señal",
            "consec_red": "Consec rojas N",
            "consec_green": "Consec verdes N",
            "w_buy_rsi": "Peso compra RSI",
            "w_buy_macd": "Peso compra MACD",
            "w_buy_consec": "Peso compra Consecutivas",
            "buy_th": "Umbral compra",
            "w_sell_rsi": "Peso venta RSI",
            "w_sell_macd": "Peso venta MACD",
            "w_sell_consec": "Peso venta Consecutivas",
            "sell_th": "Umbral venta",
            "take_profit": "Take Profit",
            "stop_loss": "Stop Loss",
            "cooldown": "Cooldown (velas)",
            "edge_trigger": "EdgeTrigger (0/1)",
        }
        return m.get(key, key)

    def log(self, s: str):
        print(s, flush=True)
        try:
            self.txt.insert("end", s + "\n")
            self.txt.see("end")
            self.root.update_idletasks()
        except Exception:
            pass

    def read_space_from_ui(self):
        for name, (vmin, vmax, vstep) in self.range_entries.items():
            s = self.space.spec[name]
            try:
                smin = float(vmin.get())
                smax = float(vmax.get())
                sst = float(vstep.get())
            except Exception:
                raise ValueError(f"Rango inválido en {name}")
            if s["type"] == "int":
                smin, smax, sst = int(round(smin)), int(round(smax)), int(round(sst)) if sst != 0 else 1
                if sst <= 0:
                    sst = 1
            else:
                if sst <= 0:
                    sst = s["step"]
            if smax < smin:
                smin, smax = smax, smin
            s["min"], s["max"], s["step"] = smin, smax, sst

    def build_cfg(self):
        pop = int(self.var_pop.get())
        elite = int(self.var_elite.get())
        n_cons = int(self.var_cons.get())
        n_exp = int(self.var_exp.get())
        n_wild = int(self.var_wild.get())
        if elite < 1:
            elite = 1
        if elite >= pop:
            elite = max(1, pop // 10)

        total_kids = n_cons + n_exp + n_wild
        max_kids = pop - elite
        if total_kids > max_kids:
            scale = max_kids / max(1, total_kids)
            n_cons = int(round(n_cons * scale))
            n_exp = int(round(n_exp * scale))
            n_wild = max_kids - n_cons - n_exp

        return GAConfig(
            population=pop,
            generations=int(self.var_gens.get()),
            elite=elite,
            n_cons=n_cons,
            n_exp=n_exp,
            n_wild=n_wild,
            fee_side=float(self.var_fee.get()),
            fee_mult=float(self.var_fee_mult.get()),
            slip_side=float(self.var_slip.get()),
            dd_weight=float(self.var_dd_w.get()),
            trade_floor=int(self.var_trade_floor.get()),
            pen_per_missing_trade=float(self.var_pen_missing.get()),
        )

    def save_params(self):
        try:
            self.read_space_from_ui()
            cfg = self.build_cfg()
            payload = {
                "symbol": self.var_symbol.get().strip(),
                "timeframe": self.var_tf.get().strip(),
                "candles": int(self.var_candles.get()),
                "ga": asdict(cfg),
                "space": self.space.spec,
            }
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            self.log(f"[OK] Guardado: {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_params(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
            if not path:
                return
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.var_symbol.set(payload.get("symbol", SYMBOL))
            self.var_tf.set(payload.get("timeframe", INTERVAL))
            self.var_candles.set(int(payload.get("candles", 3000)))

            ga = payload.get("ga", {})
            self.var_pop.set(int(ga.get("population", self.var_pop.get())))
            self.var_gens.set(int(ga.get("generations", self.var_gens.get())))
            self.var_elite.set(int(ga.get("elite", self.var_elite.get())))
            self.var_cons.set(int(ga.get("n_cons", self.var_cons.get())))
            self.var_exp.set(int(ga.get("n_exp", self.var_exp.get())))
            self.var_wild.set(int(ga.get("n_wild", self.var_wild.get())))

            self.var_fee.set(float(ga.get("fee_side", self.var_fee.get())))
            self.var_fee_mult.set(float(ga.get("fee_mult", self.var_fee_mult.get())))
            self.var_slip.set(float(ga.get("slip_side", self.var_slip.get())))

            self.var_dd_w.set(float(ga.get("dd_weight", self.var_dd_w.get())))
            self.var_trade_floor.set(int(ga.get("trade_floor", self.var_trade_floor.get())))
            self.var_pen_missing.set(float(ga.get("pen_per_missing_trade", self.var_pen_missing.get())))

            space = payload.get("space", {})
            for k in self.space.spec:
                if k in space:
                    self.space.spec[k].update(space[k])
                    vmin, vmax, vstep = self.range_entries[k]
                    vmin.set(str(self.space.spec[k]["min"]))
                    vmax.set(str(self.space.spec[k]["max"]))
                    vstep.set(str(self.space.spec[k]["step"]))

            self.log(f"[OK] Cargado: {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Info", "Ya está corriendo.")
            return

        try:
            self.read_space_from_ui()
            cfg = self.build_cfg()
            symbol = self.var_symbol.get().strip().upper()
            tf = self.var_tf.get().strip()
            n = int(self.var_candles.get())

            self._stop = False
            self.txt.delete("1.0", "end")

            self.log(f"[OK] Temporalidad objetivo: {tf} -> {n} velas")
            self.log(f"[GA] pop={cfg.population} gens={cfg.generations} elite={cfg.elite} | cons={cfg.n_cons} exp={cfg.n_exp} wild={cfg.n_wild}")
            self.log("[DESCARGA] Bajando datos desde Binance...")

            def job():
                try:
                    candles = fetch_klines_public(symbol, tf, n)
                    self.cached_candles = candles
                    self.log(f"[OK] {symbol} cargado: {len(candles)} velas ({tf})")
                    best_global, best_metrics = run_ga(
                        candles=candles,
                        space=self.space,
                        cfg=cfg,
                        stop_flag=lambda: self._stop,
                        log_fn=lambda s: self.root.after(0, self.log, s),
                    )
                    self.best_genome = best_global[1] if best_global else None
                    self.best_metrics = best_metrics
                    if self.best_genome:
                        self.root.after(0, self.log, "[FIN] Mejor encontrado:")
                        self.root.after(0, self.log, json.dumps(asdict(self.best_genome), indent=2, ensure_ascii=False))
                except Exception as e:
                    self.root.after(0, lambda err=e: messagebox.showerror("Error", str(err)))

            self.worker = threading.Thread(target=job, daemon=True)
            self.worker.start()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def stop(self):
        self._stop = True
        self.log("[STOP] señal enviada.")

    def quick_backtest(self):
        try:
            if not self.cached_candles:
                symbol = self.var_symbol.get().strip().upper()
                tf = self.var_tf.get().strip()
                n = int(self.var_candles.get())
                self.log("[DESCARGA] Bajando datos desde Binance para backtest...")
                self.cached_candles = fetch_klines_public(symbol, tf, n)

            ge = self.space.sample()
            metrics = simulate_spot(self.cached_candles, ge, self.var_fee.get() * self.var_fee_mult.get(), self.var_slip.get())
            self.log(
                f"[BACKTEST] net={metrics.net:.4f} DD={metrics.max_dd:.4f} ({metrics.dd_pct:.2f}%) "
                f"trades={metrics.trades} wr={metrics.winrate:.1f}% PF={metrics.pf:.2f}"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_best(self):
        if not self.best_genome:
            messagebox.showwarning("Aviso", "Primero ejecuta la optimización.")
            return
        params = asdict(self.best_genome)
        self.apply_callback(params)
        self.log("[OK] Parámetros aplicados al Bot.")


def build_main_ui(root: tk.Tk):
    root.title("Bot + Optimizador (RSI+MACD+Consec)")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    opt_frame = ttk.Frame(notebook)
    bot_frame = ttk.Frame(notebook)

    notebook.add(opt_frame, text="Optimizador")
    notebook.add(bot_frame, text="Bot")

    bot_gui = BotGUI(bot_frame)

    def apply_params_to_bot(params):
        bot_gui.set_params(params, log_msg="[OK] Parámetros del optimizador cargados en Bot.")

    if "OptimizerGUI" not in globals():
        messagebox.showerror(
            "Error",
            "OptimizerGUI no está definido. Asegúrate de usar el archivo completo con la sección del optimizador.",
        )
        return bot_gui

    OptimizerGUI(opt_frame, apply_params_to_bot)

    bot_gui.log(f"DRY_RUN={DRY_RUN} | SYMBOL={SYMBOL} | INTERVAL={INTERVAL}")
    bot_gui.log("TIP: Parte con DRY_RUN=True. Cuando estés seguro, cambia DRY_RUN=False.")

    return bot_gui


def main():
    root = tk.Tk()
    build_main_ui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
