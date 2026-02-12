#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BOT SPOT Binance - RSI + MACD + Consecutivas (con HA opcional) + GUI + Excel
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
import sys
import time
import math
import json
import random
import statistics
import copy
import threading
import traceback
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
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

# Rango de fechas para simulador/optimizador
START_DATE = "2023-01-01"
END_DATE = "2025-01-01"
CACHE_INDICATORS = False
OUT_DIR = r"C:\Users\EQCONF\Documents\programas python\historial de velas"

# Logs / journal
JOURNAL_CSV = "bot_journal.csv"

# Debug de mutaciones (GA)
DEBUG_MUTATION = False
DEBUG_FULL_POPULATION = False
DEBUG_GA_PROGRESS = True
GA_PROGRESS_EVERY = 25
GA_INDIVIDUAL_WARN_S = 5.0

BLOCK_STUCK_LIMIT = 5

# Capital inicial para reportes de optimización
START_CAPITAL = 100000.0

# Red / Binance
BINANCE_TIMEOUT_SEC = 30
BINANCE_MAX_RETRIES = 3
BINANCE_RETRY_BACKOFF_SEC = 1.5
SELL_NOTIONAL_BUFFER_USDT = 1.0
BUY_NOTIONAL_BUFFER_USDT = 1.0
RECONNECT_MAX_RETRIES = 5
RECONNECT_BACKOFF_SEC = 2.0
GUI_CONNECT_RETRY_SEC = 5.0


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


def normalize_tp_sl(tp_input: float, sl_input: float) -> tuple[float, float]:
    # [TP/SL ABS NORMALIZATION] magnitudes only (no signo)
    tp_pct = abs(float(tp_input))
    sl_pct = abs(float(sl_input))
    if tp_pct >= 1.0 or sl_pct >= 1.0:
        raise ValueError("TP/SL inválido: porcentaje debe ser < 1 (100%).")
    return tp_pct, sl_pct


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
    entry_price: float = 0.0
    tp_pct: float = 0.0
    sl_pct: float = 0.0
    tp_price: float = 0.0
    sl_price: float = 0.0
    pnl_pct_real: float = 0.0
    pnl_usdt_real: float = 0.0
    order_id: str = ""
    raw: str = ""


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_quote_spent: float
    entry_time_utc: str
    tp_price: Optional[float]
    sl_price: Optional[float]
    tp_pct: float
    sl_pct: float


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

def ema_series(series, length: int):
    if length <= 0:
        return [series[0]] * len(series)
    alpha = 2.0 / (length + 1.0)
    out = []
    prev = series[0]
    out.append(prev)
    for x in series[1:]:
        prev = prev + alpha * (x - prev)
        out.append(prev)
    return out

def rsi_wilder_series(close, period: int):
    period = max(2, int(period))
    deltas = [0.0]
    for i in range(1, len(close)):
        deltas.append(close[i] - close[i - 1])
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    alpha = 1.0 / period
    avg_gain = gains[0]
    avg_loss = losses[0]
    rsis = [50.0] * len(close)
    for i in range(1, len(close)):
        avg_gain = avg_gain + alpha * (gains[i] - avg_gain)
        avg_loss = avg_loss + alpha * (losses[i] - avg_loss)
        if avg_loss <= 1e-12:
            rsis[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsis[i] = 100 - (100 / (1 + rs))
    return rsis

def macd_hist_series(close, fast: int, slow: int, signal: int):
    fast = max(2, int(fast))
    slow = max(fast + 1, int(slow))
    signal = max(2, int(signal))
    ema_fast = ema_series(close, fast)
    ema_slow = ema_series(close, slow)
    macd_line = [a - b for a, b in zip(ema_fast, ema_slow)]
    macd_signal = ema_series(macd_line, signal)
    return [a - b for a, b in zip(macd_line, macd_signal)]

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

def normalize3_safe(a: float, b: float, c: float):
    a = max(0.0, a)
    b = max(0.0, b)
    c = max(0.0, c)
    s = a + b + c
    if s <= 1e-12:
        r1, r2, r3 = random.random(), random.random(), random.random()
        s = r1 + r2 + r3
        return (r1 / s, r2 / s, r3 / s)
    return (a / s, b / s, c / s)

WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.90

def clamp_weight(value: float, min_w: float = WEIGHT_MIN, max_w: float = WEIGHT_MAX) -> float:
    return max(min_w, min(max_w, value))

def normalize3_clamped(a: float, b: float, c: float, min_w: float = WEIGHT_MIN, max_w: float = WEIGHT_MAX):
    a = clamp_weight(a, min_w, max_w)
    b = clamp_weight(b, min_w, max_w)
    c = clamp_weight(c, min_w, max_w)
    s = a + b + c
    if s <= 1e-12:
        return (1 / 3, 1 / 3, 1 / 3)
    return (a / s, b / s, c / s)

def apply_weight_constraints(child: dict):
    child["w_buy_rsi"], child["w_buy_macd"], child["w_buy_consec"] = normalize3_clamped(
        child["w_buy_rsi"], child["w_buy_macd"], child["w_buy_consec"]
    )
    child["w_sell_rsi"], child["w_sell_macd"], child["w_sell_consec"] = normalize3_clamped(
        child["w_sell_rsi"], child["w_sell_macd"], child["w_sell_consec"]
    )

def score_components_at_index(
    close: list,
    rsi_arr: list,
    mh_arr: list,
    i: int,
    *,
    rsi_oversold: float,
    rsi_overbought: float,
    consec_red: int,
    consec_green: int,
    edge_trigger: int,
    w_buy_rsi: float,
    w_buy_macd: float,
    w_buy_consec: float,
    w_sell_rsi: float,
    w_sell_macd: float,
    w_sell_consec: float,
) -> dict:
    """
    Ecuación (fuente de verdad):

    BUY_SCORE  = Wbuy_rsi * RSI_buy + Wbuy_macd * MACD_buy + Wbuy_consec * CONSEC_buy
    SELL_SCORE = Wsell_rsi * RSI_sell + Wsell_macd * MACD_sell + Wsell_consec * CONSEC_sell

    Donde las señales están en rango 0..1 y los pesos se normalizan para sumar 1.
    """
    rsi_val = rsi_arr[i]
    mh = mh_arr[i]
    mh_prev = mh_arr[i - 1] if i > 0 else mh

    buy_rsi = buy_rsi_signal(rsi_val, rsi_oversold, rsi_overbought)
    sell_rsi = sell_rsi_signal(rsi_val, rsi_oversold, rsi_overbought)
    buy_macd = buy_macd_signal(mh, mh_prev, edge_trigger)
    sell_macd = sell_macd_signal(mh, mh_prev, edge_trigger)
    buy_consec = consec_up(close[: i + 1], consec_green)
    sell_consec = consec_down(close[: i + 1], consec_red)

    wbr, wbm, wbc = normalize3(w_buy_rsi, w_buy_macd, w_buy_consec)
    wsr, wsm, wsc = normalize3(w_sell_rsi, w_sell_macd, w_sell_consec)

    buy_score = wbr * buy_rsi + wbm * buy_macd + wbc * buy_consec
    sell_score = wsr * sell_rsi + wsm * sell_macd + wsc * sell_consec

    return {
        "buy_score": buy_score,
        "sell_score": sell_score,
        "signals": {
            "buy_rsi": buy_rsi,
            "buy_macd": buy_macd,
            "buy_consec": buy_consec,
            "sell_rsi": sell_rsi,
            "sell_macd": sell_macd,
            "sell_consec": sell_consec,
        },
    }


def compute_score_snapshot_from_params(params: dict, candles: list, index: Optional[int] = None):
    """
    Calcula scores y señales usando los parámetros del bot (dict).
    """
    use_ha = int(params["use_ha"]) == 1
    edge = int(params["edge_trigger"])

    src = heikin_ashi(candles) if use_ha else candles
    close = [c["close"] for c in src]

    rsi_period = int(params["rsi_period"])
    rsi_vals = rsi(close, rsi_period)
    hist = macd_hist(close, int(params["macd_fast"]), int(params["macd_slow"]), int(params["macd_signal"]))

    idx = len(close) - 1 if index is None else int(index)
    payload = score_components_at_index(
        close,
        rsi_vals,
        hist,
        idx,
        rsi_oversold=float(params["rsi_oversold"]),
        rsi_overbought=float(params["rsi_overbought"]),
        consec_red=int(params["consec_red"]),
        consec_green=int(params["consec_green"]),
        edge_trigger=edge,
        w_buy_rsi=float(params["w_buy_rsi"]),
        w_buy_macd=float(params["w_buy_macd"]),
        w_buy_consec=float(params["w_buy_consec"]),
        w_sell_rsi=float(params["w_sell_rsi"]),
        w_sell_macd=float(params["w_sell_macd"]),
        w_sell_consec=float(params["w_sell_consec"]),
    )
    return payload, close[idx]


def compute_score_snapshot_from_genome(ge: "Genome", candles: list, index: Optional[int] = None):
    """
    Calcula scores y señales usando un Genome (simulador/GA).
    """
    data = heikin_ashi(candles) if ge.use_ha == 1 else candles
    close = [c["close"] for c in data]
    rsi_arr = rsi(close, ge.rsi_period)
    mh_arr = macd_hist(close, ge.macd_fast, ge.macd_slow, ge.macd_signal)
    idx = len(close) - 1 if index is None else int(index)
    payload = score_components_at_index(
        close,
        rsi_arr,
        mh_arr,
        idx,
        rsi_oversold=ge.rsi_oversold,
        rsi_overbought=ge.rsi_overbought,
        consec_red=ge.consec_red,
        consec_green=ge.consec_green,
        edge_trigger=ge.edge_trigger,
        w_buy_rsi=ge.w_buy_rsi,
        w_buy_macd=ge.w_buy_macd,
        w_buy_consec=ge.w_buy_consec,
        w_sell_rsi=ge.w_sell_rsi,
        w_sell_macd=ge.w_sell_macd,
        w_sell_consec=ge.w_sell_consec,
    )
    return payload, close[idx]


def compute_scores(params: dict, candles: list):
    """
    candles: list of dict open/high/low/close
    returns (buyScore, sellScore, last_close)
    """
    payload, last_close = compute_score_snapshot_from_params(params, candles)
    return payload["buy_score"], payload["sell_score"], last_close


def interval_to_ms(interval: str) -> int:
    if interval not in INTERVAL_MS:
        raise ValueError(f"Intervalo no soportado: {interval}")
    return INTERVAL_MS[interval]


def is_candle_closed(close_time_ms: int, server_time_ms: int, interval_ms: int) -> bool:
    """
    Criterio único compartido por bot y simulador.
    """
    return int(close_time_ms) < int(server_time_ms) - int(interval_ms)


def filter_closed_candles(candles: list[dict], interval: str, server_time_ms: int) -> list[dict]:
    interval_ms = interval_to_ms(interval)
    return filter_closed_candles_by_interval_ms(candles, interval_ms, server_time_ms)


def filter_closed_candles_by_interval_ms(candles: list[dict], interval_ms: int, server_time_ms: int) -> list[dict]:
    return [c for c in candles if is_candle_closed(c["close_time"], server_time_ms, interval_ms)]


def assert_simulator_uses_closed_last_price(candles_closed: list[dict], used_close: float):
    if not candles_closed:
        return
    expected = float(candles_closed[-1]["close"])
    if abs(float(used_close) - expected) > 1e-12:
        raise AssertionError(
            f"[VALIDACION] El simulador no usa el último close cerrado. used={used_close} expected={expected}"
        )


def assert_bot_simulator_score_parity(ge: "Genome", candles_closed: list[dict], index: Optional[int] = None):
    if not candles_closed:
        return
    idx = len(candles_closed) - 1 if index is None else int(index)
    ge_payload, _ = compute_score_snapshot_from_genome(ge, candles_closed, idx)
    pa_payload, _ = compute_score_snapshot_from_params(asdict(ge), candles_closed, idx)
    if abs(ge_payload["buy_score"] - pa_payload["buy_score"]) > 1e-12 or abs(ge_payload["sell_score"] - pa_payload["sell_score"]) > 1e-12:
        raise AssertionError(
            "[VALIDACION] Score inconsistente bot vs simulador "
            f"(buy ge={ge_payload['buy_score']} bot={pa_payload['buy_score']} | "
            f"sell ge={ge_payload['sell_score']} bot={pa_payload['sell_score']})"
        )


def infer_interval_ms_from_candles(candles: list[dict]) -> int:
    if len(candles) < 2:
        return INTERVAL_MS.get(INTERVAL, 60_000)
    diffs = []
    for c in candles:
        ot = c.get("open_time")
        ct = c.get("close_time")
        if ot is not None and ct is not None and ct > ot:
            diffs.append(int(ct) - int(ot))
    if diffs:
        return int(statistics.median(diffs))
    for i in range(1, len(candles)):
        prev = candles[i - 1].get("open_time")
        cur = candles[i].get("open_time")
        if prev is not None and cur is not None and cur > prev:
            diffs.append(int(cur) - int(prev))
    if diffs:
        return int(statistics.median(diffs))
    return INTERVAL_MS.get(INTERVAL, 60_000)


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
    min_qty = None
    for f in info["filters"]:
        if f["filterType"] == "PRICE_FILTER":
            tick = float(f["tickSize"])
        elif f["filterType"] == "LOT_SIZE":
            step = float(f["stepSize"])
            min_qty = float(f.get("minQty", 0.0))
        elif f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
            # NOTIONAL aparece en algunos símbolos; usamos minNotional si existe
            min_notional = float(f.get("minNotional", f.get("notional", 0.0)))
    return tick, step, min_notional, min_qty

def get_free_balance(client: Client, asset: str) -> float:
    bal = client.get_asset_balance(asset=asset)
    if not bal:
        return 0.0
    return float(bal["free"])

def get_spot_balances(client: Client) -> list[dict]:
    account = client.get_account()
    balances = []
    for bal in account.get("balances", []):
        free = float(bal.get("free", 0.0))
        locked = float(bal.get("locked", 0.0))
        total = free + locked
        if total > 0:
            balances.append(
                {
                    "asset": bal.get("asset", ""),
                    "free": free,
                    "locked": locked,
                    "total": total,
                }
            )
    balances.sort(key=lambda item: item["asset"])
    return balances

def fetch_klines_closed(client: Client, symbol: str, interval: str, limit: int):
    """
    Trae velas y devuelve SOLO velas cerradas usando el mismo criterio de cierre
    compartido por bot y simulador.
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

    server_time_ms = int(client.get_server_time()["serverTime"])
    return filter_closed_candles(candles, interval, server_time_ms)


# =========================
# Bot core
# =========================

class SpotBot:
    def __init__(self, client: Client, symbol: str, interval: str, params: dict, ui_cb=None):
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.params = params.copy()
        tp_pct, sl_pct = normalize_tp_sl(self.params.get("take_profit", 0.0), self.params.get("stop_loss", 0.0))
        self.params["take_profit"] = tp_pct
        self.params["stop_loss"] = sl_pct
        self.ui_cb = ui_cb  # callback para UI

        self.tick, self.step, self.min_notional, self.min_qty = get_filters(client, symbol)

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
        # [EXCEL EXPORT] incluye entry_price, tp/sl, y PNL con signo
        open_rows = [asdict(t) for t in self.open_trades]
        closed_rows = [asdict(t) for t in self.closed_trades]
        df_open = pd.DataFrame(open_rows)
        df_closed = pd.DataFrame(closed_rows)
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            df_open.to_excel(w, index=False, sheet_name="abiertas")
            df_closed.to_excel(w, index=False, sheet_name="cerradas")

    def _now_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _retry_connection(self) -> bool:
        self._emit("net", {"status": "reconnecting"})
        cycle = 1
        while self.running:
            for attempt in range(1, RECONNECT_MAX_RETRIES + 1):
                if not self.running:
                    self._emit("net", {"status": "offline"})
                    return False
                try:
                    self.client.ping()
                    self._emit("net", {"status": "online"})
                    self._emit(
                        "log",
                        {
                            "msg": (
                                f"[NET] Reconexion OK (ciclo {cycle}, intento "
                                f"{attempt}/{RECONNECT_MAX_RETRIES})"
                            )
                        },
                    )
                    return True
                except Exception as e:
                    wait_s = RECONNECT_BACKOFF_SEC * attempt
                    self._emit(
                        "log",
                        {
                            "msg": (
                                f"[NET] Reintentando conexion (ciclo {cycle}) "
                                f"{attempt}/{RECONNECT_MAX_RETRIES} en {wait_s:.1f}s: {e}"
                            )
                        },
                    )
                    time.sleep(wait_s)
            cycle += 1
        self._emit("net", {"status": "offline"})
        return False

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

    def _min_quote_for_notional(self, price: float) -> Optional[float]:
        if not self.min_notional or not price or price <= 0:
            return None
        step = self.step if self.step else 0.0
        if step <= 0:
            return self.min_notional
        qty_min = math.ceil((self.min_notional / price) / step) * step
        return qty_min * price

    def _place_market_buy_by_quote(self, quote_amount: float, reason: str):
        """
        Compra gastando quote_amount USDT (aprox).
        En SPOT Binance permite quoteOrderQty.
        Verificación: gasto real vs objetivo.
        """
        if quote_amount <= 0:
            return None

        if DRY_RUN:
            # Simulación simple: asumimos ejecución a precio actual
            price = float(self.client.get_symbol_ticker(symbol=self.symbol)["price"])
            qty = quote_amount / price
            qty = round_step(qty, self.step)
            if qty <= 0:
                self._emit("log", {"msg": "[BUY] qty calculada <= 0 en simulación"})
                return None
            if self.min_notional and quote_amount < self.min_notional:
                self._emit(
                    "log",
                    {
                        "msg": f"[BUY] (DRY_RUN) quote {quote_amount:.4f} < minNotional {self.min_notional:.4f} -> simulo igual"
                    },
                )
            if self.min_qty and qty < self.min_qty:
                self._emit(
                    "log",
                    {
                        "msg": f"[BUY] (DRY_RUN) qty {qty:.8f} < minQty {self.min_qty:.8f} -> simulo igual"
                    },
                )
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

        price = None
        try:
            price = float(self.client.get_symbol_ticker(symbol=self.symbol)["price"])
        except Exception as e:
            self._emit("log", {"msg": f"[AVISO] No pude leer precio para validar compra: {e}"})

        if self.min_notional:
            min_quote = self._min_quote_for_notional(price) if price else self.min_notional
            min_quote = min_quote + BUY_NOTIONAL_BUFFER_USDT if min_quote else self.min_notional
            if quote_amount < min_quote:
                self._emit(
                    "log",
                    {
                        "msg": (
                            f"[BUY] quote {quote_amount:.4f} < minNotional+buffer {min_quote:.4f} -> no compro"
                        )
                    },
                )
                return None

        if self.min_qty and price:
            est_qty = quote_amount / price
            if est_qty < self.min_qty:
                self._emit(
                    "log",
                    {
                        "msg": f"[BUY] qty estimada {est_qty:.8f} < minQty {self.min_qty:.8f} -> no compro"
                    },
                )
                return None

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
            self._emit("log", {"msg": f"[ERROR] Error de Binance al comprar: {e}"})
            return None

    def _place_market_sell_qty(self, qty: float, reason: str):
        qty = round_step(qty, self.step)
        if qty <= 0:
            return None

        est_notional = None
        try:
            price = float(self.client.get_symbol_ticker(symbol=self.symbol)["price"])
            est_notional = price * qty
        except Exception as e:
            self._emit("log", {"msg": f"[AVISO] No pude estimar notional para venta: {e}"})

        if DRY_RUN:
            if est_notional is not None and self.min_notional:
                if est_notional < (self.min_notional + SELL_NOTIONAL_BUFFER_USDT):
                    self._emit(
                        "log",
                        {
                            "msg": (
                                "[SELL] (DRY_RUN) notional bajo "
                                f"{est_notional:.4f} < {self.min_notional + SELL_NOTIONAL_BUFFER_USDT:.4f} -> simulo igual"
                            )
                        },
                    )

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

        if est_notional is not None and self.min_notional:
            min_required = self.min_notional + SELL_NOTIONAL_BUFFER_USDT
            if est_notional < min_required:
                self._emit(
                    "log",
                    {
                        "msg": (
                            f"[SELL] notional {est_notional:.4f} < {min_required:.4f} "
                            f"(minNotional+buffer) -> no vendo (dejo dust)"
                        )
                    },
                )
                return None

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
            self._emit("log", {"msg": f"[ERROR] Error de Binance al vender: {e}"})
            return None

    def manual_buy_by_quote_pct(self, pct: float) -> Optional[Trade]:
        usdt_free = get_free_balance(self.client, QUOTE_ASSET)
        if usdt_free <= 0:
            self._emit("log", {"msg": "[MANUAL BUY] USDT libre = 0, no compro"})
            return None
        target = usdt_free * (pct / 100.0)
        target = float(target)
        if target <= 0:
            self._emit("log", {"msg": "[MANUAL BUY] Porcentaje inválido, no compro"})
            return None
        if not DRY_RUN and self.min_notional:
            price = None
            try:
                price = float(self.client.get_symbol_ticker(symbol=self.symbol)["price"])
            except Exception as e:
                self._emit("log", {"msg": f"[AVISO] No pude leer precio para compra manual: {e}"})
            min_quote = self._min_quote_for_notional(price) if price else self.min_notional
            min_quote = min_quote + BUY_NOTIONAL_BUFFER_USDT if min_quote else self.min_notional
            if target < min_quote:
                if usdt_free >= min_quote:
                    self._emit(
                        "log",
                        {
                            "msg": f"[MANUAL BUY] % muy bajo para minNotional, ajusto a {min_quote:.4f} USDT"
                        },
                    )
                    target = min_quote
                else:
                    self._emit(
                        "log",
                        {
                            "msg": f"[MANUAL BUY] USDT libre {usdt_free:.4f} < minNotional {min_quote:.4f}"
                        },
                    )
                    return None
            else:
                pass
        return self._place_market_buy_by_quote(target, reason="MANUAL_BUY")

    def manual_sell_all_base(self) -> Optional[Trade]:
        sol_free = get_free_balance(self.client, BASE_ASSET)
        qty = round_step(sol_free, self.step)
        if qty <= 0:
            self._emit("log", {"msg": "[MANUAL SELL] No hay SOL libre para vender"})
            return None
        return self._place_market_sell_qty(qty, reason="MANUAL_SELL")

    def _open_position_from_trade(self, buy_trade: Trade, last_close_time_ms: int):
        ep = buy_trade.price
        qty = buy_trade.qty
        tp_pct, sl_pct = normalize_tp_sl(self.params["take_profit"], self.params["stop_loss"])
        # [TP/SL PRICE CALCULATION] usando magnitudes abs
        tp = ep * (1.0 + tp_pct) if tp_pct > 0 else None
        sl = ep * (1.0 - sl_pct) if sl_pct > 0 else None
        self.position = Position(
            symbol=self.symbol,
            qty=qty,
            entry_price=ep,
            entry_quote_spent=buy_trade.quote_spent,
            entry_time_utc=buy_trade.time_utc,
            tp_price=tp,
            sl_price=sl,
            tp_pct=tp_pct,
            sl_pct=sl_pct
        )
        self.last_action_bar_close_time = last_close_time_ms
        buy_trade.entry_price = ep
        buy_trade.tp_pct = tp_pct
        buy_trade.sl_pct = sl_pct
        buy_trade.tp_price = tp or 0.0
        buy_trade.sl_price = sl or 0.0
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
            # [P&L WITH SIGN] resultado real con signo
            pnl_pct_real = (sell_trade.price - self.position.entry_price) / (self.position.entry_price + 1e-12)
            pnl_usdt_real = sell_trade.quote_spent - self.position.entry_quote_spent
            sell_trade.entry_price = self.position.entry_price
            sell_trade.tp_pct = self.position.tp_pct
            sell_trade.sl_pct = self.position.sl_pct
            sell_trade.tp_price = self.position.tp_price or 0.0
            sell_trade.sl_price = self.position.sl_price or 0.0
            sell_trade.pnl_pct_real = pnl_pct_real
            sell_trade.pnl_usdt_real = pnl_usdt_real
            self.closed_trades.append(sell_trade)
            self._append_journal(sell_trade)

        self.position = None
        self.last_action_bar_close_time = last_close_time_ms
        self._emit("position", {"pos": None})

    def _loop(self):
        self._emit("log", {"msg": f"[BOT] Inicio {self.symbol} {self.interval} | DRY_RUN={DRY_RUN} | tick={self.tick} step={self.step} minNot={self.min_notional} minQty={self.min_qty}"})

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
                bScore, sScore, score_close = compute_scores(self.params, candles)
                last_close = score_close
                close_time_utc = datetime.fromtimestamp(close_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                self._emit("log", {"msg": f"[VELA] Nueva vela cerrada {close_time_utc} close={last_close:.4f}"})

                in_pos = self.position is not None
                cool_ok = self._cooldown_ok(close_time)

                buy_cond  = (not in_pos) and cool_ok and (bScore >= float(self.params["buy_th"]))
                sell_cond = (in_pos) and cool_ok and (sScore >= float(self.params["sell_th"]))

                # TP/SL (si hay posición)
                tp_hit = sl_hit = False
                if in_pos:
                    if self.position.tp_price is not None and last_close >= self.position.tp_price:
                        tp_hit = True
                    if self.position.sl_price is not None and last_close <= self.position.sl_price:
                        sl_hit = True

                spot_price = last_close
                try:
                    spot_price = float(self.client.get_symbol_ticker(symbol=self.symbol)["price"])
                except Exception as e:
                    self._emit("log", {"msg": f"[AVISO] No pude leer precio actual: {e}"})

                wallet = None
                try:
                    wallet = get_spot_balances(self.client)
                except Exception as e:
                    self._emit("log", {"msg": f"[AVISO] No pude leer billetera spot: {e}"})

                recent_candles = [
                    {
                        "time_utc": datetime.fromtimestamp(c["close_time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        "close": c["close"],
                    }
                    for c in candles[-5:]
                ]

                self._emit("tick", {
                    "time_utc": self._now_utc(),
                    "close": last_close,
                    "spot_price": spot_price,
                    "buyScore": bScore,
                    "sellScore": sScore,
                    "buyCond": buy_cond,
                    "sellCond": sell_cond,
                    "tp_hit": tp_hit,
                    "sl_hit": sl_hit,
                    "wallet": wallet,
                    "ops_count": len(self.closed_trades) + len(self.open_trades),
                    "recent_candles": recent_candles,
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
                        self._emit("log", {"msg": f"[BUY] intento {attempt} gastado={spent_total:.4f}/{target:.4f} ({fill_ratio*100:.1f}%) qty={t.qty:.6f} avg={t.price:.4f}"})

                        if fill_ratio >= MIN_FILL_RATIO:
                            break

                        time.sleep(RETRY_SLEEP_SEC)

                    if last_trade:
                        if spent_total >= (MIN_FILL_RATIO * target):
                            # Consolidar posición por “precio” usando último trade (simple). En real podrías promediar.
                            self._open_position_from_trade(last_trade, close_time)
                        else:
                            self._emit("log", {"msg": f"[BUY] Fill insuficiente: spent={spent_total:.4f} < {MIN_FILL_RATIO*100:.0f}% target={target:.4f}. No abro posición."})

                # Cierre por TP/SL primero
                if in_pos and tp_hit:
                    self._emit("log", {"msg": "[TP] Take Profit alcanzado -> cierro"})
                    self._close_position(reason="TP", last_close_time_ms=close_time)
                    continue
                if in_pos and sl_hit:
                    self._emit("log", {"msg": "[SL] Stop Loss alcanzado -> cierro"})
                    self._close_position(reason="SL", last_close_time_ms=close_time)
                    continue

                # Cierre por señal
                if in_pos and sell_cond:
                    self._emit("log", {"msg": "[SELL] Señal de cierre -> cierro"})
                    self._close_position(reason="SIGNAL_SELL", last_close_time_ms=close_time)
                    continue

            except Exception as e:
                self._emit("log", {"msg": f"[ERROR] {e}"})
                if self.client and self._retry_connection():
                    continue
                time.sleep(2)

        self._emit("log", {"msg": "[BOT] Detenido"})


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
        self.var_spot_price = tk.StringVar(value="spot=-")
        self.var_flags = tk.StringVar(value="buyCond=- sellCond=- tp=- sl=-")
        self.var_pos = tk.StringVar(value="pos: NONE")
        self.var_wallet_summary = tk.StringVar(value="wallet: -")
        self.var_ops = tk.StringVar(value="ops: -")
        self.var_recent_candles = tk.StringVar(value="velas: -")
        self.var_manual_pct = tk.DoubleVar(value=10.0)
        self.var_connection = tk.StringVar(value="offline")
        self._connect_stop = threading.Event()
        self._connect_thread: Optional[threading.Thread] = None

        self._build()
        self._start_auto_connect()

    def _build(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)

        bot_tab = ttk.Frame(nb, padding=10)
        wallet_tab = ttk.Frame(nb, padding=10)
        manual_tab = ttk.Frame(nb, padding=10)
        nb.add(bot_tab, text="Bot")
        nb.add(wallet_tab, text="Billetera SPOT")
        nb.add(manual_tab, text="Manual")

        frm = bot_tab

        top = ttk.Frame(frm)
        top.pack(fill="x")

        ttk.Label(top, text="Estado:").grid(row=0, column=0, sticky="w")
        ttk.Label(top, textvariable=self.var_status).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(top, text="Conexión:").grid(row=1, column=0, sticky="w")
        ttk.Label(top, textvariable=self.var_connection).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(top, text="Último:").grid(row=2, column=0, sticky="w")
        ttk.Label(top, textvariable=self.var_last).grid(row=2, column=1, sticky="w", padx=5)

        ttk.Label(top, textvariable=self.var_price).grid(row=0, column=2, sticky="w", padx=10)
        ttk.Label(top, textvariable=self.var_scores).grid(row=1, column=2, sticky="w", padx=10)
        ttk.Label(top, textvariable=self.var_flags).grid(row=2, column=2, sticky="w", padx=10)
        ttk.Label(top, textvariable=self.var_spot_price).grid(row=0, column=3, sticky="w", padx=10)
        ttk.Label(top, textvariable=self.var_wallet_summary).grid(row=1, column=3, sticky="w", padx=10)
        ttk.Label(top, textvariable=self.var_ops).grid(row=2, column=3, sticky="w", padx=10)

        ttk.Label(top, textvariable=self.var_pos).grid(row=3, column=0, columnspan=2, sticky="w", pady=5)

        ttk.Label(top, textvariable=self.var_recent_candles).grid(row=4, column=0, columnspan=4, sticky="w", pady=5)

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

        cols = (
            "time_utc",
            "side",
            "qty",
            "price",
            "quote_spent",
            "reason",
            "entry_price",
            "tp_pct",
            "sl_pct",
            "tp_price",
            "sl_price",
            "pnl_pct_real",
            "pnl_usdt_real",
        )
        self.tree = ttk.Treeview(tfrm, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="w")
        self.tree.pack(fill="both", expand=True)

        # Log
        lfrm = ttk.LabelFrame(frm, text="Log")
        lfrm.pack(fill="both", expand=True)

        log_frame = ttk.Frame(lfrm)
        log_frame.pack(fill="both", expand=True)
        self.txt_log = tk.Text(log_frame, height=8, width=90, wrap="word")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=log_scroll.set)
        self.txt_log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        wallet_controls = ttk.Frame(wallet_tab)
        wallet_controls.pack(fill="x")

        ttk.Button(wallet_controls, text="Actualizar billetera", command=self.refresh_wallet).pack(side="left")

        self.wallet_tree = ttk.Treeview(wallet_tab, columns=("asset", "free", "locked", "total"), show="headings", height=16)
        for col in ("asset", "free", "locked", "total"):
            self.wallet_tree.heading(col, text=col)
            self.wallet_tree.column(col, width=120, anchor="center")
        self.wallet_tree.pack(fill="both", expand=True, pady=8)

        manual_info = ttk.LabelFrame(manual_tab, text="Operaciones manuales (spot)")
        manual_info.pack(fill="x", pady=8)

        pct_row = ttk.Frame(manual_info)
        pct_row.pack(fill="x", padx=6, pady=6)
        ttk.Label(pct_row, text="Porcentaje USDT libre (%):").pack(side="left")
        ttk.Entry(pct_row, textvariable=self.var_manual_pct, width=8).pack(side="left", padx=6)
        ttk.Button(pct_row, text="Comprar % USDT (market)", command=self.manual_buy_pct).pack(side="left", padx=6)

        sell_row = ttk.Frame(manual_info)
        sell_row.pack(fill="x", padx=6, pady=6)
        ttk.Button(sell_row, text="Vender TODO SOL (market)", command=self.manual_sell_all).pack(side="left")

        manual_hint = ttk.Label(
            manual_tab,
            text="Nota: usa DRY_RUN=False para órdenes reales. Las operaciones manuales usan el mismo cliente SPOT.",
        )
        manual_hint.pack(fill="x", pady=6)

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
                self.var_spot_price.set(f"spot={payload.get('spot_price', payload['close']):.4f}")
                self.var_scores.set(f"bScore={payload['buyScore']:.3f} sScore={payload['sellScore']:.3f}")
                self.var_flags.set(f"buyCond={payload['buyCond']} sellCond={payload['sellCond']} tp={payload['tp_hit']} sl={payload['sl_hit']}")
                self.var_ops.set(f"ops={payload.get('ops_count', 0)}")
                recent = payload.get("recent_candles", [])
                if recent:
                    candles_txt = " | ".join(f"{c['time_utc']}={c['close']:.4f}" for c in recent)
                    self.var_recent_candles.set(f"velas: {candles_txt}")
                wallet = payload.get("wallet")
                if wallet is not None:
                    self._update_wallet_table(wallet)
                    summary = ", ".join(
                        f"{item['asset']} {item['total']:.6f}" for item in wallet if item["total"] > 0
                    )
                    self.var_wallet_summary.set(f"wallet: {summary}" if summary else "wallet: -")
            elif kind == "position":
                if payload["pos"] is None:
                    self.var_pos.set("pos: NONE")
                else:
                    pos = payload["pos"]
                    tp_disp = f"{pos['tp_price']:.4f}" if pos["tp_price"] is not None else "OFF"
                    sl_disp = f"{pos['sl_price']:.4f}" if pos["sl_price"] is not None else "OFF"
                    self.var_pos.set(
                        f"pos: qty={pos['qty']:.6f} entry={pos['entry_price']:.4f} "
                        f"TP={pos['tp_pct']*100:.2f}%({tp_disp}) SL={pos['sl_pct']*100:.2f}%({sl_disp})"
                    )
            elif kind == "net":
                self.var_connection.set(payload["status"])
            # refresh table
            if self.bot:
                # render last 200 closed trades
                self.tree.delete(*self.tree.get_children())
                for tr in self.bot.closed_trades[-200:]:
                    self.tree.insert(
                        "",
                        "end",
                        values=(
                            tr.time_utc,
                            tr.side,
                            f"{tr.qty:.6f}",
                            f"{tr.price:.4f}",
                            f"{tr.quote_spent:.4f}",
                            tr.reason,
                            f"{tr.entry_price:.4f}",
                            f"{tr.tp_pct*100:.2f}%",
                            f"{tr.sl_pct*100:.2f}%",
                            f"{tr.tp_price:.4f}",
                            f"{tr.sl_price:.4f}",
                            f"{tr.pnl_pct_real:+.4f}",
                            f"{tr.pnl_usdt_real:+.4f}",
                        ),
                    )

        self.root.after(0, _upd)

    def _update_wallet_table(self, wallet: list[dict]):
        if not hasattr(self, "wallet_tree"):
            return
        self.wallet_tree.delete(*self.wallet_tree.get_children())
        for item in wallet:
            self.wallet_tree.insert(
                "",
                "end",
                values=(
                    item["asset"],
                    f"{item['free']:.6f}",
                    f"{item['locked']:.6f}",
                    f"{item['total']:.6f}",
                ),
            )

    def refresh_wallet(self):
        if not self.client:
            messagebox.showwarning("Aviso", "Primero conecta.")
            return
        try:
            wallet = get_spot_balances(self.client)
            self._update_wallet_table(wallet)
            summary = ", ".join(
                f"{item['asset']} {item['total']:.6f}" for item in wallet if item["total"] > 0
            )
            self.var_wallet_summary.set(f"wallet: {summary}" if summary else "wallet: -")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def manual_buy_pct(self):
        if not self.client:
            messagebox.showwarning("Aviso", "Primero conecta.")
            return
        try:
            pct = float(self.var_manual_pct.get())
        except (TypeError, ValueError):
            messagebox.showwarning("Aviso", "Porcentaje inválido.")
            return
        if pct <= 0:
            messagebox.showwarning("Aviso", "Porcentaje inválido.")
            return
        if not self.bot:
            try:
                self.bot = SpotBot(self.client, SYMBOL, INTERVAL, self.params, ui_cb=self.ui_callback)
            except ValueError as e:
                messagebox.showerror("Parámetros", str(e))
                return
        trade = self.bot.manual_buy_by_quote_pct(pct)
        if trade:
            self.log(f"[MANUAL BUY] qty={trade.qty:.6f} price={trade.price:.4f} spent={trade.quote_spent:.4f}")
            self.refresh_wallet()
        else:
            self.log("[MANUAL BUY] No se pudo ejecutar la compra. Revisa minNotional/minQty y saldo USDT.")

    def manual_sell_all(self):
        if not self.client:
            messagebox.showwarning("Aviso", "Primero conecta.")
            return
        if not self.bot:
            try:
                self.bot = SpotBot(self.client, SYMBOL, INTERVAL, self.params, ui_cb=self.ui_callback)
            except ValueError as e:
                messagebox.showerror("Parámetros", str(e))
                return
        trade = self.bot.manual_sell_all_base()
        if trade:
            self.log(f"[MANUAL SELL] qty={trade.qty:.6f} price={trade.price:.4f} got={trade.quote_spent:.4f}")
            self.refresh_wallet()
        else:
            self.log("[MANUAL SELL] No se pudo ejecutar la venta. Revisa saldo SOL disponible.")

    def _start_auto_connect(self):
        if self._connect_thread and self._connect_thread.is_alive():
            return
        self._connect_stop.clear()
        self._connect_thread = threading.Thread(target=self._connect_loop, daemon=True)
        self._connect_thread.start()

    def _connect_loop(self):
        while not self._connect_stop.is_set():
            if self.client is None:
                if self.connect(show_errors=False):
                    if not self.bot or not self.bot.running:
                        self.start()
            time.sleep(GUI_CONNECT_RETRY_SEC)

    def connect(self, *, show_errors: bool = True) -> bool:
        try:
            key = read_key(API_KEY_PATH)
            sec = read_key(API_SECRET_PATH)
            self.client = Client(
                key,
                sec,
                requests_params={"timeout": BINANCE_TIMEOUT_SEC},
            )
            self.client.ping()
            self.log("[OK] Conectado a Binance SPOT (ping ok).")
            self.var_connection.set("online")
            self.refresh_wallet()
            return True
        except Exception as e:
            self.client = None
            self.var_connection.set("offline")
            if show_errors:
                messagebox.showerror("Error", str(e))
            else:
                self.log(f"[NET] Error de conexion: {e}")
            return False

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
        if self.bot and self.bot.running:
            self.var_status.set("RUN")
            self.log("[BOT] Ya estaba iniciado.")
            return

        # leer params desde caja (por si editaste)
        try:
            txt = self.txt_params.get("1.0", "end").strip()
            self.params = json.loads(txt)
        except Exception as e:
            messagebox.showerror("Parámetros", f"JSON inválido: {e}")
            return

        try:
            self.bot = SpotBot(self.client, SYMBOL, INTERVAL, self.params, ui_cb=self.ui_callback)
        except ValueError as e:
            messagebox.showerror("Parámetros", str(e))
            return
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

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def tv_parse_date_ms(date_str: str, *, end: bool = False) -> int:
    base = datetime.strptime(date_str, "%Y-%m-%d")
    if end:
        base = base.replace(hour=23, minute=59, second=59, microsecond=999000)
    return int(base.replace(tzinfo=timezone.utc).timestamp() * 1000)

def tv_build_json_path(symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
    safe_symbol = symbol.upper()
    filename = f"{safe_symbol}_{timeframe}_{start_date}__{end_date}.json"
    return os.path.join(OUT_DIR, filename)

def fetch_ohlcv_by_date_range(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    exchange: str = "binance",
):
    if timeframe not in INTERVAL_MS:
        raise ValueError(f"Intervalo no soportado: {timeframe}")
    if exchange.lower() != "binance":
        raise ValueError(f"Exchange no soportado: {exchange}")

    tv_start_ms = tv_parse_date_ms(start_date)
    tv_end_ms = tv_parse_date_ms(end_date, end=True)
    tv_interval_ms = INTERVAL_MS[timeframe]

    out = []
    cursor = tv_start_ms
    while cursor <= tv_end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "limit": 1000,
            "startTime": cursor,
            "endTime": tv_end_ms,
        }

        data = None
        last_err = None
        for attempt in range(1, BINANCE_MAX_RETRIES + 1):
            try:
                r = requests.get(
                    BINANCE_BASE + "/api/v3/klines",
                    params=params,
                    timeout=BINANCE_TIMEOUT_SEC,
                )
                r.raise_for_status()
                data = r.json()
                break
            except requests.exceptions.RequestException as exc:
                last_err = exc
                if attempt == BINANCE_MAX_RETRIES:
                    raise
                time.sleep(BINANCE_RETRY_BACKOFF_SEC * attempt)

        if data is None:
            if last_err:
                raise last_err
            raise RuntimeError("Sin respuesta de Binance en klines.")
        if not data:
            break

        for k in data:
            ts = int(k[0])
            if ts > tv_end_ms:
                break
            out.append({
                "datetime": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                "timestamp_ms": ts,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        cursor = int(data[-1][0]) + tv_interval_ms
        time.sleep(0.12)

    tv_df = pd.DataFrame(out)
    if not tv_df.empty:
        tv_df = tv_df.sort_values("timestamp_ms").reset_index(drop=True)
    return tv_df

def save_candles_json(df, symbol: str, timeframe: str, start_date: str, end_date: str, exchange: str):
    ensure_out_dir()
    tv_path = tv_build_json_path(symbol, timeframe, start_date, end_date)
    tv_records = df.to_dict("records")
    for rec in tv_records:
        rec["symbol"] = symbol.upper()
        rec["timeframe"] = timeframe
        rec["exchange"] = exchange
    with open(tv_path, "w", encoding="utf-8") as f:
        json.dump(tv_records, f, indent=2, ensure_ascii=False)
    return tv_path

def load_candles_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "candles" in payload:
        payload = payload["candles"]
    tv_df = pd.DataFrame(payload)
    if not tv_df.empty:
        tv_df = tv_df.sort_values("timestamp_ms").reset_index(drop=True)
    return tv_df

def tv_indicator_cache_key(params: dict) -> str:
    keys = [
        "use_ha",
        "rsi_period",
        "rsi_oversold",
        "rsi_overbought",
        "macd_fast",
        "macd_slow",
        "macd_signal",
        "consec_red",
        "consec_green",
        "w_buy_rsi",
        "w_buy_macd",
        "w_buy_consec",
        "w_sell_rsi",
        "w_sell_macd",
        "w_sell_consec",
        "buy_th",
        "sell_th",
    ]
    payload = {k: params.get(k) for k in keys}
    return json.dumps(payload, sort_keys=True)

def tv_apply_indicator_cache(tv_df: pd.DataFrame, params: dict):
    if tv_df.empty:
        return tv_df
    use_ha = int(params.get("use_ha", 0)) == 1
    if use_ha:
        ha_candles = heikin_ashi(tv_df.to_dict("records"))
        ha_close = [c["close"] for c in ha_candles]
        tv_df["ha_close"] = ha_close
        src_close = ha_close
    else:
        src_close = tv_df["close"].tolist()

    tv_df["rsi"] = rsi_wilder_series(src_close, int(params.get("rsi_period", 14)))
    tv_df["macd_hist"] = macd_hist_series(
        src_close,
        int(params.get("macd_fast", 12)),
        int(params.get("macd_slow", 26)),
        int(params.get("macd_signal", 9)),
    )
    rsi_os = float(params.get("rsi_oversold", 30))
    rsi_ob = float(params.get("rsi_overbought", 70))
    wbr, wbm, wbc = normalize3(
        float(params.get("w_buy_rsi", 1.0)),
        float(params.get("w_buy_macd", 1.0)),
        float(params.get("w_buy_consec", 1.0)),
    )
    wsr, wsm, wsc = normalize3(
        float(params.get("w_sell_rsi", 1.0)),
        float(params.get("w_sell_macd", 1.0)),
        float(params.get("w_sell_consec", 1.0)),
    )
    prev_close = tv_df["close"].shift(1)
    rsi_buy = (tv_df["rsi"] <= rsi_os).astype(float)
    rsi_sell = (tv_df["rsi"] >= rsi_ob).astype(float)
    macd_buy = (tv_df["macd_hist"] > 0).astype(float)
    macd_sell = (tv_df["macd_hist"] < 0).astype(float)
    consec_buy = (tv_df["close"] <= prev_close).astype(float)
    consec_sell = (tv_df["close"] >= prev_close).astype(float)
    tv_df["buy_score"] = wbr * rsi_buy + wbm * macd_buy + wbc * consec_buy
    tv_df["sell_score"] = wsr * rsi_sell + wsm * macd_sell + wsc * consec_sell
    cache_key = tv_indicator_cache_key(params)
    tv_df["cache_key"] = cache_key
    return tv_df

def tv_df_to_candles(tv_df: pd.DataFrame, timeframe: str):
    interval_ms = INTERVAL_MS[timeframe]
    candles = []
    for rec in tv_df.to_dict("records"):
        ts = int(rec["timestamp_ms"])
        candles.append(
            {
                "t": ts,
                "open_time": ts,
                "close_time": ts + interval_ms,
                "open": float(rec["open"]),
                "high": float(rec["high"]),
                "low": float(rec["low"]),
                "close": float(rec["close"]),
                "v": float(rec.get("volume", rec.get("v", 0.0))),
                "rsi": rec.get("rsi"),
                "macd_hist": rec.get("macd_hist"),
                "ha_close": rec.get("ha_close"),
                "cache_key": rec.get("cache_key"),
            }
        )
    return candles


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

        data = None
        last_err = None
        for attempt in range(1, BINANCE_MAX_RETRIES + 1):
            try:
                r = requests.get(
                    BINANCE_BASE + "/api/v3/klines",
                    params=params,
                    timeout=BINANCE_TIMEOUT_SEC,
                )
                r.raise_for_status()
                data = r.json()
                break
            except requests.exceptions.RequestException as exc:
                last_err = exc
                if attempt == BINANCE_MAX_RETRIES:
                    raise
                time.sleep(BINANCE_RETRY_BACKOFF_SEC * attempt)

        if data is None:
            if last_err:
                raise last_err
            raise RuntimeError("Sin respuesta de Binance en klines.")
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

        apply_weight_constraints(g)
        if g["macd_slow"] <= g["macd_fast"]:
            g["macd_slow"] = min(self.spec["macd_slow"]["max"], g["macd_fast"] + 5)
        return Genome(**g)

    def clamp_genome(self, ge: Genome):
        d = asdict(ge)
        for k in d:
            d[k] = self.quantize(k, d[k])

        apply_weight_constraints(d)
        if d["macd_slow"] <= d["macd_fast"]:
            d["macd_slow"] = min(self.spec["macd_slow"]["max"], d["macd_fast"] + 5)
        return Genome(**d)


@dataclass
class Metrics:
    score: float
    net: float
    balance: float
    profit: float
    gross_profit: float
    gross_loss: float
    max_dd: float
    dd_pct: float
    trades: int
    wins: int
    losses: int
    winrate: float
    pf: float
    years: float
    trades_per_year: float
    buy_signal_rate: float
    sell_signal_rate: float


def simulate_spot(candles, ge: Genome, fee_per_side: float, slip_per_side: float, trace: bool = False):
    interval_ms = infer_interval_ms_from_candles(candles)
    server_time_ms = max(c.get("close_time", 0) for c in candles) + interval_ms + 1 if candles else 0
    candles = filter_closed_candles_by_interval_ms(candles, interval_ms, server_time_ms)

    if len(candles) < 200:
        metrics = Metrics(
            score=-1e9,
            net=-1.0,
            balance=0.0,
            profit=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            max_dd=1.0,
            dd_pct=100.0,
            trades=0,
            wins=0,
            losses=0,
            winrate=0.0,
            pf=0.0,
            years=0.0,
            trades_per_year=0.0,
            buy_signal_rate=0.0,
            sell_signal_rate=0.0,
        )
        return (metrics, {"events": [], "equity_curve": [], "dd_curve": [], "returns": []}) if trace else metrics

    cache_key = tv_indicator_cache_key(asdict(ge))
    use_cached = all(
        ("cache_key" in c and c.get("cache_key") == cache_key and c.get("rsi") is not None and c.get("macd_hist") is not None)
        for c in candles
    )
    if use_cached:
        if ge.use_ha == 1:
            close_sig = [
                c.get("ha_close") if c.get("ha_close") is not None else c["close"]
                for c in candles
            ]
        else:
            close_sig = [c["close"] for c in candles]
        rsi_arr = [float(c["rsi"]) for c in candles]
        mh_arr = [float(c["macd_hist"]) for c in candles]
    else:
        data = heikin_ashi(candles) if ge.use_ha == 1 else candles
        close_sig = [c["close"] for c in data]
        rsi_arr = rsi_wilder_series(close_sig, ge.rsi_period)
        mh_arr = macd_hist_series(close_sig, ge.macd_fast, ge.macd_slow, ge.macd_signal)

    equity = 1.0
    peak = equity
    max_dd = 0.0

    in_pos = False
    entry = 0.0
    entry_i = -10_000
    last_trade_i = -10_000
    entry_buy_score = 0.0
    entry_rsi = 0.0
    entry_macd = 0.0
    entry_close = 0.0

    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    trades = 0

    # [TP/SL ABS NORMALIZATION] magnitudes only (no signo)
    tp_pct = abs(float(ge.take_profit))
    sl_pct = abs(float(ge.stop_loss))
    if tp_pct >= 1.0 or sl_pct >= 1.0:
        metrics = Metrics(
            score=-1e9,
            net=-1.0,
            balance=0.0,
            profit=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            max_dd=1.0,
            dd_pct=100.0,
            trades=0,
            wins=0,
            losses=0,
            winrate=0.0,
            pf=0.0,
            years=0.0,
            trades_per_year=0.0,
            buy_signal_rate=0.0,
            sell_signal_rate=0.0,
        )
        return (metrics, {"events": [], "equity_curve": [], "dd_curve": [], "returns": []}) if trace else metrics

    events = []
    equity_curve = []
    dd_curve = []
    returns = []
    buy_signal_hits = 0
    sell_signal_hits = 0

    for i in range(1, len(candles) - 1):
        peak = max(peak, equity)
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

        payload = score_components_at_index(
            close_sig,
            rsi_arr,
            mh_arr,
            i,
            rsi_oversold=ge.rsi_oversold,
            rsi_overbought=ge.rsi_overbought,
            consec_red=ge.consec_red,
            consec_green=ge.consec_green,
            edge_trigger=ge.edge_trigger,
            w_buy_rsi=ge.w_buy_rsi,
            w_buy_macd=ge.w_buy_macd,
            w_buy_consec=ge.w_buy_consec,
            w_sell_rsi=ge.w_sell_rsi,
            w_sell_macd=ge.w_sell_macd,
            w_sell_consec=ge.w_sell_consec,
        )
        bs = payload["buy_score"]
        ss = payload["sell_score"]
        rsi_val = rsi_arr[i]
        mh_val = mh_arr[i]
        if bs >= ge.buy_th:
            buy_signal_hits += 1
        if ss >= ge.sell_th:
            sell_signal_hits += 1

        if i - last_trade_i < int(ge.cooldown):
            continue

        next_candle = candles[i + 1]
        exec_open = next_candle["open"]
        if not in_pos:
            if bs >= ge.buy_th:
                buy_price = exec_open * (1.0 + slip_per_side)
                equity *= (1.0 - fee_per_side)
                entry = buy_price
                in_pos = True
                entry_i = i + 1
                last_trade_i = i + 1
                entry_buy_score = bs
                entry_rsi = rsi_val
                entry_macd = mh_val
                entry_close = close_sig[i]
        else:
            if i + 1 <= entry_i:
                continue

            # [TP/SL PRICE CALCULATION] usando magnitudes abs
            tp_px = entry * (1.0 + tp_pct) if tp_pct > 0 else None
            sl_px = entry * (1.0 - sl_pct) if sl_pct > 0 else None

            exit_reason = None
            if sl_px is not None and next_candle["low"] <= sl_px:
                exit_reason = "SL"
                px = sl_px
            elif tp_px is not None and next_candle["high"] >= tp_px:
                exit_reason = "TP"
                px = tp_px
            elif ss >= ge.sell_th:
                exit_reason = "SIG"
                px = exec_open

            if exit_reason is not None:
                sell_price = px * (1.0 - slip_per_side)
                gross_ratio = (sell_price / (entry + 1e-12))
                trade_ratio = (1.0 - fee_per_side) * gross_ratio
                trade_ret = trade_ratio - 1.0

                equity *= trade_ratio
                if trace:
                    candle_time = next_candle.get("open_time", next_candle.get("t"))
                    events.append({
                        "i": i + 1,
                        "time": candle_time,
                        "side": "SELL",
                        "reason": exit_reason,
                        "entry": entry,
                        "exit": px,
                        "ret": trade_ret,
                        "equity": equity,
                        "dd": (peak - equity) / peak if peak > 0 else 0.0,
                        "buyScore": entry_buy_score,
                        "sellScore": ss,
                        "RSI": entry_rsi,
                        "MACD_hist": entry_macd,
                        "close": entry_close,
                    })
                    returns.append(trade_ret)

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
                last_trade_i = i + 1

        if trace:
            equity_curve.append(equity)
            dd_curve.append((peak - equity) / peak if peak > 0 else 0.0)

    net = equity - 1.0
    assert_simulator_uses_closed_last_price(candles, close_sig[-1])
    assert_bot_simulator_score_parity(ge, candles)

    balance = START_CAPITAL * equity
    profit = balance - START_CAPITAL
    dd_pct = (max_dd / (peak + 1e-12)) * 100.0

    pf_raw = gross_profit / (gross_loss + 1e-6)
    pf = max(0.0, min(pf_raw, 50.0))
    winrate = (wins / trades * 100.0) if trades > 0 else 0.0

    first_ts = candles[0].get("open_time", candles[0].get("t"))
    last_ts = candles[-1].get("close_time", candles[-1].get("t"))
    years = max((last_ts - first_ts) / (1000 * 60 * 60 * 24 * 365), 1e-9)
    trades_per_year = trades / years
    buy_signal_rate = buy_signal_hits / len(close_sig)
    sell_signal_rate = sell_signal_hits / len(close_sig)

    score = pf

    metrics = Metrics(
        score=score,
        net=net,
        balance=balance,
        profit=profit,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        max_dd=max_dd,
        dd_pct=dd_pct,
        trades=trades,
        wins=wins,
        losses=losses,
        winrate=winrate,
        pf=pf,
        years=years,
        trades_per_year=trades_per_year,
        buy_signal_rate=buy_signal_rate,
        sell_signal_rate=sell_signal_rate,
    )

    if trace:
        return metrics, {
            "events": events,
            "equity_curve": equity_curve,
            "dd_curve": dd_curve,
            "returns": returns,
        }
    return metrics


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


def fitness_from_metrics(m: Metrics, cfg: GAConfig) -> float:
    if m.trades == 0:
        return -1e9

    pf_cap = min(m.pf, 50.0)
    pf_term = math.log1p(pf_cap)
    dd_pen = cfg.dd_weight * (m.dd_pct / 100.0)
    missing = max(0, cfg.trade_floor - m.trades)
    trade_pen = missing * cfg.pen_per_missing_trade
    bonus = max(0.0, m.net) * 0.2
    return pf_term - dd_pen - trade_pen + bonus


PARAM_GENES = [
    "rsi_period",
    "rsi_oversold",
    "rsi_overbought",
    "macd_fast",
    "macd_slow",
    "macd_signal",
    "consec_red",
    "consec_green",
    "buy_th",
    "sell_th",
    "take_profit",
    "stop_loss",
    "cooldown",
    "edge_trigger",
    "use_ha",
]

PARAM_BLOCKS = ["RSI", "MACD", "CONSEC", "RISK", "CONF"]
WEIGHT_KEYS = [
    "w_buy_rsi", "w_buy_macd", "w_buy_consec",
    "w_sell_rsi", "w_sell_macd", "w_sell_consec",
]

WEIGHT_MUT_RATE = 0.35
PARAM_MUT_RATE = 0.08
WEIGHT_MUT_SIGMA = 0.06
BLOCK_SELECTION_WEIGHTS = {
    "RSI": 1.0,
    "MACD": 1.4,
    "CONSEC": 1.0,
    "RISK": 1.0,
    "CONF": 0.8,
}
BLOCK_COUNT_WEIGHTS = {
    1: 0.35,
    2: 0.45,
    3: 0.20,
}

BLOCKS = {
    "RSI": ["rsi_period", "rsi_oversold", "rsi_overbought"],
    "MACD": ["macd_fast", "macd_slow", "macd_signal"],
    "CONSEC": ["consec_red", "consec_green"],
    "RISK": ["take_profit", "stop_loss", "cooldown", "edge_trigger", "use_ha"],
    "WEIGHTS": [
        "w_buy_rsi", "w_buy_macd", "w_buy_consec",
        "w_sell_rsi", "w_sell_macd", "w_sell_consec",
    ],
    "CONF": ["buy_th", "sell_th"],
}

def mutate_weights(child: dict, *, sigma: float = WEIGHT_MUT_SIGMA):
    chosen = random.sample(WEIGHT_KEYS, k=random.choice([1, 2]))
    for key in chosen:
        child[key] = clamp_weight(child[key] + random.gauss(0, sigma))
    apply_weight_constraints(child)


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

    if block_name == "WEIGHTS":
        apply_weight_constraints(child)

    if block_name == "MACD":
        if child["macd_slow"] <= child["macd_fast"]:
            child["macd_slow"] = min(space.spec["macd_slow"]["max"], child["macd_fast"] + 5)
            child["macd_slow"] = space.quantize("macd_slow", child["macd_slow"])


def make_child_blocky(
    space: ParamSpace,
    p1: Genome,
    p2: Genome,
    param_blocks: Optional[list[str]],
    *,
    param_mut_rate: float = PARAM_MUT_RATE,
    weight_mut_rate: float = WEIGHT_MUT_RATE,
    weight_sigma: float = WEIGHT_MUT_SIGMA,
    strength: float = 0.55,
    return_meta: bool = False,
):
    d1 = asdict(p1)
    d2 = asdict(p2)
    child = {k: d1[k] if random.random() < 0.5 else d2[k] for k in d1}

    mutated_blocks = []
    mutated_genes = []
    mutation_type = "copiado"

    pre_mutation = child.copy()

    if random.random() < weight_mut_rate:
        mutate_weights(child, sigma=weight_sigma)
        mutated_weight_genes = [k for k in WEIGHT_KEYS if child[k] != pre_mutation[k]]
        if mutated_weight_genes:
            mutated_blocks.append("WEIGHTS")
            mutated_genes.extend(mutated_weight_genes)

    if param_blocks:
        for param_block in param_blocks:
            if random.random() >= param_mut_rate:
                continue
            before = child.copy()
            mutate_block(child, space, param_block, strength=strength)
            if not any(child[g] != before[g] for g in BLOCKS[param_block]):
                gk = random.choice(BLOCKS[param_block])
                spec = space.spec[gk]
                lo, hi, tp = spec["min"], spec["max"], spec["type"]
                if tp == "int":
                    current = int(child[gk])
                    options = [v for v in range(int(lo), int(hi) + 1, int(spec["step"])) if v != current]
                    if options:
                        child[gk] = random.choice(options)
                else:
                    current = float(child[gk])
                    new_val = lo + (hi - lo) * random.random()
                    if abs(new_val - current) < 1e-9:
                        new_val = lo if abs(lo - current) > abs(hi - current) else hi
                    child[gk] = space.quantize(gk, new_val)
            mutated_param_genes = [g for g in BLOCKS[param_block] if child[g] != before[g]]
            if mutated_param_genes:
                mutated_blocks.append(param_block)
                mutated_genes.extend(mutated_param_genes)

    mutated = len(mutated_genes) > 0
    if mutated:
        mutation_type = "mutado"

    ge = space.clamp_genome(Genome(**child))
    if return_meta:
        all_genes = set(space.spec.keys())
        mutated_genes_set = set(mutated_genes)
        untouched = sorted(all_genes - mutated_genes_set)
        meta = {
            "mutated": mutated,
            "mutation_type": mutation_type,
            "mutated_blocks": sorted(set(mutated_blocks)),
            "mutated_genes": sorted(mutated_genes_set),
            "untouched_genes": untouched,
        }
        return ge, meta
    return ge


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


def make_base_genome(space: ParamSpace, base_params: Optional[dict] = None) -> Genome:
    payload = DEFAULT_GEN.copy()
    if base_params:
        payload.update(base_params)
    ge = Genome(**payload)
    return space.clamp_genome(ge)


def make_freeze_fn(space: ParamSpace, base_genome: Genome, optimize_genes: list[str]):
    optimize_set = set(optimize_genes)
    frozen_genes = [k for k in asdict(base_genome).keys() if k not in optimize_set]

    def _freeze(ge: Genome) -> Genome:
        d = asdict(ge)
        for k in frozen_genes:
            d[k] = getattr(base_genome, k)
        return space.clamp_genome(Genome(**d))

    return _freeze


def run_ga(
    candles,
    space: ParamSpace,
    cfg: GAConfig,
    stop_flag,
    log_fn,
    sample_fn=None,
    postprocess_fn=None,
    allowed_blocks: Optional[list[str]] = None,
    initial_population=None,
    gen_offset: int = 0,
    return_population: bool = False,
):
    GA_DASHBOARD_MODE = False
    is_weight_opt = False  # compat: modo de pesos separado eliminado
    fee_per_side = cfg.fee_side * cfg.fee_mult
    slip_per_side = cfg.slip_side

    # logging mínimo: evitar logs internos del GA

    if sample_fn is None:
        sample_fn = space.sample
    if postprocess_fn is None:
        postprocess_fn = lambda ge: ge

    pop = []
    if initial_population:
        pop = [postprocess_fn(ge) for ge in initial_population]
    if len(pop) < cfg.population:
        pop.extend(postprocess_fn(sample_fn()) for _ in range(cfg.population - len(pop)))
    if len(pop) > cfg.population:
        pop = pop[:cfg.population]

    if not initial_population:
        cov = force_coverage(space, max(12, cfg.population // 6))
        pop[:len(cov)] = [postprocess_fn(ge) for ge in cov]

    best_global = None
    best_metrics = None
    prev_best_hash = None
    stuck = 0
    prev_best_hash = None
    active_param_blocks = None
    recent_blocks = deque(maxlen=7)
    param_blocks = [b for b in (allowed_blocks or PARAM_BLOCKS) if b in PARAM_BLOCKS]
    if not param_blocks:
        param_blocks = PARAM_BLOCKS[:]
    print(f"[GA] bloques_disponibles={param_blocks}", flush=True)

    def strength_for_combo(count: int) -> float:
        if count <= 1:
            return 0.55
        if count == 2:
            return 0.85
        return 1.15

    def format_gene(value, mutated: bool, fmt: str) -> str:
        out = format(value, fmt)
        return f"{out}*" if mutated else out

    def format_genome(ge: Genome, mutated_genes: Optional[set] = None) -> str:
        mutated_genes = mutated_genes or set()
        ha = format_gene(ge.use_ha, "use_ha" in mutated_genes, "d")
        rsi_p = format_gene(ge.rsi_period, "rsi_period" in mutated_genes, "d")
        rsi_os = format_gene(ge.rsi_oversold, "rsi_oversold" in mutated_genes, ".0f")
        rsi_ob = format_gene(ge.rsi_overbought, "rsi_overbought" in mutated_genes, ".0f")
        macd_f = format_gene(ge.macd_fast, "macd_fast" in mutated_genes, "d")
        macd_s = format_gene(ge.macd_slow, "macd_slow" in mutated_genes, "d")
        macd_sig = format_gene(ge.macd_signal, "macd_signal" in mutated_genes, "d")
        consec_r = format_gene(ge.consec_red, "consec_red" in mutated_genes, "d")
        consec_g = format_gene(ge.consec_green, "consec_green" in mutated_genes, "d")
        tp = format_gene(ge.take_profit, "take_profit" in mutated_genes, ".3f")
        sl = format_gene(ge.stop_loss, "stop_loss" in mutated_genes, ".3f")
        cd = format_gene(ge.cooldown, "cooldown" in mutated_genes, "d")
        edge = format_gene(ge.edge_trigger, "edge_trigger" in mutated_genes, "d")
        wbr = format_gene(ge.w_buy_rsi, "w_buy_rsi" in mutated_genes, ".2f")
        wbm = format_gene(ge.w_buy_macd, "w_buy_macd" in mutated_genes, ".2f")
        wbc = format_gene(ge.w_buy_consec, "w_buy_consec" in mutated_genes, ".2f")
        wsr = format_gene(ge.w_sell_rsi, "w_sell_rsi" in mutated_genes, ".2f")
        wsm = format_gene(ge.w_sell_macd, "w_sell_macd" in mutated_genes, ".2f")
        wsc = format_gene(ge.w_sell_consec, "w_sell_consec" in mutated_genes, ".2f")
        buy_th = format_gene(ge.buy_th, "buy_th" in mutated_genes, ".2f")
        sell_th = format_gene(ge.sell_th, "sell_th" in mutated_genes, ".2f")
        return (
            f"HA={ha} "
            f"RSI(p={rsi_p},os={rsi_os},ob={rsi_ob}) "
            f"MACD({macd_f},{macd_s},{macd_sig}) "
            f"consec(R={consec_r},G={consec_g}) "
            f"TP={tp} SL={sl} cd={cd} edge={edge} "
            f"Wbuy=({wbr},{wbm},{wbc}) Wsell=({wsr},{wsm},{wsc}) "
            f"umbrales(buy={buy_th},sell={sell_th})"
        )

    mutation_counter = {gene: 0 for gene in space.spec.keys()}

    def print_mutation_counter(reason: str):
        print(f"[MUTATION COUNTER] {reason}")
        for gene in sorted(mutation_counter.keys()):
            print(f"{gene}={mutation_counter[gene]}")

    def pick_random_block_combo() -> Optional[list[str]]:
        if not param_blocks:
            return None
        weights = [BLOCK_SELECTION_WEIGHTS.get(b, 1.0) for b in param_blocks]
        counts = list(BLOCK_COUNT_WEIGHTS.keys())
        count_weights = [BLOCK_COUNT_WEIGHTS[c] for c in counts]
        while True:
            count = random.choices(counts, weights=count_weights, k=1)[0]
            count = min(count, len(param_blocks))
            chosen = set()
            while len(chosen) < count:
                block = random.choices(param_blocks, weights=weights, k=1)[0]
                chosen.add(block)
            combo = tuple(sorted(chosen))
            if combo not in recent_blocks:
                recent_blocks.append(combo)
                return list(combo)

    total_gens = cfg.generations
    for gen in range(1, total_gens + 1):
        if stop_flag():
            return best_global, best_metrics

        gen_display = gen + gen_offset
        gen_start = time.perf_counter()
        last_heartbeat = gen_start
        scored = []
        for idx, ge in enumerate(pop, start=1):
            indiv_start = time.perf_counter()
            if DEBUG_GA_PROGRESS and idx % GA_PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - gen_start
                print(
                    f"[GA] GEN {gen_display} progreso {idx}/{len(pop)} | elapsed={elapsed:.1f}s",
                    flush=True,
                )
                last_heartbeat = time.perf_counter()
            m = simulate_spot(candles, ge, fee_per_side, slip_per_side)
            indiv_elapsed = time.perf_counter() - indiv_start
            if DEBUG_GA_PROGRESS and indiv_elapsed >= GA_INDIVIDUAL_WARN_S:
                print(
                    f"[GA][WARN] GEN {gen_display} individuo {idx}/{len(pop)} lento "
                    f"({indiv_elapsed:.1f}s) | bloques={active_param_blocks} | {format_genome(ge)}",
                    flush=True,
                )
                last_heartbeat = time.perf_counter()
            score = fitness_from_metrics(m, cfg)
            scored.append((score, ge, m))

        eval_elapsed = time.perf_counter() - gen_start
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_ge, best_m = scored[0]
        best_hash = hash(tuple(asdict(best_ge).items()))
        unique_hashes = {hash(tuple(asdict(ge).items())) for _, ge, _ in scored}
        elite_hashes = [hash(tuple(asdict(ge).items())) for ge in [x[1] for x in scored[:cfg.elite]]]
        elite_unique = len(set(elite_hashes))

        if best_global is None or best_score > best_global[0] + 1e-9:
            prev_best_global_score = best_global[0] if best_global else None
            best_global = (best_score, copy.deepcopy(best_ge))
            best_metrics = best_m
            stuck = 0
            active_param_blocks = None
        else:
            stuck += 1
            if is_weight_opt:
                weight_no_improve += 1

        gen_display = gen + gen_offset
        if not GA_DASHBOARD_MODE:
            log_fn(
                f"[GEN {gen_display}] score={best_score:.4f} PF={best_m.pf:.2f} "
                f"trades={best_m.trades} tpy={best_m.trades_per_year:.1f} "
                f"DD={best_m.dd_pct:.2f}% net={best_m.net:.4f} | "
                f"ganancia={best_m.profit:.2f} balance={best_m.balance:.2f} | "
                f"wr={best_m.winrate:.1f}% | "
                f"HA={best_ge.use_ha} RSI(p={best_ge.rsi_period},os={best_ge.rsi_oversold:.0f},ob={best_ge.rsi_overbought:.0f}) | "
                f"MACD({best_ge.macd_fast},{best_ge.macd_slow},{best_ge.macd_signal}) | "
                f"consec(R={best_ge.consec_red},G={best_ge.consec_green}) | "
                f"TP={best_ge.take_profit:.3f} SL={best_ge.stop_loss:.3f} cd={best_ge.cooldown} edge={best_ge.edge_trigger} | "
                f"buy_th={best_ge.buy_th:.2f} sell_th={best_ge.sell_th:.2f} | "
                f"Wbuy({best_ge.w_buy_rsi:.2f},{best_ge.w_buy_macd:.2f},{best_ge.w_buy_consec:.2f}) "
                f"Wsell({best_ge.w_sell_rsi:.2f},{best_ge.w_sell_macd:.2f},{best_ge.w_sell_consec:.2f})"
            )
            log_fn(
                f"[DBG] uniq={len(unique_hashes)} elite_unique={elite_unique} "
                f"best_hash={best_hash} repeated={best_hash == prev_best_hash}"
            )
        prev_best_hash = best_hash
        if is_weight_opt:
            print(
                f"[WEIGHT-OPT] gen={gen_display} | attempts={weight_attempts} | "
                f"unique={weight_unique} | best_score={best_score}",
                flush=True,
            )
            if weight_no_improve >= WEIGHT_NO_IMPROVE_LIMIT:
                print("[INFO] Weight optimizer: no improvement → exit", flush=True)
                break

        print(
            f"GEN {gen_display} | best_score={best_score:.4f} | net={best_m.net:.4f} | "
            f"PF={best_m.pf:.2f} | DD={best_m.dd_pct:.1f} | unique={len(unique_hashes)}",
            flush=True,
        )
        if DEBUG_GA_PROGRESS:
            print(f"[GA] GEN {gen_display} eval_time={eval_elapsed:.1f}s", flush=True)
        prev_best_hash = best_hash
        mask_saturated = False
        if stuck >= BLOCK_STUCK_LIMIT:
            mask_saturated = True

        if active_param_blocks is None:
            active_param_blocks = pick_random_block_combo()
            print(
                f"[BLOQUE] activos: {active_param_blocks} | pesos siempre mutan | gens={BLOCK_STUCK_LIMIT}",
                flush=True,
            )
        elif mask_saturated:
            print("[INFO] Bloque sin mejora -> cambiando bloque", flush=True)
            before_switch = time.perf_counter()
            active_param_blocks = pick_random_block_combo()
            switch_elapsed = time.perf_counter() - before_switch
            print(
                f"[BLOQUE] activos: {active_param_blocks} | pesos siempre mutan | "
                f"gens={BLOCK_STUCK_LIMIT} | cambio_en={switch_elapsed:.3f}s",
                flush=True,
            )
            stuck = 0

        if DEBUG_FULL_POPULATION:
            print(
                f"===== GENERACION {gen_display} | POP_SIZE={len(pop)} | "
                f"BLOQUE(S) ACTIVOS: {active_param_blocks} (pesos siempre mutan) ====="
            )

        extra_cov = []

        elite = [best_global[1]] if best_global else [scored[0][1]]
        pool = [x[1] for x in scored[:max(cfg.elite * 6, cfg.population // 2)]]

        children = []
        child_meta = [] if DEBUG_MUTATION else None

        forced_strength = strength_for_combo(len(active_param_blocks or [])) if active_param_blocks else None

        for _ in range(cfg.n_cons):
            p1 = random.choice(pool)
            p2 = random.choice(pool)
            child_result = make_child_blocky(
                space,
                p1,
                p2,
                param_blocks=active_param_blocks,
                strength=forced_strength if active_param_blocks else 0.55,
                return_meta=DEBUG_MUTATION,
            )
            if DEBUG_MUTATION:
                child, meta = child_result
                child_meta.append(meta)
            else:
                child = child_result
            children.append(postprocess_fn(child))

        for _ in range(cfg.n_exp):
            p1 = random.choice(pool)
            p2 = random.choice(pool)
            child_result = make_child_blocky(
                space,
                p1,
                p2,
                param_blocks=active_param_blocks,
                strength=forced_strength if active_param_blocks else 0.80,
                return_meta=DEBUG_MUTATION,
            )
            if DEBUG_MUTATION:
                child, meta = child_result
                child_meta.append(meta)
            else:
                child = child_result
            children.append(postprocess_fn(child))

        for _ in range(cfg.n_wild):
            p1 = random.choice(pool)
            p2 = random.choice(pool)
            child_result = make_child_blocky(
                space,
                p1,
                p2,
                param_blocks=active_param_blocks,
                strength=forced_strength if active_param_blocks else 1.00,
                return_meta=DEBUG_MUTATION,
            )
            if DEBUG_MUTATION:
                child, meta = child_result
                child_meta.append(meta)
            else:
                child = child_result
            children.append(postprocess_fn(child))

        if DEBUG_MUTATION:
            all_genes = sorted(space.spec.keys())
            block_counts = {bn: 0 for bn in BLOCKS.keys()}
            mutated_count = 0
            for meta in child_meta:
                if meta["mutated"]:
                    mutated_count += 1
                    for bn in meta["mutated_blocks"]:
                        block_counts[bn] += 1
                    for gene in meta["mutated_genes"]:
                        mutation_counter[gene] += 1

            structural_blocks = {"RSI", "MACD", "CONSEC", "RISK"}
            structural_mutated = any(block_counts[bn] > 0 for bn in structural_blocks)

            def log_individual(label: str, ge: Genome, meta: Optional[dict] = None):
                if meta is None:
                    meta = {
                        "mutation_type": "copiado",
                        "mutated_genes": [],
                        "untouched_genes": all_genes,
                    }
                elif not meta.get("untouched_genes"):
                    meta["untouched_genes"] = all_genes
                if "mutated_genes" not in meta:
                    meta["mutated_genes"] = []
                mutated_set = set(meta.get("mutated_genes", []))
                print(
                    f"[{label.upper()}] tipo={meta['mutation_type']} "
                    f"mutados={meta['mutated_genes']} no_tocados={meta['untouched_genes']} | "
                    f"{format_genome(ge, mutated_set)}"
                )

            if gen_display % 10 == 0:
                print_mutation_counter("cada_10_gen")
            if stuck > 0:
                print_mutation_counter("sin_mejora")

        new_pop = elite + children
        population_entries = []
        population_entries.append(
            {"genome": elite[0], "meta": None, "type": "elite", "blocks": []}
        )
        for idx, child in enumerate(children):
            meta = child_meta[idx] if child_meta else None
            if meta and meta.get("mutation_type") == "aleatorio":
                tipo = "aleatorio"
            elif meta and meta.get("mutated"):
                tipo = "mutado"
            else:
                tipo = "clonado"
            blocks = meta.get("mutated_blocks", []) if meta else []
            population_entries.append(
                {"genome": child, "meta": meta, "type": tipo, "blocks": blocks}
            )

        if extra_cov:
            k = min(len(extra_cov), max(5, cfg.population // 10))
            mutated_extra = []
            for ge in extra_cov[:k]:
                child_result = make_child_blocky(
                    space,
                    ge,
                    ge,
                    param_blocks=active_param_blocks,
                    strength=forced_strength if active_param_blocks else 0.80,
                    return_meta=DEBUG_MUTATION,
                )
                if DEBUG_MUTATION:
                    child, meta = child_result
                else:
                    child = child_result
                    meta = None
                mutated_extra.append((child, meta))
            new_pop[-k:] = [entry[0] for entry in mutated_extra]
            for idx, (child, meta) in enumerate(mutated_extra):
                pop_idx = len(new_pop) - k + idx
                tipo = "mutado" if meta and meta.get("mutated") else "clonado"
                blocks = meta.get("mutated_blocks", []) if meta else []
                population_entries[pop_idx] = {
                    "genome": child,
                    "meta": meta,
                    "type": tipo,
                    "blocks": blocks,
                }

        if len(new_pop) > cfg.population:
            new_pop = new_pop[:cfg.population]
            population_entries = population_entries[:cfg.population]
        while len(new_pop) < cfg.population:
            seed = postprocess_fn(sample_fn())
            child_result = make_child_blocky(
                space,
                seed,
                seed,
                param_blocks=active_param_blocks,
                strength=forced_strength if active_param_blocks else 0.80,
                return_meta=DEBUG_MUTATION,
            )
            if DEBUG_MUTATION:
                child, meta = child_result
            else:
                child = child_result
                meta = None
            new_pop.append(child)
            population_entries.append(
                {
                    "genome": child,
                    "meta": meta,
                    "type": "mutado" if meta and meta.get("mutated") else "clonado",
                    "blocks": meta.get("mutated_blocks", []) if meta else [],
                }
            )

        pop = new_pop

        evaluated = len(scored)
        pop_size = len(pop)
        sim_ok = evaluated == pop_size
        if not sim_ok:
            print(f"[ERROR] GEN {gen_display}: simulación incompleta (evaluated={evaluated}, expected={pop_size})", flush=True)
            return best_global, best_metrics

    if return_population:
        return best_global, best_metrics, pop
    return best_global, best_metrics


def run_ga_params_only(
    candles,
    space: ParamSpace,
    cfg: GAConfig,
    stop_flag,
    log_fn,
    base_params=None,
    initial_population=None,
    gen_offset: int = 0,
    return_population: bool = False,
):
    base = make_base_genome(space, base_params)
    freeze_fn = make_freeze_fn(space, base, PARAM_GENES)
    return run_ga(
        candles=candles,
        space=space,
        cfg=cfg,
        stop_flag=stop_flag,
        log_fn=log_fn,
        postprocess_fn=freeze_fn,
        allowed_blocks=PARAM_BLOCKS,
        initial_population=initial_population,
        gen_offset=gen_offset,
        return_population=return_population,
    )


class OptimizerGUI:
    def __init__(self, root: tk.Widget, apply_callback):
        self.root = root
        self.apply_callback = apply_callback

        self.worker = None
        self._stop = False
        self.best_genome = None
        self.best_metrics = None
        self.cached_candles = None
        self.last_trace = None
        self.last_trace_metrics = None

        self.var_symbol = tk.StringVar(value=SYMBOL)
        self.var_tf = tk.StringVar(value=INTERVAL)
        self.var_candles = tk.IntVar(value=3000)
        self.var_start_date = tk.StringVar(value=START_DATE)
        self.var_end_date = tk.StringVar(value=END_DATE)
        self.var_cache_ind = tk.BooleanVar(value=CACHE_INDICATORS)

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

        self.var_opt_mode = tk.StringVar(value="Ambos")

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
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)

        opt_tab = ttk.Frame(nb)
        insp_tab = ttk.Frame(nb)
        nb.add(opt_tab, text="Optimización")
        nb.add(insp_tab, text="Inspector")

        frm = ttk.Frame(opt_tab, padding=10)
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
        add_labeled(top, "Inicio:", self.var_start_date, 12)
        add_labeled(top, "Fin:", self.var_end_date, 12)
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
        ttk.Checkbutton(fit, text="Cache indicadores", variable=self.var_cache_ind).pack(side="left", padx=8)

        mode = ttk.LabelFrame(frm, text="Modo de optimización")
        mode.pack(fill="x", padx=pad, pady=pad)
        for text in ("Optimizar Parámetros", "Ambos"):
            ttk.Radiobutton(mode, text=text, variable=self.var_opt_mode, value=text).pack(side="left", padx=8)

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
        self.build_inspector(insp_tab)

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

    def build_inspector(self, parent: tk.Widget):
        pad = 6
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=pad, pady=pad)

        self.inspector_warn = tk.StringVar(value="Sin datos todavía.")
        ttk.Label(top, textvariable=self.inspector_warn, foreground="#b00020").pack(side="left")

        summary = ttk.LabelFrame(parent, text="Resumen")
        summary.pack(fill="x", padx=pad, pady=pad)
        self.inspector_summary = tk.Text(summary, height=6, wrap="word")
        self.inspector_summary.pack(fill="both", expand=True, padx=pad, pady=pad)

        table = ttk.LabelFrame(parent, text="Trades")
        table.pack(fill="both", expand=True, padx=pad, pady=pad)
        cols = ("time", "reason", "ret", "equity", "dd", "buyScore", "sellScore")
        self.inspector_tree = ttk.Treeview(table, columns=cols, show="headings", height=10)
        for c in cols:
            self.inspector_tree.heading(c, text=c)
            self.inspector_tree.column(c, width=100, anchor="center")
        self.inspector_tree.pack(fill="both", expand=True, padx=pad, pady=pad)

        charts = ttk.LabelFrame(parent, text="Charts")
        charts.pack(fill="both", expand=True, padx=pad, pady=pad)

        fig = Figure(figsize=(7, 4), dpi=100)
        self.ax_equity = fig.add_subplot(311)
        self.ax_dd = fig.add_subplot(312)
        self.ax_hist = fig.add_subplot(313)
        fig.tight_layout(pad=1.0)
        self.inspector_canvas = FigureCanvasTkAgg(fig, master=charts)
        self.inspector_canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_inspector(self, metrics: Metrics, trace_payload: dict, cfg: GAConfig):
        if not metrics:
            return

        warnings = []
        if metrics.trades < cfg.trade_floor * 0.25:
            warnings.append("WARNING: Estrategia muerta (muy pocos trades)")
        if metrics.gross_loss < 1e-6 and metrics.trades <= 10:
            warnings.append("WARNING: PF inflado por ausencia de pérdidas")
        if metrics.pf > 30 and metrics.trades < 20:
            warnings.append("WARNING: Probable overfitting")
        if metrics.buy_signal_rate < 0.01:
            warnings.append("WARNING: Señales excesivamente restrictivas")

        self.inspector_warn.set(" | ".join(warnings) if warnings else "Sin warnings críticos.")

        returns = trace_payload.get("returns", [])
        avg_ret = statistics.mean(returns) if returns else 0.0
        med_ret = statistics.median(returns) if returns else 0.0
        max_win = max(returns) if returns else 0.0
        max_loss = min(returns) if returns else 0.0

        summary_lines = [
            f"trades={metrics.trades} | tpy={metrics.trades_per_year:.1f} | winrate={metrics.winrate:.1f}%",
            f"gross_profit={metrics.gross_profit:.4f} gross_loss={metrics.gross_loss:.4f} PF={metrics.pf:.2f}",
            f"net={metrics.net:.4f} balance={metrics.balance:.2f} ganancia={metrics.profit:.2f} DD%={metrics.dd_pct:.2f}",
            f"avg_ret={avg_ret:.4f} median_ret={med_ret:.4f} max_win={max_win:.4f} max_loss={max_loss:.4f}",
            f"buy_signal_rate={metrics.buy_signal_rate:.3f} sell_signal_rate={metrics.sell_signal_rate:.3f}",
        ]
        self.inspector_summary.delete("1.0", "end")
        self.inspector_summary.insert("end", "\n".join(summary_lines))

        for row in self.inspector_tree.get_children():
            self.inspector_tree.delete(row)
        for ev in trace_payload.get("events", []):
            time_str = datetime.utcfromtimestamp(ev["time"] / 1000).strftime("%Y-%m-%d %H:%M")
            self.inspector_tree.insert(
                "",
                "end",
                values=(
                    time_str,
                    ev["reason"],
                    f"{ev['ret']:.4f}",
                    f"{ev['equity']:.3f}",
                    f"{ev['dd']*100:.2f}%",
                    f"{ev['buyScore']:.2f}",
                    f"{ev['sellScore']:.2f}",
                ),
            )

        equity_curve = trace_payload.get("equity_curve", [])
        dd_curve = trace_payload.get("dd_curve", [])

        self.ax_equity.clear()
        self.ax_equity.plot(equity_curve, color="#1976d2")
        self.ax_equity.set_title("Equity curve")
        self.ax_equity.set_ylabel("Equity")

        self.ax_dd.clear()
        self.ax_dd.plot([d * 100 for d in dd_curve], color="#d32f2f")
        self.ax_dd.set_title("Drawdown %")
        self.ax_dd.set_ylabel("DD %")

        self.ax_hist.clear()
        if returns:
            self.ax_hist.hist(returns, bins=20, color="#455a64")
        self.ax_hist.set_title("Histograma retornos por trade")
        self.ax_hist.set_xlabel("Retorno")

        self.inspector_canvas.draw()

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
                "start_date": self.var_start_date.get().strip(),
                "end_date": self.var_end_date.get().strip(),
                "cache_indicators": bool(self.var_cache_ind.get()),
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
            self.var_start_date.set(payload.get("start_date", START_DATE))
            self.var_end_date.set(payload.get("end_date", END_DATE))
            self.var_cache_ind.set(bool(payload.get("cache_indicators", CACHE_INDICATORS)))

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
            start_date = self.var_start_date.get().strip()
            end_date = self.var_end_date.get().strip()
            cache_ind = bool(self.var_cache_ind.get())

            self._stop = False
            self.txt.delete("1.0", "end")

            self.log(f"[OK] Temporalidad objetivo: {tf} | rango {start_date} -> {end_date}")
            self.log(f"[GA] pop={cfg.population} gens={cfg.generations} elite={cfg.elite} | cons={cfg.n_cons} exp={cfg.n_exp} wild={cfg.n_wild}")
            self.log(f"[INFO] Ejecutando archivo: {os.path.abspath(__file__)}")
            self.log("[DATOS] Revisando cache de velas...")

            def job():
                try:
                    tv_path = tv_build_json_path(symbol, tf, start_date, end_date)
                    if not os.path.exists(tv_path):
                        self.log("[DESCARGA] Bajando datos desde Binance...")
                        tv_df = fetch_ohlcv_by_date_range(symbol, tf, start_date, end_date)
                        if cache_ind:
                            tv_df = tv_apply_indicator_cache(tv_df, DEFAULT_GEN)
                        tv_path = save_candles_json(tv_df, symbol, tf, start_date, end_date, "binance")
                        self.log(f"[OK] JSON creado: {tv_path}")
                    else:
                        self.log(f"[OK] JSON existente: {tv_path}")

                    tv_df = load_candles_json(tv_path)
                    self.log(f"[OK] {symbol} cargado: {len(tv_df)} velas ({tf})")
                    if not tv_df.empty:
                        self.log(
                            f"[VERIF] rango={tv_df['datetime'].min()} -> {tv_df['datetime'].max()} "
                            f"| velas={len(tv_df)}"
                        )
                        self.log("[VERIF] primeras 3 filas:")
                        self.log(tv_df[["datetime", "open", "high", "low", "close"]].head(3).to_string(index=False))
                        self.log("[VERIF] ultimas 3 filas:")
                        self.log(tv_df[["datetime", "open", "high", "low", "close"]].tail(3).to_string(index=False))
                        if "rsi" in tv_df.columns or "macd_hist" in tv_df.columns:
                            nan_rsi = tv_df["rsi"].isna().sum() if "rsi" in tv_df.columns else 0
                            nan_macd = tv_df["macd_hist"].isna().sum() if "macd_hist" in tv_df.columns else 0
                            self.log(f"[VERIF] NaN RSI={nan_rsi} NaN MACD={nan_macd}")

                    candles = tv_df_to_candles(tv_df, tf)
                    self.cached_candles = candles

                    test_params = DEFAULT_GEN.copy()
                    test_params.update(
                        {
                            "use_ha": 1,
                            "rsi_period": 15,
                            "rsi_oversold": 32,
                            "rsi_overbought": 80,
                            "macd_fast": 26,
                            "macd_slow": 87,
                            "macd_signal": 47,
                            "consec_red": 5,
                            "consec_green": 3,
                            "take_profit": 0.27,
                            "stop_loss": 0.04,
                            "cooldown": 12,
                            "buy_th": 0.85,
                            "sell_th": 0.55,
                        }
                    )
                    test_ge = Genome(**test_params)
                    test_metrics = simulate_spot(
                        candles,
                        test_ge,
                        self.var_fee.get() * self.var_fee_mult.get(),
                        self.var_slip.get(),
                    )
                    self.log(
                        f"[PRUEBA] GEN600 balance={test_metrics.balance:.2f} trades={test_metrics.trades} "
                        f"PF={test_metrics.pf:.2f} net={test_metrics.net:.4f} winrate={test_metrics.winrate:.1f}%"
                    )
                    log_fn = lambda s: self.root.after(0, self.log, s)
                    fee_per_side = self.var_fee.get() * self.var_fee_mult.get()
                    slip_per_side = self.var_slip.get()

                    def refresh_inspector(ge: Genome):
                        metrics, trace_payload = simulate_spot(
                            candles, ge, fee_per_side, slip_per_side, trace=True
                        )
                        self.last_trace = trace_payload
                        self.last_trace_metrics = metrics
                        self.root.after(
                            0,
                            lambda: self.update_inspector(metrics, trace_payload, cfg),
                        )

                    best_global, best_metrics = run_ga(
                        candles=candles,
                        space=self.space,
                        cfg=cfg,
                        stop_flag=lambda: self._stop,
                        log_fn=log_fn,
                    )
                    self.best_genome = best_global[1] if best_global else None
                    self.best_metrics = best_metrics
                    if self.best_genome:
                        refresh_inspector(self.best_genome)

                    if self.best_genome:
                        self.root.after(0, self.log, "[FIN] Mejor encontrado:")
                        self.root.after(0, self.log, json.dumps(asdict(self.best_genome), indent=2, ensure_ascii=False))
                except Exception as e:
                    print("[ERROR] OptimizerGUI.start failure:", flush=True)
                    print(traceback.format_exc(), flush=True)
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
                start_date = self.var_start_date.get().strip()
                end_date = self.var_end_date.get().strip()
                cache_ind = bool(self.var_cache_ind.get())
                tv_path = tv_build_json_path(symbol, tf, start_date, end_date)
                if not os.path.exists(tv_path):
                    self.log("[DESCARGA] Bajando datos desde Binance para backtest...")
                    tv_df = fetch_ohlcv_by_date_range(symbol, tf, start_date, end_date)
                    if cache_ind:
                        tv_df = tv_apply_indicator_cache(tv_df, DEFAULT_GEN)
                    tv_path = save_candles_json(tv_df, symbol, tf, start_date, end_date, "binance")
                tv_df = load_candles_json(tv_path)
                self.cached_candles = tv_df_to_candles(tv_df, tf)

            ge = self.space.sample()
            metrics = simulate_spot(self.cached_candles, ge, self.var_fee.get() * self.var_fee_mult.get(), self.var_slip.get())
            self.log(
                f"[BACKTEST] score={fitness_from_metrics(metrics, self.build_cfg()):.4f} "
                f"PF={metrics.pf:.2f} trades={metrics.trades} tpy={metrics.trades_per_year:.1f} "
                f"DD={metrics.dd_pct:.2f}% net={metrics.net:.4f} "
                f"ganancia={metrics.profit:.2f} balance={metrics.balance:.2f} "
                f"wr={metrics.winrate:.1f}%"
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
