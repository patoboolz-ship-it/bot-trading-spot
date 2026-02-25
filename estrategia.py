#!/usr/bin/env python3
"""Estrategia standalone: backtest + dashboard mini-exchange en un solo archivo."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import queue
import tkinter as tk
from tkinter import ttk

BEST_GEN = {
    "use_ha": 1,
    "rsi_period": 12,
    "rsi_oversold": 44.0,
    "rsi_overbought": 68.0,
    "macd_fast": 14,
    "macd_slow": 92,
    "macd_signal": 17,
    "consec_red": 5,
    "consec_green": 3,
    "w_buy_rsi": 0.5,
    "w_buy_macd": 0.15,
    "w_buy_consec": 0.35,
    "buy_th": 0.6,
    "w_sell_rsi": 0.42574257425742573,
    "w_sell_macd": 0.1485148514851485,
    "w_sell_consec": 0.42574257425742573,
    "sell_th": 0.5,
    "take_profit": 0.29,
    "stop_loss": 0.21,
    "cooldown": 1,
    "edge_trigger": 1,
}

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}
INITIAL_CAPITAL = 1000.0


@dataclass
class Metrics:
    net: float
    trades: int
    wins: int
    losses: int
    winrate: float
    pf: float
    dd_pct: float
    balance_final: float
    pnl_usd: float
    roi_pct: float


class DotDict(dict):
    def __getattr__(self, item):
        return self[item]


def ts_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")




def detect_local_ipv4() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


def configure_stdout() -> None:
    """Fuerza salida en tiempo real para consolas/IDEs que bufferizan print()."""
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        # Compatibilidad con runners que no soportan reconfigure
        pass

def interval_to_ms(interval: str) -> int:
    if interval not in INTERVAL_MS:
        raise ValueError(f"Intervalo no soportado: {interval}")
    return INTERVAL_MS[interval]


def is_candle_closed(close_time_ms: int, server_time_ms: int, interval_ms: int) -> bool:
    return int(close_time_ms) < int(server_time_ms) - int(interval_ms)


def filter_closed_candles(candles: list[dict[str, Any]], interval: str, server_time_ms: int) -> list[dict[str, Any]]:
    iv = interval_to_ms(interval)
    return [c for c in candles if is_candle_closed(c["close_time"], server_time_ms, iv)]


def normalize3(a: float, b: float, c: float):
    s = a + b + c
    if s <= 1e-12:
        return (1 / 3, 1 / 3, 1 / 3)
    return (a / s, b / s, c / s)


def ema_series(values: list[float], period: int) -> list[float]:
    alpha = 2.0 / (max(1, int(period)) + 1.0)
    out, ema = [], values[0]
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        out.append(float(ema))
    return out


def rsi_wilder_series(close: list[float], period: int) -> list[float]:
    n, p = max(2, len(close)), max(2, int(period))
    out = [50.0] * n
    if n <= p:
        return out
    gains, losses = [0.0] * n, [0.0] * n
    for i in range(1, n):
        d = close[i] - close[i - 1]
        gains[i], losses[i] = max(d, 0.0), max(-d, 0.0)
    avg_gain = sum(gains[1 : p + 1]) / p
    avg_loss = sum(losses[1 : p + 1]) / p
    for i in range(p, n):
        if i > p:
            avg_gain = (avg_gain * (p - 1) + gains[i]) / p
            avg_loss = (avg_loss * (p - 1) + losses[i]) / p
        rs = avg_gain / (avg_loss + 1e-12)
        out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def macd_hist_series(close: list[float], fast: int, slow: int, signal: int) -> list[float]:
    fast = max(2, int(fast))
    slow = max(fast + 1, int(slow))
    signal = max(2, int(signal))
    macd_line = [a - b for a, b in zip(ema_series(close, fast), ema_series(close, slow))]
    signal_line = ema_series(macd_line, signal)
    return [m - s for m, s in zip(macd_line, signal_line)]


def heikin_ashi(candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candles:
        return []
    out = []
    prev_ha_open = (candles[0]["open"] + candles[0]["close"]) / 2.0
    prev_ha_close = (candles[0]["open"] + candles[0]["high"] + candles[0]["low"] + candles[0]["close"]) / 4.0
    for i, c in enumerate(candles):
        ha_close = (c["open"] + c["high"] + c["low"] + c["close"]) / 4.0
        ha_open = (prev_ha_open + prev_ha_close) / 2.0 if i > 0 else prev_ha_open
        row = dict(c)
        row.update({"open": ha_open, "high": max(c["high"], ha_open, ha_close), "low": min(c["low"], ha_open, ha_close), "close": ha_close})
        out.append(row)
        prev_ha_open, prev_ha_close = ha_open, ha_close
    return out


def consec_up(close: list[float], n: int) -> float:
    n = max(1, int(n))
    if len(close) <= n:
        return 0.0
    for j in range(len(close) - n + 1, len(close)):
        if close[j] < close[j - 1]:
            return 0.0
    return 1.0


def consec_down(close: list[float], n: int) -> float:
    n = max(1, int(n))
    if len(close) <= n:
        return 0.0
    for j in range(len(close) - n + 1, len(close)):
        if close[j] > close[j - 1]:
            return 0.0
    return 1.0


def score_components_at_index(close: list[float], rsi_arr: list[float], mh_arr: list[float], i: int, ge) -> dict[str, float]:
    rsi_val, mh = rsi_arr[i], mh_arr[i]
    mh_prev = mh_arr[i - 1] if i > 0 else mh
    buy_rsi = 1.0 if rsi_val <= ge.rsi_oversold else 0.0
    sell_rsi = 1.0 if rsi_val >= ge.rsi_overbought else 0.0
    buy_macd = 1.0 if ((mh_prev <= 0.0 < mh) if ge.edge_trigger else (mh > 0.0)) else 0.0
    sell_macd = 1.0 if ((mh_prev >= 0.0 > mh) if ge.edge_trigger else (mh < 0.0)) else 0.0
    buy_consec = consec_up(close[: i + 1], ge.consec_green)
    sell_consec = consec_down(close[: i + 1], ge.consec_red)
    wbr, wbm, wbc = normalize3(ge.w_buy_rsi, ge.w_buy_macd, ge.w_buy_consec)
    wsr, wsm, wsc = normalize3(ge.w_sell_rsi, ge.w_sell_macd, ge.w_sell_consec)
    return {
        "buy_score": wbr * buy_rsi + wbm * buy_macd + wbc * buy_consec,
        "sell_score": wsr * sell_rsi + wsm * sell_macd + wsc * sell_consec,
    }


def fetch_historical_closed(client, symbol: str, interval: str, start: str, end: str) -> list[dict[str, Any]]:
    kl = client.get_historical_klines(symbol=symbol, interval=interval, start_str=start, end_str=end)
    candles = [{"open_time": int(r[0]), "close_time": int(r[6]), "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])} for r in kl]
    return filter_closed_candles(candles, interval, int(client.get_server_time()["serverTime"]))


def load_params(args: argparse.Namespace) -> dict[str, Any]:
    params = dict(BEST_GEN)
    if getattr(args, "params_json", None):
        with open(args.params_json, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    for kv in getattr(args, "set", []):
        if "=" not in kv:
            raise ValueError(f"Override inválido: {kv}. Usa key=value")
        k, raw = kv.split("=", 1)
        raw = raw.strip()
        params[k.strip()] = 1 if raw.lower() == "true" else 0 if raw.lower() == "false" else float(raw) if "." in raw else int(raw)
    return params


def run_timeline(candles: list[dict[str, Any]], ge, fee_side: float, slip_side: float, *, initial_capital: float, feedback: bool):
    data = heikin_ashi(candles) if int(ge.use_ha) == 1 else candles
    close_sig = [c["close"] for c in data]
    rsi_arr = rsi_wilder_series(close_sig, int(ge.rsi_period))
    mh_arr = macd_hist_series(close_sig, int(ge.macd_fast), int(ge.macd_slow), int(ge.macd_signal))
    in_pos, entry, entry_i, last_trade_i = False, 0.0, -10_000, -10_000
    events, trade_rows, open_trade, equity_curve = [], [], None, []
    equity = 1.0
    balance = float(initial_capital)
    peak = 1.0
    wins = losses = 0
    gross_profit = gross_loss = 0.0

    for i in range(1, len(close_sig) - 1):
        payload = score_components_at_index(close_sig, rsi_arr, mh_arr, i, ge)
        bs, ss = payload["buy_score"], payload["sell_score"]
        next_candle, exec_open = candles[i + 1], float(candles[i + 1]["open"])
        if not in_pos:
            if (i - last_trade_i) >= int(ge.cooldown) and bs >= float(ge.buy_th):
                entry, entry_i, in_pos = exec_open * (1.0 + slip_side), i + 1, True
                events.append({"type": "BUY", "i": i + 1, "time": next_candle["open_time"], "price": entry, "buy_score": bs, "sell_score": ss, "capital_antes": balance})
                open_trade = {"entrada_idx": i + 1, "entrada_fecha": ts_to_iso(next_candle["open_time"]), "entrada_precio": entry, "capital_antes": balance}
                if feedback:
                    print(f"[BUY] {open_trade['entrada_fecha']} precio={entry:.4f} capital={balance:.2f}")
        else:
            tp_px, sl_px = entry * (1.0 + abs(float(ge.take_profit))), entry * (1.0 - abs(float(ge.stop_loss)))
            exit_reason, px = ("SL", sl_px) if next_candle["low"] <= sl_px else (("TP", tp_px) if next_candle["high"] >= tp_px else (("SIG", exec_open) if ss >= float(ge.sell_th) else (None, exec_open)))
            if exit_reason:
                sell_price = px * (1.0 - slip_side)
                trade_ratio = (1.0 - fee_side) * (sell_price / (entry + 1e-12))
                trade_ret = trade_ratio - 1.0
                equity, balance = equity * trade_ratio, balance * trade_ratio
                peak = max(peak, equity)
                wins += int(trade_ret >= 0)
                losses += int(trade_ret < 0)
                gross_profit += max(trade_ret, 0.0)
                gross_loss += max(-trade_ret, 0.0)
                events.append({"type": "SELL", "i": i + 1, "time": next_candle["open_time"], "price": sell_price, "reason": exit_reason, "ret": trade_ret, "entry_i": entry_i, "entry_price": entry, "buy_score": bs, "sell_score": ss, "capital_despues": balance})
                if open_trade is not None:
                    trade_rows.append({"n_operacion": len(trade_rows) + 1, "entrada_fecha": open_trade["entrada_fecha"], "salida_fecha": ts_to_iso(next_candle["open_time"]), "entrada_precio": open_trade["entrada_precio"], "salida_precio": sell_price, "motivo_salida": exit_reason, "retorno_pct": trade_ret * 100.0, "capital_antes": open_trade["capital_antes"], "capital_despues": balance, "pnl_usd": balance - open_trade["capital_antes"]})
                if feedback:
                    print(f"[SELL-{exit_reason}] {ts_to_iso(next_candle['open_time'])} ret={trade_ret*100:.2f}% capital={balance:.2f}")
                in_pos, last_trade_i, open_trade = False, i + 1, None
        equity_curve.append({"idx": i, "time": candles[i]["open_time"], "time_iso": ts_to_iso(candles[i]["open_time"]), "equity": equity, "capital": balance, "en_posicion": int(in_pos)})

    trades = wins + losses
    max_dd = max((max(r["equity"] for r in equity_curve[: idx + 1]) - row["equity"]) / max(r["equity"] for r in equity_curve[: idx + 1]) for idx, row in enumerate(equity_curve)) if equity_curve else 0.0
    metrics = Metrics(net=equity - 1.0, trades=trades, wins=wins, losses=losses, winrate=(wins / trades * 100.0) if trades else 0.0, pf=(gross_profit / (gross_loss + 1e-9)) if trades else 0.0, dd_pct=max_dd * 100.0, balance_final=balance, pnl_usd=balance - float(initial_capital), roi_pct=(equity - 1.0) * 100.0)
    return events, metrics, equity_curve, trade_rows


def export_events(events: list[dict[str, Any]], metrics: Metrics, out_dir: str, *, equity_curve: list[dict[str, Any]], trade_rows: list[dict[str, Any]], params: dict[str, Any], csv_delimiter: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eventos_crudos.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for ev in events for k in ev.keys()}) if events else ["type", "time"]
        w = csv.DictWriter(f, fieldnames=keys + ["time_iso"], delimiter=csv_delimiter)
        w.writeheader()
        for ev in events:
            row = dict(ev)
            row["time_iso"] = ts_to_iso(ev["time"])
            w.writerow(row)
    with open(os.path.join(out_dir, "operaciones_resumen.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["n_operacion", "entrada_fecha", "salida_fecha", "entrada_precio", "salida_precio", "motivo_salida", "retorno_pct", "capital_antes", "capital_despues", "pnl_usd"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter=csv_delimiter)
        w.writeheader()
        for row in trade_rows:
            w.writerow({c: row.get(c) for c in cols})
    with open(os.path.join(out_dir, "curva_capital.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "time", "time_iso", "equity", "capital", "en_posicion"], delimiter=csv_delimiter)
        w.writeheader()
        for row in equity_curve:
            w.writerow(row)
    with open(os.path.join(out_dir, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)


def plot_chart(candles: list[dict[str, Any]], events: list[dict[str, Any]], symbol: str, out_dir: str, equity_curve: list[dict[str, Any]], *, show_chart: bool, plot_candles: int):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib no disponible: {exc}")
        return
    os.makedirs(out_dir, exist_ok=True)
    start_idx = max(0, len(candles) - max(300, plot_candles))
    view = candles[start_idx:]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    ax1.plot([c["close"] for c in view], linewidth=1)
    for ev in events:
        if ev["i"] < start_idx:
            continue
        x, mk, col = ev["i"] - start_idx, "^" if ev["type"] == "BUY" else "v", "green" if ev["type"] == "BUY" else "red"
        ax1.scatter([x], [ev["price"]], marker=mk, color=col)
    ax1.set_title(f"{symbol} estrategia")
    ax2.plot([e["capital"] for e in equity_curve if e["idx"] >= start_idx], color="tab:blue")
    ax2.set_ylabel("Capital")
    png = os.path.join(out_dir, "grafico_estrategia.png")
    plt.tight_layout(); plt.savefig(png, dpi=140)
    print(f"[OK] Gráfico guardado: {png}")
    if show_chart:
        plt.show()
    else:
        plt.close(fig)


# ---------- Dashboard en un solo archivo ----------

HTML_DASHBOARD = """
<!doctype html><html><head><meta charset='utf-8'/><title>Estrategia Dashboard</title>
<script src='https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js'></script>
<style>
body{font-family:Arial;background:#0b1220;color:#e2e8f0;margin:0}
.top,.grid{display:flex;gap:10px;padding:10px}.panel{background:#111827;border:1px solid #334155;border-radius:8px;padding:10px}
.grid{align-items:stretch}.left{flex:3}.right{flex:2}
table{width:100%;font-size:12px;border-collapse:collapse}td,th{border-bottom:1px solid #334155;padding:4px;text-align:left}
#chart,#eqchart{height:280px}.tabs{display:flex;gap:6px;margin-bottom:8px}.tabbtn{padding:6px 10px;border:1px solid #475569;background:#1e293b;color:#e2e8f0;cursor:pointer}
.tabbtn.active{background:#2563eb}.tab{display:none}.tab.active{display:block}
</style>
</head><body>
<div class='top panel'>
<select id='symbol'><option>SOLUSDT</option><option>BTCUSDT</option><option>ETHUSDT</option></select>
<select id='interval'><option>1m</option><option>5m</option><option>15m</option><option selected>1h</option><option>4h</option><option>1d</option></select>
<label><input type='checkbox' id='live' checked/> LIVE</label><button id='goLive'>Go Live</button>
<button id='play'>Play</button><button id='pause'>Pause</button><button id='stop'>Stop</button>
<span>Estado: <b id='state'>IDLE</b></span><span>Última acción: <b id='lastAction'>init</b></span>
</div>
<div class='grid'>
<div class='left panel'><h4>Precio (OHLCV)</h4><div id='chart'></div><h4>Curva Equity</h4><div id='eqchart'></div></div>
<div class='right panel'><h4>Cartera / PnL</h4><div id='account'></div><h4>Activos Spot</h4><div id='assets'></div></div>
</div>
<div class='panel' style='margin:10px'>
<div class='tabs'>
<button class='tabbtn active' data-tab='ordersTab'>Órdenes</button>
<button class='tabbtn' data-tab='tradesTab'>Trades</button>
<button class='tabbtn' data-tab='logsTab'>Logs</button>
</div>
<div id='ordersTab' class='tab active'><table><thead><tr><th>Estado</th><th>Side</th><th>Qty</th><th>Precio</th><th>Hora</th></tr></thead><tbody id='orders'></tbody></table></div>
<div id='tradesTab' class='tab'><table><thead><tr><th>Side</th><th>Qty</th><th>Precio</th><th>Comisión</th><th>Hora</th></tr></thead><tbody id='trades'></tbody></table></div>
<div id='logsTab' class='tab'><pre id='logs' style='max-height:220px;overflow:auto'></pre></div>
</div>
<script>
const chart = LightweightCharts.createChart(document.getElementById('chart'), {layout:{background:{color:'#0b1220'},textColor:'#cbd5e1'}});
const s = chart.addCandlestickSeries();
const eqChart = LightweightCharts.createChart(document.getElementById('eqchart'), {layout:{background:{color:'#0b1220'},textColor:'#cbd5e1'},height:280});
const eqSeries = eqChart.addLineSeries({color:'#22c55e'});
let live=true,pending=[];let markers=[];
function toTs(ms){return Math.floor(ms/1000)}
function pushLog(m){const box=document.getElementById('logs'); box.textContent='['+new Date().toLocaleTimeString()+'] '+m+'\n'+box.textContent;}
function rows(id,data,cols){document.getElementById(id).innerHTML=(data||[]).slice(-60).reverse().map(r=>'<tr>'+cols.map(c=>'<td>'+(r[c]??'')+'</td>').join('')+'</tr>').join('');}
function account(a){document.getElementById('account').innerHTML='<div><b>USDT disponible:</b> '+((a.balances||{}).USDT||0).toFixed(2)+'</div><div><b>Equity total:</b> '+(a.equity||0).toFixed(2)+' USDT</div><div><b>Rentabilidad:</b> '+(a.pnl_usdt||0).toFixed(2)+' USDT ('+(a.pnl_pct||0).toFixed(2)+'%)</div><div><b>PnL realizado:</b> '+(a.realized_pnl_usd||0).toFixed(2)+' USDT</div><div><b>PnL no realizado:</b> '+(a.unrealized_pnl_usd||0).toFixed(2)+' USDT</div>'; document.getElementById('assets').innerHTML=(a.assets||[]).map(x=>x.asset+': '+Number(x.qty).toFixed(4)+' (~'+Number(x.est_usdt).toFixed(2)+' USDT)').join('<br>')||'Sin posiciones';}
function applyCandles(items){(items||[]).forEach(c=>{const r={time:toTs(c.time),open:c.open,high:c.high,low:c.low,close:c.close}; if(!live){pending.push(r); return;} s.update(r);});}
function setMarkersFromTrades(trades){markers=(trades||[]).map(t=>({time:toTs(t.time),position:t.side==='BUY'?'belowBar':'aboveBar',color:t.side==='BUY'?'#22c55e':'#ef4444',shape:t.side==='BUY'?'arrowUp':'arrowDown',text:t.side+' '+t.qty})); s.setMarkers(markers);}
function setEquity(perf){eqSeries.setData((perf.equity_curve||[]).map(x=>({time:toTs(x.time),value:x.equity})));}
function setLogs(payload){const merged=[...(payload.logs||[]), ...((payload.errors||[]).map(e=>'ERROR '+e))].slice(-120).reverse(); document.getElementById('logs').textContent=merged.join('\n');}
function applySnapshot(x){document.getElementById('state').textContent=x.bot_state; document.getElementById('lastAction').textContent=x.last_action||'init'; document.getElementById('symbol').value=x.symbol; document.getElementById('interval').value=x.interval; s.setData((x.market_state.candles||[]).map(c=>({time:toTs(c.time),open:c.open,high:c.high,low:c.low,close:c.close}))); account(x.account_state); rows('orders',x.orders_state,['status','side','origQty','price','time_iso']); rows('trades',x.trades_state,['side','qty','price','commission','time_iso']); setMarkersFromTrades(x.trades_state||[]); setEquity(x.performance_state||{equity_curve:[]}); setLogs({logs:x.logs||[], errors:x.errors||[]}); chart.timeScale().scrollToRealTime(); eqChart.timeScale().scrollToRealTime();}
function switchTab(id){document.querySelectorAll('.tabbtn').forEach(b=>b.classList.toggle('active',b.dataset.tab===id)); document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.id===id));}
document.querySelectorAll('.tabbtn').forEach(b=>b.onclick=()=>switchTab(b.dataset.tab));
async function post(u,b={}){await fetch(u,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)})}
async function frontendLog(level,message,extra=''){try{await fetch('/frontend-log',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({level,message,extra})});}catch(_e){}}
['play','pause','stop'].forEach(id=>document.getElementById(id).onclick=()=>post('/control/'+id));
document.getElementById('symbol').onchange=e=>post('/symbol',{symbol:e.target.value});
document.getElementById('interval').onchange=e=>post('/timeframe',{interval:e.target.value});
document.getElementById('live').onchange=e=>{live=e.target.checked; pushLog(live?'LIVE ON':'HIST ON')};
document.getElementById('goLive').onclick=()=>{live=true;document.getElementById('live').checked=true;pending.forEach(c=>s.update(c));pending=[];chart.timeScale().scrollToRealTime();eqChart.timeScale().scrollToRealTime();};
window.addEventListener('error',(ev)=>frontendLog('ERROR',ev.message,`${ev.filename||''}:${ev.lineno||0}:${ev.colno||0}`));
window.addEventListener('unhandledrejection',(ev)=>frontendLog('ERROR','Unhandled promise rejection',String(ev.reason||'')));
function ws(){const p=location.protocol==='https:'?'wss':'ws'; const w=new WebSocket(p+'://'+location.host+'/stream'); w.onopen=()=>pushLog('WS conectado'); w.onclose=()=>{pushLog('WS desconectado. Reintentando...'); setTimeout(ws,1500)}; w.onmessage=e=>{const m=JSON.parse(e.data); if(m.type==='snapshot')applySnapshot(m.payload); if(m.type==='market.candle_update')applyCandles(m.payload.candles||[]); if(m.type==='account.update')account(m.payload); if(m.type==='orders.update')rows('orders',m.payload.orders||[],['status','side','origQty','price','time_iso']); if(m.type==='trades.update'){rows('trades',m.payload.trades||[],['side','qty','price','commission','time_iso']); setMarkersFromTrades(m.payload.trades||[]);} if(m.type==='performance.update')setEquity(m.payload); if(m.type==='logs.update')setLogs(m.payload); if(m.type==='bot.state'){document.getElementById('state').textContent=m.payload.state; document.getElementById('lastAction').textContent=m.payload.last_action||'';}};}
frontendLog('INFO','Dashboard cargado',location.href); ws();
</script></body></html>
"""



@dataclass
class AppState:
    bot_state: str = "IDLE"
    symbol: str = "SOLUSDT"
    interval: str = "1h"
    last_action: str = "init"
    balances: dict[str, float] = None
    assets: list[dict[str, Any]] = None
    equity: float = INITIAL_CAPITAL
    equity_initial: float = INITIAL_CAPITAL
    pnl_usdt: float = 0.0
    pnl_pct: float = 0.0
    realized_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    position_cost_usd: float = 0.0
    orders: list[dict[str, Any]] = None
    trades: list[dict[str, Any]] = None
    candles: list[dict[str, Any]] = None
    logs: list[str] = None
    errors: list[str] = None
    equity_curve: list[dict[str, Any]] = None

    def __post_init__(self):
        self.balances = {"USDT": INITIAL_CAPITAL} if self.balances is None else self.balances
        self.assets = [] if self.assets is None else self.assets
        self.orders = [] if self.orders is None else self.orders
        self.trades = [] if self.trades is None else self.trades
        self.candles = [] if self.candles is None else self.candles
        self.logs = [] if self.logs is None else self.logs
        self.errors = [] if self.errors is None else self.errors
        self.equity_curve = [] if self.equity_curve is None else self.equity_curve


class StateStore:
    def __init__(self, db_path: str = "estrategia_dashboard.db"):
        self.state = AppState()
        self.lock = threading.RLock()
        self.db_path = db_path
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS events(ts INTEGER, kind TEXT, payload TEXT)")
        conn.commit(); conn.close()

    def persist(self, kind: str, payload: dict[str, Any]):
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO events(ts,kind,payload) VALUES(?,?,?)", (int(time.time() * 1000), kind, json.dumps(payload)))
        conn.commit(); conn.close()

    def log(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(f"[DASHBOARD] {line}", flush=True)
        with self.lock:
            self.state.logs.append(line)
            self.state.logs = self.state.logs[-300:]

    def log_error(self, msg: str):
        self.log(f"ERROR: {msg}")
        with self.lock:
            self.state.errors.append(msg)
            self.state.errors = self.state.errors[-100:]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "bot_state": self.state.bot_state,
                "last_action": self.state.last_action,
                "symbol": self.state.symbol,
                "interval": self.state.interval,
                "market_state": {
                    "candles": self.state.candles[-1500:],
                    "last_price": self.state.candles[-1]["close"] if self.state.candles else None,
                },
                "account_state": {
                    "balances": self.state.balances,
                    "assets": self.state.assets,
                    "equity": self.state.equity,
                    "pnl_usdt": self.state.pnl_usdt,
                    "pnl_pct": self.state.pnl_pct,
                    "realized_pnl_usd": self.state.realized_pnl_usd,
                    "unrealized_pnl_usd": self.state.unrealized_pnl_usd,
                },
                "orders_state": self.state.orders[-100:],
                "trades_state": self.state.trades[-100:],
                "performance_state": {"equity_curve": self.state.equity_curve[-2000:]},
                "logs": self.state.logs[-100:],
                "errors": self.state.errors[-50:],
            }


class DataLayer:
    def __init__(self):
        self.client = None
        try:
            from binance.client import Client

            self.client = Client(None, None)
        except Exception:
            self.client = None

    def get_candles(self, symbol: str, interval: str, limit: int = 500) -> list[dict[str, Any]]:
        if self.client is None:
            return self._mock(limit)
        try:
            rows = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            out = []
            for r in rows:
                out.append({"time": int(r[0]), "open_time": int(r[0]), "close_time": int(r[6]), "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])})
            return filter_closed_candles(out, interval, int(self.client.get_server_time()["serverTime"]))
        except Exception:
            return self._mock(limit)

    def _mock(self, limit: int) -> list[dict[str, Any]]:
        base = int(time.time() // 3600 * 3600 * 1000)
        price = 100.0
        out = []
        for i in range(limit):
            t = base - (limit - i) * 3600 * 1000
            o = price
            c = max(1.0, o + random.uniform(-1.8, 1.8))
            h = max(o, c) + random.uniform(0, 1.2)
            l = min(o, c) - random.uniform(0, 1.2)
            out.append({"time": t, "open_time": t, "close_time": t + 3600 * 1000 - 1, "open": o, "high": h, "low": l, "close": c, "volume": random.uniform(1000, 6000)})
            price = c
        return out


class BotEngine:
    def __init__(self, store: StateStore, genome: dict[str, Any]):
        self.store = store
        self.ge = DotDict(genome)
        self.in_pos = False
        self.entry = 0.0
        self.last_i = -10_000

    def start(self):
        self.store.state.bot_state = "RUNNING"
        self.store.state.last_action = "play"
        self.store.log("Bot RUNNING")

    def pause(self):
        self.store.state.bot_state = "PAUSED"
        self.store.state.last_action = "pause"
        self.store.log("Bot PAUSED")

    def stop(self):
        self.store.state.bot_state = "IDLE"
        self.store.state.last_action = "stop"
        self.store.log("Bot STOP")

    def on_market(self):
        if self.store.state.bot_state != "RUNNING" or len(self.store.state.candles) < 120:
            return
        data = heikin_ashi(self.store.state.candles) if int(self.ge.use_ha) == 1 else self.store.state.candles
        close = [c["close"] for c in data]
        i = len(close) - 2
        rsi = rsi_wilder_series(close, int(self.ge.rsi_period))
        mh = macd_hist_series(close, int(self.ge.macd_fast), int(self.ge.macd_slow), int(self.ge.macd_signal))
        score = score_components_at_index(close, rsi, mh, i, self.ge)
        nxt = self.store.state.candles[i + 1]
        px = float(nxt["open"])
        usdt = float(self.store.state.balances.get("USDT", 0.0))
        base_asset = self.store.state.symbol.replace("USDT", "")
        qty = float(self.store.state.balances.get(base_asset, 0.0))

        if not self.in_pos and score["buy_score"] >= float(self.ge.buy_th) and (i - self.last_i) >= int(self.ge.cooldown) and usdt > 10:
            qbuy = (usdt * 0.98) / (px + 1e-12)
            invested = usdt * 0.98
            self.store.state.balances["USDT"] = max(0.0, usdt - invested)
            self.store.state.balances[base_asset] = qty + qbuy
            self.store.state.position_cost_usd = invested
            self.in_pos, self.entry = True, px
            order = {"status": "FILLED", "side": "BUY", "origQty": round(qbuy, 6), "price": px, "time": int(time.time() * 1000), "time_iso": ts_to_iso(int(time.time() * 1000))}
            trade = {"side": "BUY", "qty": round(qbuy, 6), "price": px, "commission": round(qbuy * 0.001, 6), "time": order["time"], "time_iso": order["time_iso"]}
            self.store.state.orders.append(order); self.store.state.trades.append(trade)
            self.store.persist("order", order); self.store.persist("trade", trade)
            self.store.state.last_action = f"BUY {qbuy:.6f} {base_asset} @ {px:.4f}"
            self.store.log(f"BUY ejecutado qty={qbuy:.4f} px={px:.4f}")
        elif self.in_pos:
            tp, sl = self.entry * (1.0 + abs(float(self.ge.take_profit))), self.entry * (1.0 - abs(float(self.ge.stop_loss)))
            do_sell = nxt["low"] <= sl or nxt["high"] >= tp or score["sell_score"] >= float(self.ge.sell_th)
            if do_sell and qty > 0:
                usdt_new = qty * px * (1.0 - 0.001)
                realized = usdt_new - self.store.state.position_cost_usd
                self.store.state.realized_pnl_usd += realized
                self.store.state.position_cost_usd = 0.0
                self.store.state.balances["USDT"] = usdt_new
                self.store.state.balances[base_asset] = 0.0
                self.in_pos, self.last_i = False, i
                order = {"status": "FILLED", "side": "SELL", "origQty": round(qty, 6), "price": px, "time": int(time.time() * 1000), "time_iso": ts_to_iso(int(time.time() * 1000))}
                trade = {"side": "SELL", "qty": round(qty, 6), "price": px, "commission": round(qty * 0.001, 6), "time": order["time"], "time_iso": order["time_iso"]}
                self.store.state.orders.append(order); self.store.state.trades.append(trade)
                self.store.persist("order", order); self.store.persist("trade", trade)
                self.store.state.last_action = f"SELL {qty:.6f} {base_asset} @ {px:.4f}"
                self.store.log(f"SELL ejecutado qty={qty:.4f} px={px:.4f}")


def recalc_account(store: StateStore):
    last = store.state.candles[-1]["close"] if store.state.candles else 0.0
    eq = float(store.state.balances.get("USDT", 0.0))
    assets = []
    holdings_value = 0.0
    for asset, qty in store.state.balances.items():
        if asset == "USDT":
            continue
        est = float(qty) * float(last)
        assets.append({"asset": asset, "qty": qty, "est_usdt": est})
        eq += est
        holdings_value += est
    store.state.assets = assets
    store.state.equity = eq
    store.state.unrealized_pnl_usd = holdings_value - float(store.state.position_cost_usd)
    store.state.pnl_usdt = eq - store.state.equity_initial
    base = store.state.equity_initial or 1.0
    store.state.pnl_pct = store.state.pnl_usdt / base * 100.0
    store.state.equity_curve.append({"time": int(time.time() * 1000), "equity": eq, "pnl": store.state.pnl_usdt})
    store.state.equity_curve = store.state.equity_curve[-5000:]


async def run_dashboard(args: argparse.Namespace):
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Falta FastAPI/uvicorn. Instala con: pip install fastapi uvicorn") from exc

    store, data = StateStore(args.db_path), DataLayer()
    print("[INFO] Iniciando modo dashboard...", flush=True)
    print(f"[INFO] Configuración: host={args.host} port={args.port} symbol={args.symbol} interval={args.interval}", flush=True)
    print(f"[INFO] Poll={args.poll_seconds}s live_limit={args.live_limit} db_path={args.db_path}", flush=True)
    store.state.symbol = args.symbol
    store.state.interval = args.interval
    store.state.equity_initial = args.capital_inicial
    store.state.balances = {"USDT": args.capital_inicial}
    engine = BotEngine(store, dict(BEST_GEN))

    clients: list[WebSocket] = []

    async def broadcast(msg: dict[str, Any]):
        bad = []
        for ws in clients:
            try:
                await ws.send_json(msg)
            except Exception:
                bad.append(ws)
        for ws in bad:
            if ws in clients:
                clients.remove(ws)

    async def market_loop():
        tick_count = 0
        store.log("Market loop iniciado")
        while True:
            try:
                candles = data.get_candles(store.state.symbol, store.state.interval, limit=args.live_limit)
                with store.lock:
                    prev = store.state.candles[-1]["time"] if store.state.candles else None
                    store.state.candles = candles
                    recalc_account(store)
                    engine.on_market()
                    recalc_account(store)
                    is_new = prev is None or candles[-1]["time"] != prev
                    payload = {"candles": candles[-2:] if is_new else candles[-1:]}
                    snap = store.snapshot()
                await broadcast({"type": "market.candle_update", "payload": payload})
                await broadcast({"type": "account.update", "payload": snap["account_state"]})
                await broadcast({"type": "orders.update", "payload": {"orders": snap["orders_state"]}})
                await broadcast({"type": "trades.update", "payload": {"trades": snap["trades_state"]}})
                await broadcast({"type": "performance.update", "payload": snap["performance_state"]})
                await broadcast({"type": "logs.update", "payload": {"logs": snap["logs"], "errors": snap["errors"]}})
                await broadcast({"type": "bot.state", "payload": {"state": store.state.bot_state, "last_action": store.state.last_action}})
                tick_count += 1
                if tick_count % 10 == 0:
                    store.log(f"Heartbeat market loop | symbol={store.state.symbol} interval={store.state.interval} equity={store.state.equity:.2f}")
            except Exception:
                store.state.bot_state = "ERROR"
                store.log_error(traceback.format_exc())
            await asyncio.sleep(max(1, args.poll_seconds))

    app = FastAPI(title="Estrategia Dashboard single-file")

    @app.on_event("startup")
    async def _startup():
        store.log("En espera: inicializando interfaz y cargando velas...")
        store.state.candles = data.get_candles(store.state.symbol, store.state.interval, limit=args.live_limit)
        recalc_account(store)
        store.log(f"Interfaz lista con {len(store.state.candles)} velas iniciales")
        asyncio.create_task(market_loop())

    @app.get("/")
    def index():
        return HTMLResponse(HTML_DASHBOARD)

    @app.get("/state")
    def state():
        return store.snapshot()

    @app.post("/frontend-log")
    def frontend_log(payload: dict[str, Any]):
        level = str(payload.get("level", "INFO")).upper()
        msg = str(payload.get("message", ""))
        extra = str(payload.get("extra", ""))
        store.log(f"FRONTEND-{level}: {msg} {extra}".strip())
        return {"ok": True}

    @app.post("/control/play")
    def play():
        engine.start()
        return {"ok": True, "state": store.state.bot_state}

    @app.post("/control/pause")
    def pause():
        engine.pause()
        return {"ok": True, "state": store.state.bot_state}

    @app.post("/control/stop")
    def stop():
        engine.stop()
        return {"ok": True, "state": store.state.bot_state}

    @app.post("/symbol")
    def symbol(payload: dict[str, Any]):
        store.state.symbol = str(payload.get("symbol", store.state.symbol)).upper()
        store.log(f"Símbolo cambiado a {store.state.symbol}")
        return {"ok": True, "symbol": store.state.symbol}

    @app.post("/timeframe")
    def timeframe(payload: dict[str, Any]):
        iv = str(payload.get("interval", store.state.interval))
        interval_to_ms(iv)
        store.state.interval = iv
        store.log(f"Temporalidad cambiada a {store.state.interval}")
        return {"ok": True, "interval": store.state.interval}

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        clients.append(ws)
        await ws.send_json({"type": "snapshot", "payload": store.snapshot()})
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            if ws in clients:
                clients.remove(ws)

    cfg = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(cfg)
    local_ip = detect_local_ipv4()
    print(f"[INFO] Dashboard disponible local: http://127.0.0.1:{args.port}")
    print(f"[INFO] Dashboard disponible en tu red Wi-Fi: http://{local_ip}:{args.port}")
    print(f"[INFO] Host de escucha: {args.host} (usa 0.0.0.0 para acceso desde otros dispositivos)")
    print("[INFO] Logs de frontend/backend se imprimirán en esta consola.")
    print("[INFO] Si no abre la web en 10s, prueba: /state y revisa firewall/VPN.")
    print(f"[INFO] Healthcheck rápido: http://127.0.0.1:{args.port}/state", flush=True)
    await server.serve()


class EstrategiaLauncherGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Estrategia.py - Launcher Visual")
        self.proc: subprocess.Popen[str] | None = None
        self.log_q: queue.Queue[str] = queue.Queue()

        self.var_mode = tk.StringVar(value="dashboard")
        self.var_symbol = tk.StringVar(value="SOLUSDT")
        self.var_interval = tk.StringVar(value="1h")
        self.var_host = tk.StringVar(value="0.0.0.0")
        self.var_port = tk.StringVar(value="8085")
        self.var_start = tk.StringVar(value="2020-01-01")
        self.var_end = tk.StringVar(value="2024-12-31")
        self.var_status = tk.StringVar(value="IDLE")
        self.var_show = tk.BooleanVar(value=False)

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._drain_logs)

    def _build(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)

        row = 0
        ttk.Label(frm, text="Modo").grid(row=row, column=0, sticky="w")
        ttk.Combobox(frm, textvariable=self.var_mode, values=["dashboard", "backtest"], state="readonly", width=14).grid(row=row, column=1, sticky="w")
        ttk.Label(frm, text="Estado:").grid(row=row, column=2, sticky="e")
        ttk.Label(frm, textvariable=self.var_status).grid(row=row, column=3, sticky="w", padx=(4, 0))

        row += 1
        ttk.Label(frm, text="Símbolo").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_symbol, width=16).grid(row=row, column=1, sticky="w")
        ttk.Label(frm, text="Intervalo").grid(row=row, column=2, sticky="e")
        ttk.Combobox(frm, textvariable=self.var_interval, values=list(INTERVAL_MS.keys()), state="readonly", width=12).grid(row=row, column=3, sticky="w")

        row += 1
        ttk.Label(frm, text="Host").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_host, width=16).grid(row=row, column=1, sticky="w")
        ttk.Label(frm, text="Port").grid(row=row, column=2, sticky="e")
        ttk.Entry(frm, textvariable=self.var_port, width=12).grid(row=row, column=3, sticky="w")

        row += 1
        ttk.Label(frm, text="Inicio").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_start, width=16).grid(row=row, column=1, sticky="w")
        ttk.Label(frm, text="Fin").grid(row=row, column=2, sticky="e")
        ttk.Entry(frm, textvariable=self.var_end, width=12).grid(row=row, column=3, sticky="w")

        row += 1
        ttk.Checkbutton(frm, text="Abrir ventana matplotlib en backtest (--show)", variable=self.var_show).grid(row=row, column=0, columnspan=4, sticky="w", pady=(4, 6))

        row += 1
        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=4, sticky="w")
        ttk.Button(btns, text="Iniciar", command=self.start_process).pack(side="left")
        ttk.Button(btns, text="Detener", command=self.stop_process).pack(side="left", padx=6)
        ttk.Button(btns, text="Abrir Web", command=self.open_web).pack(side="left", padx=6)

        row += 1
        ttk.Label(frm, text="Consola / errores").grid(row=row, column=0, columnspan=4, sticky="w", pady=(8, 2))
        row += 1
        self.txt = tk.Text(frm, height=18, width=120)
        self.txt.grid(row=row, column=0, columnspan=4, sticky="nsew")
        frm.grid_columnconfigure(1, weight=1)
        frm.grid_rowconfigure(row, weight=1)

    def _cmd(self) -> list[str]:
        mode = self.var_mode.get().strip()
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--mode",
            mode,
            "--symbol",
            self.var_symbol.get().strip() or "SOLUSDT",
            "--interval",
            self.var_interval.get().strip() or "1h",
        ]
        if mode == "dashboard":
            cmd += ["--host", self.var_host.get().strip() or "0.0.0.0", "--port", self.var_port.get().strip() or "8085"]
        else:
            cmd += ["--start", self.var_start.get().strip() or "2020-01-01", "--end", self.var_end.get().strip() or "2024-12-31"]
            cmd += ["--show"] if self.var_show.get() else ["--no-show"]
        return cmd

    def start_process(self):
        if self.proc and self.proc.poll() is None:
            self._append("[GUI] Ya hay un proceso en ejecución.\n")
            return
        cmd = self._cmd()
        self._append(f"[GUI] Ejecutando: {' '.join(cmd)}\n")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.var_status.set("RUNNING")
        threading.Thread(target=self._pump_stdout, daemon=True).start()

    def stop_process(self):
        if not self.proc or self.proc.poll() is not None:
            self._append("[GUI] No hay proceso activo.\n")
            self.var_status.set("IDLE")
            return
        self.proc.terminate()
        self._append("[GUI] Proceso detenido.\n")
        self.var_status.set("IDLE")

    def open_web(self):
        url = f"http://127.0.0.1:{self.var_port.get().strip() or '8085'}"
        webbrowser.open(url)
        self._append(f"[GUI] Abriendo {url}\n")

    def _pump_stdout(self):
        assert self.proc is not None
        if self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            self.log_q.put(line)
        rc = self.proc.poll()
        self.log_q.put(f"[GUI] Proceso finalizado con código {rc}.\n")
        self.var_status.set("IDLE")

    def _drain_logs(self):
        while True:
            try:
                line = self.log_q.get_nowait()
            except queue.Empty:
                break
            self._append(line)
        self.root.after(200, self._drain_logs)

    def _append(self, line: str):
        self.txt.insert("end", line)
        self.txt.see("end")

    def _on_close(self):
        self.stop_process()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def run_visual_launcher():
    try:
        gui = EstrategiaLauncherGUI()
        gui.run()
    except tk.TclError as exc:
        raise RuntimeError(
            "No se pudo abrir la interfaz visual (Tk). En Linux sin escritorio usa --mode dashboard."
        ) from exc


def make_parser():
    p = argparse.ArgumentParser(description="Estrategia.py: backtest + dashboard")
    p.add_argument("--mode", choices=["backtest", "dashboard", "visual"], default="backtest")
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--interval", default="1h")
    p.add_argument("--out", default="estrategia_output")
    p.add_argument("--fee-side", type=float, default=0.001)
    p.add_argument("--slip-side", type=float, default=0.0005)
    p.add_argument("--params-json", default=None)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--capital-inicial", type=float, default=INITIAL_CAPITAL)
    p.add_argument("--max-candles", type=int, default=8000)
    p.add_argument("--plot-candles", type=int, default=2500)
    p.add_argument("--feedback", dest="feedback", action="store_true")
    p.add_argument("--no-feedback", dest="feedback", action="store_false")
    p.add_argument("--csv-delimiter", default=";")
    p.add_argument("--show", dest="show", action="store_true")
    p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8085)
    p.add_argument("--poll-seconds", type=int, default=3)
    p.add_argument("--live-limit", type=int, default=500)
    p.add_argument("--db-path", default="estrategia_dashboard.db")
    p.set_defaults(show=True, feedback=True)
    return p


def print_mode_help(args: argparse.Namespace) -> None:
    if args.mode == "backtest":
        print("[WARN] Estás en modo BACKTEST: este modo NO crea webserver.", flush=True)
        print("[WARN] Para dashboard web usa exactamente: --mode dashboard", flush=True)
    elif args.mode == "visual":
        print("[INFO] Estás en modo VISUAL: launcher gráfico tipo bot.py.", flush=True)
    else:
        print("[INFO] Estás en modo DASHBOARD: se creará webserver FastAPI.", flush=True)


def main():
    configure_stdout()
    args = make_parser().parse_args()
    interval_to_ms(args.interval)
    print_mode_help(args)

    if args.mode == "backtest" and args.show:
        print("[WARN] --show abre una ventana de gráfico y puede parecer 'pegado' hasta cerrarla.", flush=True)
        print("[WARN] Si quieres ejecución sin bloqueo usa --no-show.", flush=True)

    if args.mode == "dashboard":
        asyncio.run(run_dashboard(args))
        return
    if args.mode == "visual":
        run_visual_launcher()
        return

    params = load_params(args)
    params["capital_inicial"] = args.capital_inicial
    ge = DotDict(params)
    print("[INFO] Parámetros finales:")
    print(json.dumps(dict(ge), indent=2, ensure_ascii=False))

    try:
        from binance.client import Client
    except Exception as exc:
        raise RuntimeError("Falta dependencia python-binance. Instala con: pip install python-binance") from exc

    client = Client(None, None)
    candles = fetch_historical_closed(client, args.symbol, args.interval, args.start, args.end)
    if args.max_candles > 0 and len(candles) > args.max_candles:
        candles = candles[-args.max_candles :]
    if len(candles) < 220:
        raise RuntimeError(f"Muy pocas velas cerradas para simular: {len(candles)}")

    events, metrics, equity_curve, trade_rows = run_timeline(candles, ge, args.fee_side, args.slip_side, initial_capital=args.capital_inicial, feedback=args.feedback)
    print(f"[RESUMEN] rentable={'SI' if metrics.pnl_usd > 0 else 'NO'} trades={metrics.trades} winrate={metrics.winrate:.2f}% pf={metrics.pf:.2f} roi={metrics.roi_pct:.2f}% pnl_usd={metrics.pnl_usd:.2f}")
    export_events(events, metrics, args.out, equity_curve=equity_curve, trade_rows=trade_rows, params=params, csv_delimiter=args.csv_delimiter)
    plot_chart(candles, events, args.symbol, args.out, equity_curve, show_chart=args.show, plot_candles=args.plot_candles)


if __name__ == "__main__":
    main()
