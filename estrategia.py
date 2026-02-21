#!/usr/bin/env python3
"""Visualizador standalone de estrategia RSI+MACD+Consecutivas.

No depende de `bot.py` para ejecutarse.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

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
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


@dataclass
class Metrics:
    net: float
    trades: int
    wins: int
    losses: int
    winrate: float
    pf: float
    dd_pct: float


class DotDict(dict):
    def __getattr__(self, item):
        return self[item]


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
    period = max(1, int(period))
    alpha = 2.0 / (period + 1.0)
    out = []
    ema = values[0]
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        out.append(float(ema))
    return out


def rsi_wilder_series(close: list[float], period: int) -> list[float]:
    n = max(2, len(close))
    p = max(2, int(period))
    out = [50.0] * n
    if n <= p:
        return out

    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        d = close[i] - close[i - 1]
        gains[i] = max(d, 0.0)
        losses[i] = max(-d, 0.0)

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
    ema_fast = ema_series(close, fast)
    ema_slow = ema_series(close, slow)
    macd_line = [a - b for a, b in zip(ema_fast, ema_slow)]
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
        ha_high = max(c["high"], ha_open, ha_close)
        ha_low = min(c["low"], ha_open, ha_close)
        row = dict(c)
        row.update({"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close})
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


def buy_rsi_signal(rsi_val: float, oversold: float, _overbought: float) -> float:
    return 1.0 if rsi_val <= oversold else 0.0


def sell_rsi_signal(rsi_val: float, _oversold: float, overbought: float) -> float:
    return 1.0 if rsi_val >= overbought else 0.0


def buy_macd_signal(hist_now: float, hist_prev: float, edge_trigger: int) -> float:
    return 1.0 if ((hist_prev <= 0.0 < hist_now) if edge_trigger else (hist_now > 0.0)) else 0.0


def sell_macd_signal(hist_now: float, hist_prev: float, edge_trigger: int) -> float:
    return 1.0 if ((hist_prev >= 0.0 > hist_now) if edge_trigger else (hist_now < 0.0)) else 0.0


def score_components_at_index(close: list[float], rsi_arr: list[float], mh_arr: list[float], i: int, ge) -> dict[str, float]:
    rsi_val = rsi_arr[i]
    mh = mh_arr[i]
    mh_prev = mh_arr[i - 1] if i > 0 else mh

    buy_rsi = buy_rsi_signal(rsi_val, ge.rsi_oversold, ge.rsi_overbought)
    sell_rsi = sell_rsi_signal(rsi_val, ge.rsi_oversold, ge.rsi_overbought)
    buy_macd = buy_macd_signal(mh, mh_prev, ge.edge_trigger)
    sell_macd = sell_macd_signal(mh, mh_prev, ge.edge_trigger)
    buy_consec = consec_up(close[: i + 1], ge.consec_green)
    sell_consec = consec_down(close[: i + 1], ge.consec_red)

    wbr, wbm, wbc = normalize3(ge.w_buy_rsi, ge.w_buy_macd, ge.w_buy_consec)
    wsr, wsm, wsc = normalize3(ge.w_sell_rsi, ge.w_sell_macd, ge.w_sell_consec)

    buy_score = wbr * buy_rsi + wbm * buy_macd + wbc * buy_consec
    sell_score = wsr * sell_rsi + wsm * sell_macd + wsc * sell_consec
    return {"buy_score": buy_score, "sell_score": sell_score}


def fetch_historical_closed(client, symbol: str, interval: str, start: str, end: str) -> list[dict[str, Any]]:
    kl = client.get_historical_klines(symbol=symbol, interval=interval, start_str=start, end_str=end)
    candles = [
        {
            "open_time": int(row[0]),
            "close_time": int(row[6]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        }
        for row in kl
    ]
    server_time_ms = int(client.get_server_time()["serverTime"])
    return filter_closed_candles(candles, interval, server_time_ms)


def load_params(args: argparse.Namespace) -> dict[str, Any]:
    params = dict(BEST_GEN)
    if args.params_json:
        with open(args.params_json, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    for kv in args.set:
        if "=" not in kv:
            raise ValueError(f"Override inválido: {kv}. Usa formato key=value")
        k, v = kv.split("=", 1)
        raw = v.strip()
        if raw.lower() in {"true", "false"}:
            params[k.strip()] = 1 if raw.lower() == "true" else 0
        else:
            try:
                params[k.strip()] = int(raw)
            except ValueError:
                params[k.strip()] = float(raw)
    return params


def build_genome(params: dict[str, Any]) -> DotDict:
    return DotDict(params)


def run_timeline(candles: list[dict[str, Any]], ge, fee_side: float, slip_side: float):
    data = heikin_ashi(candles) if int(ge.use_ha) == 1 else candles
    close_sig = [c["close"] for c in data]
    rsi_arr = rsi_wilder_series(close_sig, int(ge.rsi_period))
    mh_arr = macd_hist_series(close_sig, int(ge.macd_fast), int(ge.macd_slow), int(ge.macd_signal))

    in_pos = False
    entry = 0.0
    entry_i = -10_000
    last_trade_i = -10_000
    events: list[dict[str, Any]] = []
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for i in range(1, len(close_sig) - 1):
        payload = score_components_at_index(close_sig, rsi_arr, mh_arr, i, ge)
        bs = payload["buy_score"]
        ss = payload["sell_score"]

        next_candle = candles[i + 1]
        exec_open = float(next_candle["open"])

        if not in_pos:
            can_buy = (i - last_trade_i) >= int(ge.cooldown)
            if can_buy and bs >= float(ge.buy_th):
                buy_price = exec_open * (1.0 + slip_side)
                entry = buy_price
                entry_i = i + 1
                in_pos = True
                events.append({"type": "BUY", "i": i + 1, "time": next_candle["open_time"], "price": buy_price, "buy_score": bs, "sell_score": ss})
        else:
            tp_px = entry * (1.0 + abs(float(ge.take_profit)))
            sl_px = entry * (1.0 - abs(float(ge.stop_loss)))
            exit_reason = None
            px = exec_open
            if next_candle["low"] <= sl_px:
                exit_reason = "SL"
                px = sl_px
            elif next_candle["high"] >= tp_px:
                exit_reason = "TP"
                px = tp_px
            elif ss >= float(ge.sell_th):
                exit_reason = "SIG"
            if exit_reason:
                sell_price = px * (1.0 - slip_side)
                gross_ratio = sell_price / (entry + 1e-12)
                trade_ratio = (1.0 - fee_side) * gross_ratio
                trade_ret = trade_ratio - 1.0
                equity *= trade_ratio
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak)
                if trade_ret >= 0:
                    wins += 1
                    gross_profit += trade_ret
                else:
                    losses += 1
                    gross_loss += -trade_ret
                events.append({
                    "type": "SELL", "i": i + 1, "time": next_candle["open_time"], "price": sell_price,
                    "reason": exit_reason, "ret": trade_ret, "entry_i": entry_i, "entry_price": entry,
                    "buy_score": bs, "sell_score": ss,
                })
                in_pos = False
                last_trade_i = i + 1

    trades = wins + losses
    winrate = (wins / trades * 100.0) if trades else 0.0
    pf = gross_profit / (gross_loss + 1e-9) if trades else 0.0
    metrics = Metrics(net=equity - 1.0, trades=trades, wins=wins, losses=losses, winrate=winrate, pf=pf, dd_pct=max_dd * 100.0)
    return events, metrics


def ts_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def export_events(events: list[dict[str, Any]], metrics: Metrics, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    events_path = os.path.join(out_dir, "operaciones.csv")
    metrics_path = os.path.join(out_dir, "metricas.json")

    if events:
        keys = sorted({k for ev in events for k in ev.keys()})
        with open(events_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys + ["time_iso"])
            w.writeheader()
            for ev in events:
                row = dict(ev)
                row["time_iso"] = ts_to_iso(ev["time"])
                w.writerow(row)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)

    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(events)
        if not df.empty:
            df["time_iso"] = df["time"].apply(ts_to_iso)
        excel_path = os.path.join(out_dir, "reporte_estrategia.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            df.to_excel(writer, sheet_name="operaciones", index=False)
            pd.DataFrame([asdict(metrics)]).to_excel(writer, sheet_name="metricas", index=False)
        print(f"[OK] Excel exportado: {excel_path}")
    except Exception as exc:
        print(f"[WARN] Excel no disponible ({exc}). CSV/JSON sí fueron exportados.")

    print(f"[OK] Operaciones CSV: {events_path}")
    print(f"[OK] Métricas JSON: {metrics_path}")


def plot_chart(
    candles: list[dict[str, Any]],
    events: list[dict[str, Any]],
    symbol: str,
    out_dir: str,
    *,
    show_chart: bool,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:
        print(f"[WARN] No se pudo abrir gráfico (matplotlib no disponible): {exc}")
        return

    os.makedirs(out_dir, exist_ok=True)
    fig, (ax_price, ax_ret) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    width = 0.65
    for i, c in enumerate(candles):
        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
        color = "#2ecc71" if cl >= o else "#e74c3c"
        ax_price.vlines(i, l, h, color=color, linewidth=1)
        body_low = min(o, cl)
        body_h = max(abs(cl - o), 1e-9)
        rect = Rectangle((i - width / 2, body_low), width, body_h, facecolor=color, edgecolor=color, alpha=0.9)
        ax_price.add_patch(rect)

    buy_x, buy_y, sell_x, sell_y, ret_x, rets = [], [], [], [], [], []
    for ev in events:
        if ev["type"] == "BUY":
            buy_x.append(ev["i"])
            buy_y.append(ev["price"])
        else:
            sell_x.append(ev["i"])
            sell_y.append(ev["price"])
            if "ret" in ev:
                ret_x.append(ev["i"])
                rets.append(ev["ret"] * 100)

    if buy_x:
        ax_price.scatter(buy_x, buy_y, marker="^", s=35, color="#00b894", label="Entradas BUY")
    if sell_x:
        ax_price.scatter(sell_x, sell_y, marker="v", s=35, color="#d63031", label="Salidas SELL")

    ax_price.set_title(f"{symbol} | Estrategia (1h)")
    ax_price.set_ylabel("Precio")
    ax_price.grid(alpha=0.2)
    ax_price.legend(loc="upper left")

    ax_ret.axhline(0, color="gray", linewidth=1)
    if ret_x:
        colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in rets]
        ax_ret.bar(ret_x, rets, color=colors, alpha=0.85, width=0.8)
    ax_ret.set_ylabel("Ret %")
    ax_ret.set_xlabel("Velas")
    ax_ret.grid(alpha=0.2)

    png_path = os.path.join(out_dir, "grafico_estrategia.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    print(f"[OK] Gráfico guardado: {png_path}")
    if show_chart:
        try:
            plt.show()
            return
        except Exception as exc:
            print(f"[WARN] No se pudo abrir ventana interactiva del gráfico: {exc}")
    plt.close(fig)


def make_parser():
    p = argparse.ArgumentParser(description="Visualizador de estrategia trading (1h)")
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--interval", default="1h")
    p.add_argument("--out", default="estrategia_output")
    p.add_argument("--fee-side", type=float, default=0.001)
    p.add_argument("--slip-side", type=float, default=0.0005)
    p.add_argument("--params-json", default=None)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--show", dest="show", action="store_true", help="Abrir ventana del gráfico al finalizar")
    p.add_argument("--no-show", dest="show", action="store_false", help="No abrir ventana (solo guardar PNG)")
    p.set_defaults(show=True)
    return p


def main():
    args = make_parser().parse_args()
    interval_to_ms(args.interval)
    params = load_params(args)
    ge = build_genome(params)
    print("[INFO] Parámetros finales:")
    print(json.dumps(dict(ge), indent=2, ensure_ascii=False))

    try:
        from binance.client import Client
    except Exception as exc:
        raise RuntimeError("Falta dependencia python-binance. Instala con: pip install python-binance") from exc

    client = Client(None, None)
    candles = fetch_historical_closed(client, args.symbol, args.interval, args.start, args.end)
    if len(candles) < 220:
        raise RuntimeError(f"Muy pocas velas cerradas para simular: {len(candles)}")

    events, metrics = run_timeline(candles, ge, args.fee_side, args.slip_side)
    print(f"[RESUMEN] trades={metrics.trades} winrate={metrics.winrate:.2f}% pf={metrics.pf:.2f} net={metrics.net:.4f} dd={metrics.dd_pct:.2f}%")
    export_events(events, metrics, args.out)
    plot_chart(candles, events, args.symbol, args.out, show_chart=args.show)


if __name__ == "__main__":
    main()
