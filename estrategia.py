#!/usr/bin/env python3
"""Visualizador profesional de estrategia (1h) con export de operaciones.

- Usa la MISMA lógica de señales del bot (`score_components_at_index`)
- Permite cargar/modificar parámetros desde JSON/CLI
- Descarga velas históricas de Binance por rango de fechas
- Grafica precio tipo vela + entradas/salidas
- Exporta operaciones/metricas a CSV y (si hay pandas/openpyxl) también a XLSX
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
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


def parse_ts(text: str) -> int:
    dt = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ts_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fetch_historical_closed(
    client,
    symbol: str,
    interval: str,
    start: str,
    end: str,
) -> list[dict[str, Any]]:
    from bot import filter_closed_candles
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
        k = k.strip()
        raw = v.strip()
        if raw.lower() in {"true", "false"}:
            val: Any = 1 if raw.lower() == "true" else 0
        else:
            try:
                val = int(raw)
            except ValueError:
                val = float(raw)
        params[k] = val
    return params


def build_genome(params: dict[str, Any]):
    from bot import Genome, ParamSpace
    space = ParamSpace()
    ge = Genome(**params)
    return space.clamp_genome(ge)


def run_timeline(candles: list[dict[str, Any]], ge, fee_side: float, slip_side: float):
    from bot import heikin_ashi, macd_hist_series, rsi_wilder_series, score_components_at_index
    data = heikin_ashi(candles) if ge.use_ha == 1 else candles
    close_sig = [c["close"] for c in data]
    rsi_arr = rsi_wilder_series(close_sig, ge.rsi_period)
    mh_arr = macd_hist_series(close_sig, ge.macd_fast, ge.macd_slow, ge.macd_signal)

    in_pos = False
    entry = 0.0
    entry_i = -10_000
    last_trade_i = -10_000
    events: list[dict[str, Any]] = []

    for i in range(1, len(close_sig) - 1):
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

        next_candle = candles[i + 1]
        exec_open = float(next_candle["open"])

        if not in_pos:
            can_buy = (i - last_trade_i) >= ge.cooldown
            if can_buy and bs >= ge.buy_th:
                buy_price = exec_open * (1.0 + slip_side)
                entry = buy_price
                entry_i = i + 1
                in_pos = True
                events.append(
                    {
                        "type": "BUY",
                        "i": i + 1,
                        "time": next_candle["open_time"],
                        "price": buy_price,
                        "buy_score": bs,
                        "sell_score": ss,
                    }
                )
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
            elif ss >= ge.sell_th:
                exit_reason = "SIG"
                px = exec_open

            if exit_reason:
                sell_price = px * (1.0 - slip_side)
                gross_ratio = sell_price / (entry + 1e-12)
                trade_ratio = (1.0 - fee_side) * gross_ratio
                trade_ret = trade_ratio - 1.0
                events.append(
                    {
                        "type": "SELL",
                        "i": i + 1,
                        "time": next_candle["open_time"],
                        "price": sell_price,
                        "reason": exit_reason,
                        "ret": trade_ret,
                        "entry_i": entry_i,
                        "entry_price": entry,
                        "buy_score": bs,
                        "sell_score": ss,
                    }
                )
                in_pos = False
                last_trade_i = i + 1

    return events


def export_events(events: list[dict[str, Any]], metrics, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    events_path = os.path.join(out_dir, "operaciones.csv")
    metrics_path = os.path.join(out_dir, "metricas.json")

    if events:
        keys = sorted({k for ev in events for k in ev.keys()})
        with open(events_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
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


def plot_chart(candles: list[dict[str, Any]], events: list[dict[str, Any]], symbol: str, out_dir: str):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:
        print(f"[WARN] No se pudo abrir gráfico (matplotlib no disponible): {exc}")
        return

    os.makedirs(out_dir, exist_ok=True)
    x = list(range(len(candles)))

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

    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    rets = []
    ret_x = []
    for ev in events:
        if ev["type"] == "BUY":
            buy_x.append(ev["i"])
            buy_y.append(ev["price"])
        elif ev["type"] == "SELL":
            sell_x.append(ev["i"])
            sell_y.append(ev["price"])
            if "ret" in ev:
                ret_x.append(ev["i"])
                rets.append(ev["ret"] * 100)

    if buy_x:
        ax_price.scatter(buy_x, buy_y, marker="^", s=40, color="#00b894", label="Entradas BUY")
    if sell_x:
        ax_price.scatter(sell_x, sell_y, marker="v", s=40, color="#d63031", label="Salidas SELL")

    ax_price.set_title(f"{symbol} | Estrategia RSI+MACD+Consecutivas (1h)")
    ax_price.set_ylabel("Precio")
    ax_price.grid(alpha=0.2)
    ax_price.legend(loc="upper left")

    ax_ret.axhline(0, color="gray", linewidth=1)
    if ret_x:
        colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in rets]
        ax_ret.bar(ret_x, rets, color=colors, alpha=0.85, width=0.8)
    ax_ret.set_ylabel("Ret %")
    ax_ret.set_xlabel("Velas (índice cronológico)")
    ax_ret.grid(alpha=0.2)

    png_path = os.path.join(out_dir, "grafico_estrategia.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    print(f"[OK] Gráfico guardado: {png_path}")

    if os.environ.get("DISPLAY"):
        plt.show()
    else:
        plt.close(fig)


def make_parser():
    p = argparse.ArgumentParser(description="Visualizador de estrategia trading (1h)")
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--interval", default="1h", help="Timeframe (recomendado: 1h)")
    p.add_argument("--out", default="estrategia_output")
    p.add_argument("--fee-side", type=float, default=0.001)
    p.add_argument("--slip-side", type=float, default=0.0005)
    p.add_argument("--params-json", default=None, help="Archivo JSON con parámetros")
    p.add_argument("--set", action="append", default=[], help="Overrides key=value, repetible")
    return p


def main():
    args = make_parser().parse_args()
    if args.interval != "1h":
        print(f"[WARN] Estás usando interval={args.interval}. El modo recomendado es 1h.")

    from bot import interval_to_ms, simulate_spot
    from binance.client import Client

    interval_to_ms(args.interval)  # valida intervalo soportado

    params = load_params(args)
    ge = build_genome(params)

    print("[INFO] Parámetros finales:")
    print(json.dumps(asdict(ge), indent=2, ensure_ascii=False))

    client = Client(None, None)
    candles = fetch_historical_closed(client, args.symbol, args.interval, args.start, args.end)
    if len(candles) < 220:
        raise RuntimeError(f"Muy pocas velas cerradas para simular: {len(candles)}")

    metrics, _trace = simulate_spot(candles, ge, args.fee_side, args.slip_side, trace=True)
    events = run_timeline(candles, ge, args.fee_side, args.slip_side)

    sell_events = [e for e in events if e["type"] == "SELL"]
    if abs(len(sell_events) - metrics.trades) > 1:
        print(
            f"[WARN] Diferencia detectada entre timeline SELL={len(sell_events)} y metrics.trades={metrics.trades}."
        )

    print(
        f"[RESUMEN] trades={metrics.trades} winrate={metrics.winrate:.2f}% "
        f"pf={metrics.pf:.2f} net={metrics.net:.4f} dd={metrics.dd_pct:.2f}%"
    )

    export_events(events, metrics, args.out)
    plot_chart(candles, events, args.symbol, args.out)


if __name__ == "__main__":
    main()
