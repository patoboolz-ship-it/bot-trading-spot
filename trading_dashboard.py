import importlib
import importlib.util
import math
import queue
import threading
import time
import tkinter as tk
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Callable, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

API_KEY_PATH = r"C:\Users\EQCONF\Documents\programas python\claves\bot trading\Clave API.txt"
API_SECRET_PATH = r"C:\Users\EQCONF\Documents\programas python\claves\bot trading\Clave secreta.txt"

BEST_GEN = {
    "use_ha": 0,
    "rsi_period": 18,
    "rsi_oversold": 30.0,
    "rsi_overbought": 71.0,
    "macd_fast": 9,
    "macd_slow": 146,
    "macd_signal": 27,
    "consec_red": 3,
    "consec_green": 5,
    "w_buy_rsi": 0.38,
    "w_buy_macd": 0.31,
    "w_buy_consec": 0.31,
    "buy_th": 0.55,
    "w_sell_rsi": 0.33,
    "w_sell_macd": 0.15,
    "w_sell_consec": 0.52,
    "sell_th": 0.60,
    "take_profit": 0.150,
    "stop_loss": 0.050,
    "cooldown": 11,
    "edge_trigger": 0,
}

INTERVAL = "1h"
INTERVAL_MS = {"1h": 3_600_000}


def add_months(dt: datetime, months: int) -> datetime:
    m = dt.month - 1 + months
    y = dt.year + m // 12
    m = m % 12 + 1
    # Evitar días inválidos (31,30,29)
    day = min(dt.day, [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return dt.replace(year=y, month=m, day=day)


def ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


def ema(vals: list[float], period: int) -> list[float]:
    if not vals:
        return []
    a = 2 / (period + 1)
    out = [vals[0]]
    for i in range(1, len(vals)):
        out.append(a * vals[i] + (1 - a) * out[-1])
    return out


def rsi(close: list[float], period: int) -> list[float]:
    if not close:
        return []
    if len(close) < 2:
        return [50.0]
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(close)):
        ch = close[i] - close[i - 1]
        gains.append(max(0.0, ch))
        losses.append(max(0.0, -ch))

    period = max(2, int(period))
    avg_gain = sum(gains[1 : period + 1]) / period if len(gains) > period else sum(gains[1:]) / max(1, len(gains) - 1)
    avg_loss = sum(losses[1 : period + 1]) / period if len(losses) > period else sum(losses[1:]) / max(1, len(losses) - 1)

    out = [50.0] * len(close)
    for i in range(1, len(close)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = (avg_gain / avg_loss) if avg_loss > 1e-12 else math.inf
        out[i] = 100.0 - (100.0 / (1.0 + rs)) if rs != math.inf else 100.0
    return out


def macd_hist(close: list[float], fast: int, slow: int, signal: int) -> list[float]:
    f = ema(close, max(2, int(fast)))
    s = ema(close, max(3, int(slow)))
    macd = [f[i] - s[i] for i in range(len(close))]
    sig = ema(macd, max(2, int(signal)))
    return [macd[i] - sig[i] for i in range(len(close))]


def heikin_ashi(candles: list[dict]) -> list[dict]:
    if not candles:
        return []
    out = []
    prev_ho = float(candles[0]["open"])
    prev_hc = float(candles[0]["close"])
    for i, c in enumerate(candles):
        o = float(c["open"])
        h = float(c["high"])
        l = float(c["low"])
        cl = float(c["close"])
        hc = (o + h + l + cl) / 4.0
        ho = (o + cl) / 2.0 if i == 0 else (prev_ho + prev_hc) / 2.0
        hh = max(h, ho, hc)
        hl = min(l, ho, hc)
        prev_ho, prev_hc = ho, hc
        out.append({**c, "open": ho, "high": hh, "low": hl, "close": hc, "ha_close": hc})
    return out


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def buy_rsi_signal(rsi_val: float, os: float, ob: float) -> float:
    os2 = min(os, ob - 1e-6)
    ob2 = max(ob, os2 + 1e-6)
    if rsi_val <= os2:
        return 1.0
    if rsi_val >= ob2:
        return 0.0
    return clamp01((ob2 - rsi_val) / (ob2 - os2))


def sell_rsi_signal(rsi_val: float, os: float, ob: float) -> float:
    os2 = min(os, ob - 1e-6)
    ob2 = max(ob, os2 + 1e-6)
    if rsi_val >= ob2:
        return 1.0
    if rsi_val <= os2:
        return 0.0
    return clamp01((rsi_val - os2) / (ob2 - os2))


def buy_macd_signal(hist_now: float, hist_prev: float, edge_trigger: int) -> float:
    return 1.0 if ((hist_prev <= 0 and hist_now > 0) if edge_trigger else (hist_now > 0)) else 0.0


def sell_macd_signal(hist_now: float, hist_prev: float, edge_trigger: int) -> float:
    return 1.0 if ((hist_prev >= 0 and hist_now < 0) if edge_trigger else (hist_now < 0)) else 0.0


def consec_up(close: list[float], n: int) -> float:
    if n <= 0 or len(close) < n + 1:
        return 0.0
    for i in range(n):
        if not (close[-1 - i] >= close[-2 - i]):
            return 0.0
    return 1.0


def consec_down(close: list[float], n: int) -> float:
    if n <= 0 or len(close) < n + 1:
        return 0.0
    for i in range(n):
        if not (close[-1 - i] <= close[-2 - i]):
            return 0.0
    return 1.0


def normalize3(a: float, b: float, c: float) -> tuple[float, float, float]:
    s = a + b + c
    if s <= 1e-12:
        return (1 / 3, 1 / 3, 1 / 3)
    return (a / s, b / s, c / s)


def is_candle_closed(close_time_ms: int, server_time_ms: int, interval_ms: int) -> bool:
    return int(close_time_ms) < int(server_time_ms) - int(interval_ms)


@dataclass
class SimOp:
    side: str
    index: int
    time_ms: int
    price: float
    qty: float
    equity_after: float
    reason: str


class _BinancePublicFallbackAPI:
    BASE_URL = "https://api.binance.com"

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._running = False
        self._cache_last_candles: list[dict[str, Any]] = []

    def _get_json(self, path: str, params: dict[str, Any]):
        import json
        import urllib.parse
        import urllib.request

        qs = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f"{self.BASE_URL}{path}?{qs}", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_server_time_ms(self) -> int:
        row = self._get_json("/api/v3/time", {})
        return int(row["serverTime"])

    def get_balance(self):
        return {"USDT": 0.0, "asset_qty": 0.0}

    def get_price(self, symbol: str):
        row = self._get_json("/api/v3/ticker/price", {"symbol": symbol})
        return float(row["price"])

    def get_trades(self):
        return []

    def get_candles(self, symbol: str, interval: str):
        rows = self._get_json("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": 600})
        candles = [
            {
                "open_time": int(r[0]),
                "close_time": int(r[6]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            }
            for r in rows
        ]
        server_ms = self.get_server_time_ms()
        out = [c for c in candles if is_candle_closed(c["close_time"], server_ms, INTERVAL_MS[INTERVAL])]
        self._cache_last_candles = out
        return out

    def get_candles_range(self, symbol: str, interval: str, start_ms: int, end_ms: int):
        out = []
        cur = int(start_ms)
        end_ms = int(end_ms)
        while cur <= end_ms:
            rows = self._get_json(
                "/api/v3/klines",
                {"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": 1000},
            )
            if not rows:
                break
            for r in rows:
                out.append(
                    {
                        "open_time": int(r[0]),
                        "close_time": int(r[6]),
                        "open": float(r[1]),
                        "high": float(r[2]),
                        "low": float(r[3]),
                        "close": float(r[4]),
                        "volume": float(r[5]),
                    }
                )
            last_open = int(rows[-1][0])
            if last_open <= cur:
                break
            cur = last_open + INTERVAL_MS[INTERVAL]
            if len(rows) < 1000:
                break
        server_ms = self.get_server_time_ms()
        filtered = [c for c in out if is_candle_closed(c["close_time"], server_ms, INTERVAL_MS[INTERVAL])]
        filtered.sort(key=lambda x: int(x.get("open_time", 0)))
        dedup = []
        seen = set()
        for c in filtered:
            ot = int(c.get("open_time", 0))
            if ot in seen:
                continue
            seen.add(ot)
            dedup.append(c)
        self._cache_last_candles = dedup
        return dedup

    def start_bot(self):
        self._running = True

    def pause_bot(self):
        self._running = False

    def stop_bot(self):
        self._running = False


class _BotModuleAdapter:
    def __init__(self, bot_mod: Any, symbol: str):
        self.bot_mod = bot_mod
        self.symbol = symbol
        self.client = self._build_client()
        self._bot = None
        self._params = getattr(bot_mod, "DEFAULT_GEN", {}).copy() if hasattr(bot_mod, "DEFAULT_GEN") else BEST_GEN.copy()

    def _build_client(self):
        key_path = getattr(self.bot_mod, "API_KEY_PATH", API_KEY_PATH)
        sec_path = getattr(self.bot_mod, "API_SECRET_PATH", API_SECRET_PATH)
        key = self.bot_mod.read_key(key_path)
        sec = self.bot_mod.read_key(sec_path)
        return self.bot_mod.Client(key, sec)

    def get_server_time_ms(self) -> int:
        return int(self.client.get_server_time()["serverTime"])

    def get_balance(self):
        quote = getattr(self.bot_mod, "QUOTE_ASSET", "USDT")
        base = getattr(self.bot_mod, "BASE_ASSET", self.symbol.replace(quote, ""))
        return {
            "USDT": float(self.bot_mod.get_free_balance(self.client, quote)),
            "asset_qty": float(self.bot_mod.get_free_balance(self.client, base)),
        }

    def get_price(self, symbol: str):
        return float(self.client.get_symbol_ticker(symbol=symbol)["price"])

    def get_trades(self):
        # Evitamos ruido de market trades; las operaciones visuales salen de la simulación de estrategia.
        return []

    def get_candles(self, symbol: str, interval: str):
        return self.bot_mod.fetch_klines_closed(self.client, symbol, interval, limit=600)

    def get_candles_range(self, symbol: str, interval: str, start_ms: int, end_ms: int):
        out = []
        cur = int(start_ms)
        end_ms = int(end_ms)
        while cur <= end_ms:
            rows = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=cur,
                endTime=end_ms,
                limit=1000,
            )
            if not rows:
                break
            for r in rows:
                out.append(
                    {
                        "open_time": int(r[0]),
                        "close_time": int(r[6]),
                        "open": float(r[1]),
                        "high": float(r[2]),
                        "low": float(r[3]),
                        "close": float(r[4]),
                        "volume": float(r[5]),
                    }
                )
            last_open = int(rows[-1][0])
            if last_open <= cur:
                break
            cur = last_open + INTERVAL_MS[INTERVAL]
            if len(rows) < 1000:
                break
        server_ms = self.get_server_time_ms()
        filtered = [c for c in out if is_candle_closed(c["close_time"], server_ms, INTERVAL_MS[INTERVAL])]
        filtered.sort(key=lambda x: int(x.get("open_time", 0)))
        dedup = []
        seen = set()
        for c in filtered:
            ot = int(c.get("open_time", 0))
            if ot in seen:
                continue
            seen.add(ot)
            dedup.append(c)
        return dedup

    def start_bot(self):
        if self._bot and getattr(self._bot, "running", False):
            return
        self._bot = self.bot_mod.SpotBot(self.client, self.symbol, INTERVAL, self._params, ui_cb=None)
        self._bot.start()

    def pause_bot(self):
        if self._bot:
            self._bot.stop()

    def stop_bot(self):
        if self._bot:
            self._bot.stop()


def _import_bot_module_from_candidates() -> Any | None:
    try:
        return importlib.import_module("bot")
    except Exception:
        pass

    for c in [Path.cwd() / "bot.py", Path(__file__).resolve().parent / "bot.py"]:
        if not c.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("bot_local", c)
            if not spec or not spec.loader:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception:
            continue
    return None


class TradingDashboard(tk.Tk):
    def __init__(self, symbol: str = "SOLUSDT"):
        super().__init__()
        self.title("Trading Dashboard - Mini Exchange")
        self.geometry("1600x980")

        self.symbol = symbol
        self.interval = INTERVAL
        self.params = BEST_GEN.copy()

        self.balance = 0.0
        self.equity = 0.0
        self.pnl = 0.0
        self.price = 0.0
        self.trades: list[dict[str, Any]] = []
        self.candles: list[dict[str, Any]] = []
        self.bot_running = False

        self.asset_qty = 0.0
        self.initial_equity = 1000.0
        self.pnl_percent = 0.0
        self.status_text = "IDLE"
        self.last_error = ""

        self.strategy_ops: list[SimOp] = []
        self.strategy_equity_curve: list[tuple[int, float]] = []
        self.period_rows: list[dict[str, Any]] = []
        self.active_candles: list[dict[str, Any]] = []

        self.date_from_var = tk.StringVar(value="")
        self.date_to_var = tk.StringVar(value="")
        self.range_locked = False
        self.range_loading = False
        self.selected_start_ms: Optional[int] = None
        self.selected_end_ms: Optional[int] = None
        self.chart_segments: list[tuple[int, int]] = []
        self.current_segment_idx = 0
        self._last_candles_sig: tuple[int, int, float] = (0, 0, 0.0)

        self._api, self._api_mode = self._resolve_api_functions()
        self._ui_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        self._build_layout()
        self._load_initial_data()

        self.after(150, self._process_ui_queue)
        self._thread = threading.Thread(target=self.update_loop, daemon=True)
        self._thread.start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _resolve_api_functions(self) -> tuple[dict[str, Callable[..., Any]], str]:
        required = ["get_balance", "get_price", "get_trades", "get_candles", "start_bot", "pause_bot"]
        optional = ["stop_bot"]
        resolved: dict[str, Callable[..., Any]] = {}

        for module_name in ["bot", "api", "trading_api"]:
            try:
                mod = importlib.import_module(module_name)
            except Exception:
                continue
            for fn in required + optional:
                if fn not in resolved and hasattr(mod, fn) and callable(getattr(mod, fn)):
                    resolved[fn] = getattr(mod, fn)

        if all(name in resolved for name in required):
            if "stop_bot" not in resolved:
                resolved["stop_bot"] = resolved["pause_bot"]
            return resolved, "direct"

        bot_mod = _import_bot_module_from_candidates()
        if bot_mod and all(hasattr(bot_mod, n) for n in ["read_key", "Client", "fetch_klines_closed", "get_free_balance", "SpotBot"]):
            self.symbol = getattr(bot_mod, "SYMBOL", self.symbol)
            adapter = _BotModuleAdapter(bot_mod, self.symbol)
            return {
                "get_balance": adapter.get_balance,
                "get_price": adapter.get_price,
                "get_trades": adapter.get_trades,
                "get_candles": adapter.get_candles,
                "get_candles_range": adapter.get_candles_range,
                "start_bot": adapter.start_bot,
                "pause_bot": adapter.pause_bot,
                "stop_bot": adapter.stop_bot,
            }, "bot_adapter"

        fb = _BinancePublicFallbackAPI(self.symbol)
        return {
            "get_balance": fb.get_balance,
            "get_price": fb.get_price,
            "get_trades": fb.get_trades,
            "get_candles": fb.get_candles,
            "get_candles_range": fb.get_candles_range,
            "start_bot": fb.start_bot,
            "pause_bot": fb.pause_bot,
            "stop_bot": fb.stop_bot,
        }, "public_fallback"

    def _build_layout(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        top = ttk.Frame(self, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        for c in range(6):
            top.grid_columnconfigure(c, weight=1)

        self.lbl_balance = ttk.Label(top, text="Balance USDT: 0.00", font=("Segoe UI", 11, "bold"))
        self.lbl_equity = ttk.Label(top, text="Equity: 0.00", font=("Segoe UI", 11, "bold"))
        self.lbl_pnl = ttk.Label(top, text="PnL: 0.00 (0.00%)", font=("Segoe UI", 11, "bold"))
        self.lbl_price = ttk.Label(top, text=f"Precio {self.symbol}: 0.00", font=("Segoe UI", 11, "bold"))
        self.lbl_state = ttk.Label(top, text="Estado: IDLE", font=("Segoe UI", 11, "bold"))
        self.lbl_error = ttk.Label(top, text="", foreground="#b71c1c")

        self.lbl_balance.grid(row=0, column=0, sticky="w")
        self.lbl_equity.grid(row=0, column=1, sticky="w")
        self.lbl_pnl.grid(row=0, column=2, sticky="w")
        self.lbl_price.grid(row=0, column=3, sticky="w")
        self.lbl_state.grid(row=0, column=4, sticky="w")
        self.lbl_error.grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        tab_dash = self._make_scrollable_tab("Dashboard 1H")
        tab_perf = self._make_scrollable_tab("Rentabilidades")
        tab_bot = self._make_scrollable_tab("Bot")

        tab_dash.grid_rowconfigure(0, weight=1)
        tab_dash.grid_columnconfigure(0, weight=4)
        tab_dash.grid_columnconfigure(1, weight=1)

        chart_frame = ttk.LabelFrame(tab_dash, text="Gráfico de Velas + Operaciones (estrategia)", padding=8)
        chart_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        chart_frame.grid_rowconfigure(0, weight=1)
        chart_frame.grid_columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        nav = ttk.Frame(chart_frame)
        nav.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.btn_prev_chart = ttk.Button(nav, text="◀ Anterior", command=self.show_prev_chart)
        self.btn_prev_chart.pack(side="left", padx=4)
        self.btn_next_chart = ttk.Button(nav, text="Siguiente ▶", command=self.show_next_chart)
        self.btn_next_chart.pack(side="left", padx=4)
        self.lbl_chart_page = ttk.Label(nav, text="Gráfico 0/0")
        self.lbl_chart_page.pack(side="left", padx=10)

        trades_frame = ttk.LabelFrame(tab_dash, text="Operaciones de la Estrategia (1h, cierre de vela)", padding=8)
        trades_frame.grid(row=0, column=1, sticky="nsew")
        trades_frame.grid_rowconfigure(0, weight=1)
        trades_frame.grid_columnconfigure(0, weight=1)

        self.trades_listbox = tk.Listbox(trades_frame, font=("Consolas", 10))
        self.trades_listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(trades_frame, orient="vertical", command=self.trades_listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.trades_listbox.configure(yscrollcommand=scroll.set)

        # Tab rentabilidades
        tab_perf.grid_rowconfigure(1, weight=1)
        tab_perf.grid_columnconfigure(0, weight=1)

        btns = ttk.Frame(tab_perf)
        btns.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(btns, text="Desde (YYYY/MM/DD):").pack(side="left", padx=(2, 4))
        ttk.Entry(btns, textvariable=self.date_from_var, width=14).pack(side="left", padx=(0, 8))
        ttk.Label(btns, text="Hasta (YYYY/MM/DD):").pack(side="left", padx=(2, 4))
        ttk.Entry(btns, textvariable=self.date_to_var, width=14).pack(side="left", padx=(0, 8))
        self.btn_apply_range = ttk.Button(btns, text="Aplicar Rango", command=self.apply_date_range)
        self.btn_apply_range.pack(side="left", padx=4)
        self.btn_clear_range = ttk.Button(btns, text="Limpiar Rango", command=self.clear_date_range)
        self.btn_clear_range.pack(side="left", padx=4)
        self.btn_recalc = ttk.Button(btns, text="Recalcular Rentabilidades", command=self._recompute_strategy_views)
        self.btn_recalc.pack(side="left", padx=4)
        ttk.Button(btns, text="Exportar Excel", command=self.export_excel).pack(side="left", padx=4)

        cols = ("periodo", "operaciones", "roi_pct", "pnl_usdt", "equity_final")
        self.tree_periods = ttk.Treeview(tab_perf, columns=cols, show="headings", height=8)
        for c in cols:
            self.tree_periods.heading(c, text=c.upper())
            self.tree_periods.column(c, anchor="center", width=140)
        self.tree_periods.grid(row=1, column=0, sticky="nsew")

        chart_perf_frame = ttk.LabelFrame(tab_perf, text="Curva de Equity de la Estrategia", padding=8)
        chart_perf_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        chart_perf_frame.grid_rowconfigure(0, weight=1)
        chart_perf_frame.grid_columnconfigure(0, weight=1)
        tab_perf.grid_rowconfigure(2, weight=2)

        self.figure_perf = Figure(figsize=(9, 3.8), dpi=100)
        self.ax_perf = self.figure_perf.add_subplot(111)
        self.canvas_perf = FigureCanvasTkAgg(self.figure_perf, master=chart_perf_frame)
        self.canvas_perf.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Tab BOT integrado con el bot real (si está disponible)
        self._build_bot_tab(tab_bot)

        bottom = ttk.Frame(self, padding=8)
        bottom.grid(row=2, column=0, sticky="ew")
        self.btn_play = ttk.Button(bottom, text="PLAY", command=self.start_bot)
        self.btn_pause = ttk.Button(bottom, text="PAUSE", command=self.pause_bot)
        self.btn_stop = ttk.Button(bottom, text="STOP", command=self.stop_bot)
        self.btn_play.pack(side="left", padx=6)
        self.btn_pause.pack(side="left", padx=6)
        self.btn_stop.pack(side="left", padx=6)

    def _make_scrollable_tab(self, title: str, *, padding: int = 6) -> ttk.Frame:
        outer = ttk.Frame(self.notebook)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        yscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        xscroll = ttk.Scrollbar(outer, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        inner = ttk.Frame(canvas, padding=padding)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(win, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        self.notebook.add(outer, text=title)
        return inner

    def _load_initial_data(self) -> None:
        if self._api_mode == "public_fallback":
            self._set_error("Modo fallback: no se encontró bot.py local. Se usa mercado público + simulación estrategia (sin cuenta real).")
        try:
            self.balance, self.asset_qty = self._parse_balance(self._api["get_balance"]())
            self.price = float(self._api["get_price"](self.symbol))
            self.candles = self._api["get_candles"](self.symbol, self.interval) or []
            self._recompute_strategy_views()
            self._refresh_ui()
        except Exception as exc:
            self._set_error(f"Error inicializando: {exc}")

    def _parse_balance(self, raw: Any) -> tuple[float, float]:
        if isinstance(raw, tuple) and len(raw) >= 2:
            return float(raw[0]), float(raw[1])
        if isinstance(raw, dict):
            return float(raw.get("USDT", raw.get("balance_usdt", 0.0))), float(raw.get("asset_qty", 0.0))
        return float(raw), 0.0

    def _set_error(self, msg: str) -> None:
        self.last_error = msg
        self.lbl_error.config(text=msg)

    def _refresh_ui(self) -> None:
        self.lbl_balance.config(text=f"Balance USDT: {self.balance:,.4f}")
        self.lbl_equity.config(text=f"Equity: {self.equity:,.4f}")
        color = "#2e7d32" if self.pnl >= 0 else "#c62828"
        self.lbl_pnl.config(text=f"PnL: {self.pnl:,.4f} ({self.pnl_percent:.2f}%)", foreground=color)
        self.lbl_price.config(text=f"Precio {self.symbol}: {self.price:,.4f}")
        self.lbl_state.config(text=f"Estado: {self.status_text}")

    def _compute_strategy_ops(self, candles: list[dict]) -> tuple[list[SimOp], list[tuple[int, float]]]:
        if not candles:
            return [], []

        src = heikin_ashi(candles) if int(self.params["use_ha"]) == 1 else candles
        close = [float(c["close"]) for c in src]
        rsi_arr = rsi(close, int(self.params["rsi_period"]))
        mh_arr = macd_hist(close, int(self.params["macd_fast"]), int(self.params["macd_slow"]), int(self.params["macd_signal"]))

        wbr, wbm, wbc = normalize3(float(self.params["w_buy_rsi"]), float(self.params["w_buy_macd"]), float(self.params["w_buy_consec"]))
        wsr, wsm, wsc = normalize3(float(self.params["w_sell_rsi"]), float(self.params["w_sell_macd"]), float(self.params["w_sell_consec"]))

        capital = float(self.initial_equity)
        qty = 0.0
        in_pos = False
        last_buy_idx = -10_000

        ops: list[SimOp] = []
        curve: list[tuple[int, float]] = []

        for i in range(len(src)):
            px = float(src[i]["close"])
            mh_prev = mh_arr[i - 1] if i > 0 else mh_arr[i]
            buy_score = (
                wbr * buy_rsi_signal(rsi_arr[i], float(self.params["rsi_oversold"]), float(self.params["rsi_overbought"]))
                + wbm * buy_macd_signal(mh_arr[i], mh_prev, int(self.params["edge_trigger"]))
                + wbc * consec_up(close[: i + 1], int(self.params["consec_green"]))
            )
            sell_score = (
                wsr * sell_rsi_signal(rsi_arr[i], float(self.params["rsi_oversold"]), float(self.params["rsi_overbought"]))
                + wsm * sell_macd_signal(mh_arr[i], mh_prev, int(self.params["edge_trigger"]))
                + wsc * consec_down(close[: i + 1], int(self.params["consec_red"]))
            )

            if not in_pos and buy_score >= float(self.params["buy_th"]) and (i - last_buy_idx) >= int(self.params["cooldown"]):
                qty = capital / px if px > 0 else 0.0
                in_pos = qty > 0
                if in_pos:
                    ops.append(SimOp("BUY", i, int(src[i].get("close_time", 0)), px, qty, capital, "SIG"))
                    last_buy_idx = i
            elif in_pos:
                entry = ops[-1].price if ops else px
                ret = (px - entry) / entry if entry > 0 else 0.0
                reason = None
                if ret >= float(self.params["take_profit"]):
                    reason = "TP"
                elif ret <= -float(self.params["stop_loss"]):
                    reason = "SL"
                elif sell_score >= float(self.params["sell_th"]):
                    reason = "SIG"

                if reason is not None:
                    capital = qty * px
                    ops.append(SimOp("SELL", i, int(src[i].get("close_time", 0)), px, qty, capital, reason))
                    qty = 0.0
                    in_pos = False

            eq = qty * px if in_pos else capital
            curve.append((int(src[i].get("close_time", i)), eq))

        return ops, curve

    def _candles_signature(self, candles: list[dict]) -> tuple[int, int, float]:
        if not candles:
            return (0, 0, 0.0)
        last = candles[-1]
        return (
            len(candles),
            int(last.get("close_time", last.get("open_time", 0))),
            float(last.get("close", 0.0)),
        )

    def _recompute_strategy_views(self) -> None:
        self.active_candles = self._get_candles_in_selected_range(self.candles)
        self.strategy_ops, self.strategy_equity_curve = self._compute_strategy_ops(self.active_candles)
        if self.strategy_equity_curve:
            self.equity = float(self.strategy_equity_curve[-1][1])
            self.pnl = self.equity - self.initial_equity
            self.pnl_percent = (self.pnl / self.initial_equity) * 100 if self.initial_equity else 0.0
        self._render_ops_listbox()
        self._build_chart_segments()
        self.current_segment_idx = 0
        self._last_candles_sig: tuple[int, int, float] = (0, 0, 0.0)
        self._render_current_segment_chart()
        self._refresh_period_table()
        self._update_perf_chart()
        self._refresh_bot_tab()
        self._last_candles_sig = self._candles_signature(self.candles)

    def _build_chart_segments(self) -> None:
        self.chart_segments = []
        n = len(self.active_candles)
        if n == 0:
            self.lbl_chart_page.config(text="Gráfico 0/0")
            return

        # Segmentación por trimestre calendario del rango seleccionado.
        # Ejemplo: 2025/03/03 -> 2025/08/01 => [03/03-06/03], [06/03-08/01]
        start_ms = int(self.selected_start_ms or self.active_candles[0].get("close_time", 0))
        end_ms = int(self.selected_end_ms or self.active_candles[-1].get("close_time", 0))
        start_dt = ms_to_utc(start_ms)
        end_dt = ms_to_utc(end_ms)

        windows: list[tuple[int, int]] = []
        cur_start = start_dt
        while cur_start < end_dt:
            nxt = add_months(cur_start, 3)
            if nxt > end_dt:
                nxt = end_dt
            windows.append((int(cur_start.timestamp() * 1000), int(nxt.timestamp() * 1000)))
            if nxt <= cur_start:
                break
            cur_start = nxt

        if not windows:
            windows = [(start_ms, end_ms)]

        for w_start, w_end in windows:
            idxs = [
                i for i, c in enumerate(self.active_candles)
                if w_start <= int(c.get("close_time", c.get("open_time", 0))) <= w_end
            ]
            if idxs:
                self.chart_segments.append((idxs[0], idxs[-1]))

        if not self.chart_segments:
            self.chart_segments = [(0, n - 1)]

        self.lbl_chart_page.config(text=f"Gráfico 1/{len(self.chart_segments)}")

    def _render_current_segment_chart(self) -> None:
        if not self.chart_segments:
            self.update_chart([])
            self.lbl_chart_page.config(text="Gráfico 0/0")
            return
        self.current_segment_idx = max(0, min(self.current_segment_idx, len(self.chart_segments) - 1))
        a, b = self.chart_segments[self.current_segment_idx]
        sub = self.active_candles[a : b + 1]
        self.update_chart(sub, seg_start=a)
        self.lbl_chart_page.config(text=f"Gráfico {self.current_segment_idx + 1}/{len(self.chart_segments)}")

    def show_next_chart(self) -> None:
        if not self.chart_segments:
            return
        if self.current_segment_idx < len(self.chart_segments) - 1:
            self.current_segment_idx += 1
            self._render_current_segment_chart()

    def show_prev_chart(self) -> None:
        if not self.chart_segments:
            return
        if self.current_segment_idx > 0:
            self.current_segment_idx -= 1
            self._render_current_segment_chart()

    def _render_ops_listbox(self) -> None:
        self.trades_listbox.delete(0, tk.END)
        for op in self.strategy_ops[-300:]:
            self.trades_listbox.insert(
                tk.END,
                f"{op.side:<4} {self.symbol:<8} {op.price:>10.4f} qty {op.qty:<10.6f} {op.reason} t={op.time_ms}",
            )
        self.trades_listbox.yview_moveto(1.0)

    def _build_bot_tab(self, tab_bot: ttk.Frame) -> None:
        tab_bot.grid_rowconfigure(0, weight=1)
        tab_bot.grid_columnconfigure(0, weight=1)

        self.embedded_bot_gui = None
        # Intentamos incrustar el BotGUI real de bot.py (igual que en el optimizador original).
        try:
            bot_mod = _import_bot_module_from_candidates()
            if bot_mod and hasattr(bot_mod, "BotGUI"):
                host = ttk.Frame(tab_bot)
                host.grid(row=0, column=0, sticky="nsew")
                self.embedded_bot_gui = bot_mod.BotGUI(host)
                return
        except Exception as exc:
            self._set_error(f"No se pudo incrustar BotGUI real: {exc}")

        # Fallback: panel simplificado (si no existe BotGUI en bot.py)
        tab_bot.grid_rowconfigure(2, weight=1)
        bot_top = ttk.LabelFrame(tab_bot, text="Control Bot Spot (Long Only)", padding=10)
        bot_top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.btn_bot_play = ttk.Button(bot_top, text="PLAY BOT", command=self.start_bot)
        self.btn_bot_pause = ttk.Button(bot_top, text="PAUSE BOT", command=self.pause_bot)
        self.btn_bot_stop = ttk.Button(bot_top, text="STOP BOT", command=self.stop_bot)
        self.btn_bot_recalc = ttk.Button(bot_top, text="Recalcular Estrategia", command=self._recompute_strategy_views)
        self.btn_bot_play.pack(side="left", padx=4)
        self.btn_bot_pause.pack(side="left", padx=4)
        self.btn_bot_stop.pack(side="left", padx=4)
        self.btn_bot_recalc.pack(side="left", padx=4)

        bot_info = ttk.Frame(tab_bot)
        bot_info.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.lbl_bot_mode = ttk.Label(bot_info, text="Modo: PAUSADO", font=("Segoe UI", 10, "bold"))
        self.lbl_bot_last_op = ttk.Label(bot_info, text="Última operación: -")
        self.lbl_bot_open_pos = ttk.Label(bot_info, text="Posición actual: SIN POSICIÓN")
        self.lbl_bot_mode.pack(anchor="w")
        self.lbl_bot_last_op.pack(anchor="w")
        self.lbl_bot_open_pos.pack(anchor="w")

        bot_ops_frame = ttk.LabelFrame(tab_bot, text="Operaciones recientes del bot/estrategia", padding=8)
        bot_ops_frame.grid(row=2, column=0, sticky="nsew")
        bot_ops_frame.grid_rowconfigure(0, weight=1)
        bot_ops_frame.grid_columnconfigure(0, weight=1)
        self.bot_ops_listbox = tk.Listbox(bot_ops_frame, font=("Consolas", 10))
        self.bot_ops_listbox.grid(row=0, column=0, sticky="nsew")

    def _refresh_bot_tab(self) -> None:
        if getattr(self, "embedded_bot_gui", None) is not None:
            return
        if not hasattr(self, "lbl_bot_mode"):
            return
        mode_txt = "RUNNING" if self.bot_running else "PAUSADO"
        self.lbl_bot_mode.config(text=f"Modo: {mode_txt}")

        if self.strategy_ops:
            last = self.strategy_ops[-1]
            ts = datetime.fromtimestamp(int(last.time_ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            self.lbl_bot_last_op.config(text=f"Última operación: {last.side} {last.reason} @ {last.price:.4f} ({ts} UTC)")
            if last.side == "BUY":
                self.lbl_bot_open_pos.config(text=f"Posición actual: LONG ABIERTA (qty={last.qty:.6f})")
            else:
                self.lbl_bot_open_pos.config(text="Posición actual: SIN POSICIÓN (último cierre SELL)")
        else:
            self.lbl_bot_last_op.config(text="Última operación: -")
            self.lbl_bot_open_pos.config(text="Posición actual: SIN POSICIÓN")

        self.bot_ops_listbox.delete(0, tk.END)
        for op in self.strategy_ops[-200:]:
            self.bot_ops_listbox.insert(
                tk.END,
                f"{op.side:<4} {op.reason:<3} px={op.price:>10.4f} qty={op.qty:<10.6f} t={op.time_ms}",
            )
        self.bot_ops_listbox.yview_moveto(1.0)

    def _parse_date_to_ms(self, value: str, end_of_day: bool = False) -> Optional[int]:
        text = (value or "").strip()
        if not text:
            return None
        dt = datetime.strptime(text, "%Y/%m/%d")
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59, microsecond=999000)
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def _get_candles_in_selected_range(self, candles: list[dict]) -> list[dict]:
        if not candles:
            return []
        try:
            start_ms = self._parse_date_to_ms(self.date_from_var.get(), end_of_day=False)
            end_ms = self._parse_date_to_ms(self.date_to_var.get(), end_of_day=True)
        except ValueError:
            self._set_error("Formato de fecha inválido. Usa YYYY/MM/DD")
            return candles

        if start_ms is None and end_ms is None:
            return candles
        if start_ms is not None and end_ms is not None and start_ms > end_ms:
            self._set_error("Rango inválido: 'Desde' no puede ser mayor que 'Hasta'")
            return candles

        out = []
        for c in candles:
            t = int(c.get("close_time", c.get("open_time", 0)))
            if start_ms is not None and t < start_ms:
                continue
            if end_ms is not None and t > end_ms:
                continue
            out.append(c)

        if not out:
            self._set_error("No hay velas en el rango solicitado. Ajusta las fechas.")
            return candles
        self._set_error("")
        return out

    def _read_selected_range(self) -> tuple[Optional[int], Optional[int]]:
        try:
            start_ms = self._parse_date_to_ms(self.date_from_var.get(), end_of_day=False)
            end_ms = self._parse_date_to_ms(self.date_to_var.get(), end_of_day=True)
        except ValueError as exc:
            raise ValueError("Formato de fecha inválido. Usa YYYY/MM/DD") from exc
        if start_ms is not None and end_ms is not None and start_ms > end_ms:
            raise ValueError("Rango inválido: 'Desde' no puede ser mayor que 'Hasta'")
        return start_ms, end_ms

    def _save_current_range_chart(self) -> None:
        if not self.active_candles:
            return
        out_dir = Path("estrategia_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.selected_start_ms and self.selected_end_ms:
            ds = datetime.fromtimestamp(self.selected_start_ms / 1000, tz=timezone.utc).strftime("%Y%m%d")
            de = datetime.fromtimestamp(self.selected_end_ms / 1000, tz=timezone.utc).strftime("%Y%m%d")
            path = out_dir / f"grafico_rango_{ds}_{de}.png"
        else:
            path = out_dir / "grafico_rango_full.png"
        self._render_current_segment_chart()
        self.figure.savefig(path, dpi=130, bbox_inches="tight")
        self._set_error(f"Rango aplicado. Gráfico generado: {path}")

    def apply_date_range(self) -> None:
        if self.range_loading:
            self._set_error("Ya se está descargando un rango. Espera por favor...")
            return
        try:
            start_ms, end_ms = self._read_selected_range()
        except ValueError as exc:
            self._set_error(str(exc))
            return

        if start_ms is None and end_ms is None:
            self.range_locked = False
            self.selected_start_ms = None
            self.selected_end_ms = None
            self._recompute_strategy_views()
            return

        if not self.candles:
            self._set_error("No hay velas base para inferir extremos. Espera carga inicial.")
            return

        start_ms = start_ms if start_ms is not None else int(self.candles[0]["close_time"])
        end_ms = end_ms if end_ms is not None else int(self.candles[-1]["close_time"])
        if start_ms > end_ms:
            self._set_error("Rango inválido: 'Desde' no puede ser mayor que 'Hasta'.")
            return

        if "get_candles_range" not in self._api:
            self._set_error("La API actual no soporta carga histórica por rango.")
            return

        self.selected_start_ms = int(start_ms)
        self.selected_end_ms = int(end_ms)
        self.range_loading = True
        self.btn_apply_range.config(state="disabled")
        self.btn_recalc.config(state="disabled")
        self._set_error("Descargando velas del rango solicitado por bloques trimestrales... espera por favor.")

        def worker():
            try:
                merged: list[dict[str, Any]] = []
                cur_start_dt = ms_to_utc(int(start_ms))
                end_dt = ms_to_utc(int(end_ms))
                while cur_start_dt < end_dt:
                    cur_end_dt = add_months(cur_start_dt, 3)
                    if cur_end_dt > end_dt:
                        cur_end_dt = end_dt
                    c_start = int(cur_start_dt.timestamp() * 1000)
                    c_end = int(cur_end_dt.timestamp() * 1000)
                    part = self._api["get_candles_range"](self.symbol, self.interval, c_start, c_end) or []
                    merged.extend(part)
                    if cur_end_dt <= cur_start_dt:
                        break
                    cur_start_dt = cur_end_dt + timedelta(hours=1)

                merged.sort(key=lambda x: int(x.get("open_time", 0)))
                dedup: list[dict[str, Any]] = []
                seen = set()
                for c in merged:
                    ot = int(c.get("open_time", 0))
                    if ot in seen:
                        continue
                    seen.add(ot)
                    dedup.append(c)

                self._ui_queue.put({
                    "kind": "range_loaded",
                    "candles": dedup,
                    "start_ms": int(start_ms),
                    "end_ms": int(end_ms),
                })
            except Exception:
                self._ui_queue.put({"kind": "error", "message": f"Error descargando rango:\n{traceback.format_exc(limit=2)}"})

        threading.Thread(target=worker, daemon=True).start()

    def clear_date_range(self) -> None:
        self.date_from_var.set("")
        self.date_to_var.set("")
        self.range_locked = False
        self.range_loading = False
        self.selected_start_ms = None
        self.selected_end_ms = None
        self.candles = self._api["get_candles"](self.symbol, self.interval) or []
        self._recompute_strategy_views()
        self.btn_apply_range.config(state="normal")
        self.btn_recalc.config(state="normal")
        self._set_error("Rango limpiado. Volviste al dataset reciente.")

    def _refresh_period_table(self) -> None:
        for i in self.tree_periods.get_children():
            self.tree_periods.delete(i)
        if not self.strategy_equity_curve:
            return

        eq = self.strategy_equity_curve
        final_eq = eq[-1][1]
        now_ms = eq[-1][0]
        periods = [
            ("7D", 7 * 24 * INTERVAL_MS[INTERVAL]),
            ("30D", 30 * 24 * INTERVAL_MS[INTERVAL]),
            ("90D", 90 * 24 * INTERVAL_MS[INTERVAL]),
            ("180D", 180 * 24 * INTERVAL_MS[INTERVAL]),
            ("ALL", None),
        ]
        self.period_rows = []
        for name, window in periods:
            if window is None:
                base = eq[0][1]
                start_ms = eq[0][0]
            else:
                start_ms = now_ms - window
                base = next((v for t, v in eq if t >= start_ms), eq[0][1])
            pnl = final_eq - base
            roi = (pnl / base) * 100 if base else 0.0
            ops_n = sum(1 for o in self.strategy_ops if o.time_ms >= start_ms and o.side == "SELL")
            row = {
                "periodo": name,
                "operaciones": ops_n,
                "roi_pct": roi,
                "pnl_usdt": pnl,
                "equity_final": final_eq,
            }
            self.period_rows.append(row)
            self.tree_periods.insert("", tk.END, values=(name, ops_n, f"{roi:.2f}", f"{pnl:.2f}", f"{final_eq:.2f}"))

    def _update_perf_chart(self) -> None:
        self.ax_perf.clear()
        if not self.strategy_equity_curve:
            self.canvas_perf.draw_idle()
            return
        ts = [int(t) for t, _ in self.strategy_equity_curve]
        vals = [v for _, v in self.strategy_equity_curve]
        x = list(range(len(vals)))
        self.ax_perf.plot(x, vals, color="#1976d2")
        self.ax_perf.set_title("Curva de Equity (Estrategia 1h, cierre de vela)")
        self.ax_perf.grid(alpha=0.2)
        n = len(x)
        if n > 1:
            step = max(1, n // 8)
            ticks = list(range(0, n, step))
            if ticks[-1] != n - 1:
                ticks.append(n - 1)
            labels = [datetime.fromtimestamp(ts[i] / 1000, tz=timezone.utc).strftime("%Y-%m-%d") for i in ticks]
            self.ax_perf.set_xticks(ticks)
            self.ax_perf.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            self.ax_perf.set_xlabel("Tiempo (UTC)")
        self.canvas_perf.draw_idle()

    def update_chart(self, candles: list[dict], seg_start: int = 0) -> None:
        self.ax.clear()
        if not candles:
            self.canvas.draw_idle()
            return

        # Mostrar todo el rango, reduciendo densidad visual si hay demasiadas velas.
        max_draw = 1200
        step = max(1, len(candles) // max_draw)
        data = candles[::step]

        min_p = min(float(c["low"]) for c in data)
        max_p = max(float(c["high"]) for c in data)

        for i, c in enumerate(data):
            o = float(c["open"])
            h = float(c["high"])
            l = float(c["low"])
            cl = float(c["close"])
            color = "#26a69a" if cl >= o else "#ef5350"
            self.ax.vlines(i, l, h, color=color, linewidth=1)
            self.ax.add_patch(Rectangle((i - 0.28, min(o, cl)), 0.56, max(abs(cl - o), 1e-8), color=color, alpha=0.85))

        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        for op in self.strategy_ops:
            x = (op.index - seg_start) // step
            if x < 0 or x >= len(data):
                continue
            if op.side == "BUY":
                buy_x.append(x)
                buy_y.append(op.price)
            else:
                sell_x.append(x)
                sell_y.append(op.price)

        if buy_x:
            self.ax.scatter(buy_x, buy_y, marker="^", color="#1b5e20", s=45, label="BUY", zorder=5)
        if sell_x:
            self.ax.scatter(sell_x, sell_y, marker="v", color="#b71c1c", s=45, label="SELL", zorder=5)
        if buy_x or sell_x:
            self.ax.legend(loc="upper left")

        self.ax.set_xlim(-1, len(data) + 1)
        self.ax.set_ylim(min_p * 0.995, max_p * 1.005)
        title_extra = f" | vista comprimida x{step}" if step > 1 else ""
        self.ax.set_title(f"{self.symbol} - 1h (solo cierre de vela){title_extra}")
        self.ax.grid(alpha=0.2)

        n = len(data)
        if n > 1:
            ticks = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
            ticks = sorted(set(ticks))
            labels = [datetime.fromtimestamp(int(data[i]["close_time"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d") for i in ticks]
            self.ax.set_xticks(ticks)
            self.ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)

        self.canvas.draw_idle()

    def _process_ui_queue(self) -> None:
        try:
            while True:
                payload = self._ui_queue.get_nowait()
                if payload["kind"] == "update":
                    self.balance = payload["balance"]
                    self.asset_qty = payload["asset_qty"]
                    self.price = payload["price"]
                    self.status_text = payload["status"]

                    incoming_candles = payload["candles"]
                    incoming_sig = self._candles_signature(incoming_candles)

                    # Evita recálculos pesados repetidos (causa de congelamiento en navegación de gráficos).
                    # Si el dataset no cambió, solo refrescamos encabezado.
                    if incoming_sig != self._last_candles_sig:
                        self.candles = incoming_candles
                        self._recompute_strategy_views()
                    self._refresh_ui()
                elif payload["kind"] == "range_loaded":
                    self.range_loading = False
                    self.btn_apply_range.config(state="normal")
                    self.btn_recalc.config(state="normal")
                    candles = payload.get("candles") or []
                    if not candles:
                        self._set_error("No se pudieron descargar velas para ese rango. Prueba un rango menor o valida el símbolo.")
                    else:
                        self.candles = candles
                        self.range_locked = True
                        self.selected_start_ms = int(payload["start_ms"])
                        self.selected_end_ms = int(payload["end_ms"])
                        self._recompute_strategy_views()
                        self._save_current_range_chart()
                else:
                    self.range_loading = False
                    self.btn_apply_range.config(state="normal")
                    self.btn_recalc.config(state="normal")
                    self._set_error(payload["message"])
        except queue.Empty:
            pass
        self.after(150, self._process_ui_queue)

    def update_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw_balance = self._api["get_balance"]()
                balance, asset_qty = self._parse_balance(raw_balance)
                price = float(self._api["get_price"](self.symbol))
                candles = self.candles if self.range_locked else (self._api["get_candles"](self.symbol, self.interval) or [])
                self._ui_queue.put(
                    {
                        "kind": "update",
                        "balance": balance,
                        "asset_qty": asset_qty,
                        "price": price,
                        "candles": candles,
                        "status": "RUNNING" if self.bot_running else "PAUSED",
                    }
                )
            except Exception:
                self._ui_queue.put({"kind": "error", "message": f"Loop error:\n{traceback.format_exc(limit=2)}"})
            time.sleep(1)

    def export_excel(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Exportar rentabilidades/operaciones",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")],
        )
        if not path:
            return

        ops_rows = [
            {
                "side": o.side,
                "time_ms": o.time_ms,
                "price": o.price,
                "qty": o.qty,
                "equity_after": o.equity_after,
                "reason": o.reason,
            }
            for o in self.strategy_ops
        ]
        try:
            import pandas as pd

            with pd.ExcelWriter(path, engine="openpyxl") as w:
                pd.DataFrame(ops_rows).to_excel(w, index=False, sheet_name="Operaciones")
                period_export = []
                for row in self.period_rows:
                    r = row.copy()
                    r["fecha_desde"] = self.date_from_var.get() or "FULL"
                    r["fecha_hasta"] = self.date_to_var.get() or "FULL"
                    period_export.append(r)
                pd.DataFrame(period_export).to_excel(w, index=False, sheet_name="Rentabilidades")
                pd.DataFrame(self.strategy_equity_curve, columns=["time_ms", "equity"]).to_excel(
                    w, index=False, sheet_name="EquityCurve"
                )
            self._set_error(f"Exportado OK: {path}")
        except Exception:
            import csv

            base = Path(path)
            ops_csv = base.with_suffix(".operaciones.csv")
            per_csv = base.with_suffix(".rentabilidades.csv")
            eq_csv = base.with_suffix(".equity.csv")
            for fpath, rows in [
                (ops_csv, ops_rows),
                (per_csv, self.period_rows),
                (eq_csv, [{"time_ms": t, "equity": e} for t, e in self.strategy_equity_curve]),
            ]:
                with open(fpath, "w", newline="", encoding="utf-8") as f:
                    if not rows:
                        continue
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            self._set_error(f"Pandas/openpyxl no disponible. Exportado CSV: {ops_csv.name}, {per_csv.name}, {eq_csv.name}")

    def start_bot(self) -> None:
        try:
            self._api["start_bot"]()
            self.bot_running = True
            self.status_text = "RUNNING"
            self._set_error("")
        except Exception as exc:
            self.status_text = "ERROR"
            self._set_error(f"Error al iniciar bot: {exc}")

    def pause_bot(self) -> None:
        try:
            self._api["pause_bot"]()
            self.bot_running = False
            self.status_text = "PAUSED"
            self._set_error("")
        except Exception as exc:
            self.status_text = "ERROR"
            self._set_error(f"Error al pausar bot: {exc}")

    def stop_bot(self) -> None:
        try:
            self._api["stop_bot"]()
            self.bot_running = False
            self.status_text = "STOPPED"
            self._set_error("")
        except Exception as exc:
            self.status_text = "ERROR"
            self._set_error(f"Error al detener bot: {exc}")

    def _on_close(self) -> None:
        self._stop_event.set()
        self.destroy()


if __name__ == "__main__":
    app = TradingDashboard(symbol="SOLUSDT")
    app.mainloop()
