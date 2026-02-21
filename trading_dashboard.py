import importlib
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


class TradingDashboard(tk.Tk):
    def __init__(self, symbol: str = "SOLUSDT", interval: str = "1h"):
        super().__init__()
        self.title("Trading Dashboard - Mini Exchange")
        self.geometry("1400x900")
        self.minsize(1200, 760)

        self.symbol = symbol
        self.interval = interval

        # Estado obligatorio
        self.balance = 0.0
        self.equity = 0.0
        self.pnl = 0.0
        self.price = 0.0
        self.trades: list[dict[str, Any]] = []
        self.candles: list[dict[str, Any]] = []
        self.bot_running = False

        self.asset_qty = 0.0
        self.initial_equity = 0.0
        self.pnl_percent = 0.0
        self.status_text = "IDLE"
        self.last_error = ""
        self._last_trade_id: Any = None

        self._api = self._resolve_api_functions()
        self._ui_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        self._build_layout()
        self._load_initial_data()

        self.after(150, self._process_ui_queue)
        self._thread = threading.Thread(target=self.update_loop, daemon=True)
        self._thread.start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _resolve_api_functions(self) -> dict[str, Callable[..., Any]]:
        required = ["get_balance", "get_price", "get_trades", "get_candles", "start_bot", "pause_bot"]
        optional = ["stop_bot"]
        modules = ["bot", "api", "trading_api"]

        resolved: dict[str, Callable[..., Any]] = {}
        for module_name in modules:
            try:
                mod = importlib.import_module(module_name)
            except Exception:
                continue
            for fn in required + optional:
                if fn not in resolved and hasattr(mod, fn) and callable(getattr(mod, fn)):
                    resolved[fn] = getattr(mod, fn)

        missing = [name for name in required if name not in resolved]
        if missing:
            raise RuntimeError(
                "No se encontraron funciones requeridas de API: "
                f"{', '.join(missing)}. Deben existir en bot.py/api.py/trading_api.py"
            )
        if "stop_bot" not in resolved:
            resolved["stop_bot"] = resolved["pause_bot"]
        return resolved

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

        middle = ttk.Frame(self, padding=8)
        middle.grid(row=1, column=0, sticky="nsew")
        middle.grid_rowconfigure(0, weight=1)
        middle.grid_columnconfigure(0, weight=4)
        middle.grid_columnconfigure(1, weight=2)

        chart_frame = ttk.LabelFrame(middle, text="Gráfico de Velas", padding=8)
        chart_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        chart_frame.grid_rowconfigure(0, weight=1)
        chart_frame.grid_columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        trades_frame = ttk.LabelFrame(middle, text="Trades Ejecutados", padding=8)
        trades_frame.grid(row=0, column=1, sticky="nsew")
        trades_frame.grid_rowconfigure(0, weight=1)
        trades_frame.grid_columnconfigure(0, weight=1)

        self.trades_listbox = tk.Listbox(trades_frame, font=("Consolas", 10))
        self.trades_listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(trades_frame, orient="vertical", command=self.trades_listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.trades_listbox.configure(yscrollcommand=scroll.set)

        bottom = ttk.Frame(self, padding=8)
        bottom.grid(row=2, column=0, sticky="ew")

        self.btn_play = ttk.Button(bottom, text="PLAY", command=self.start_bot)
        self.btn_pause = ttk.Button(bottom, text="PAUSE", command=self.pause_bot)
        self.btn_stop = ttk.Button(bottom, text="STOP", command=self.stop_bot)

        self.btn_play.pack(side="left", padx=6)
        self.btn_pause.pack(side="left", padx=6)
        self.btn_stop.pack(side="left", padx=6)

    def _load_initial_data(self) -> None:
        try:
            self.balance, self.asset_qty = self._parse_balance(self._api["get_balance"]())
            self.price = float(self._api["get_price"](self.symbol))
            self.candles = self._api["get_candles"](self.symbol, self.interval) or []
            self.trades = self._api["get_trades"]() or []
            self.equity = self.balance + self.asset_qty * self.price
            self.initial_equity = self.equity if self.equity > 0 else 1.0
            self._last_trade_id = self._get_last_trade_id(self.trades)
            self.update_chart(self.candles)
            self._refresh_ui()
        except Exception as exc:
            self._set_error(f"Error inicializando: {exc}")

    def _parse_balance(self, raw: Any) -> tuple[float, float]:
        if isinstance(raw, tuple) and len(raw) >= 2:
            return float(raw[0]), float(raw[1])
        if isinstance(raw, dict):
            usdt = float(raw.get("USDT", raw.get("balance_usdt", raw.get("balance", 0.0))))
            asset = float(raw.get("asset_qty", raw.get("qty", raw.get("base_qty", 0.0))))
            return usdt, asset
        return float(raw), 0.0

    def _get_last_trade_id(self, trades: list[dict[str, Any]]) -> Any:
        if not trades:
            return None
        last = trades[-1]
        return last.get("id") or last.get("trade_id") or last.get("time") or len(trades)

    def _set_error(self, msg: str) -> None:
        self.last_error = msg
        self.lbl_error.config(text=msg)

    def _refresh_ui(self) -> None:
        self.lbl_balance.config(text=f"Balance USDT: {self.balance:,.4f}")
        self.lbl_equity.config(text=f"Equity: {self.equity:,.4f}")
        pnl_color = "#2e7d32" if self.pnl >= 0 else "#c62828"
        self.lbl_pnl.config(text=f"PnL: {self.pnl:,.4f} ({self.pnl_percent:.2f}%)", foreground=pnl_color)
        self.lbl_price.config(text=f"Precio {self.symbol}: {self.price:,.4f}")
        self.lbl_state.config(text=f"Estado: {self.status_text}")

    def update_chart(self, candles: list[dict[str, Any]]) -> None:
        self.ax.clear()
        if not candles:
            self.ax.set_title("Sin velas")
            self.canvas.draw_idle()
            return

        data = candles[-300:]
        min_p = min(float(c["low"]) for c in data)
        max_p = max(float(c["high"]) for c in data)

        for i, c in enumerate(data):
            o = float(c["open"])
            h = float(c["high"])
            l = float(c["low"])
            cl = float(c["close"])
            color = "#26a69a" if cl >= o else "#ef5350"
            self.ax.vlines(i, l, h, color=color, linewidth=1)
            body_y = min(o, cl)
            body_h = max(abs(cl - o), 1e-8)
            self.ax.add_patch(Rectangle((i - 0.32, body_y), 0.64, body_h, color=color, alpha=0.85))

        self.ax.set_xlim(-1, len(data) + 1)
        self.ax.set_ylim(min_p * 0.995, max_p * 1.005)
        self.ax.set_title(f"{self.symbol} - {self.interval}")
        self.ax.grid(alpha=0.2)
        self.canvas.draw_idle()

    def _render_trades(self, new_trades: list[dict[str, Any]]) -> None:
        for t in new_trades:
            side = str(t.get("side", "?")).upper()
            qty = float(t.get("qty", t.get("quantity", 0.0)))
            px = float(t.get("price", 0.0))
            ts = t.get("timestamp", t.get("time", ""))
            line = f"{side:<4} {self.symbol:<8} {px:>10.4f} qty {qty:<10.6f} {ts}"
            self.trades_listbox.insert(tk.END, line)
        self.trades_listbox.yview_moveto(1.0)

    def _process_ui_queue(self) -> None:
        try:
            while True:
                payload = self._ui_queue.get_nowait()
                kind = payload["kind"]
                if kind == "update":
                    self.balance = payload["balance"]
                    self.asset_qty = payload["asset_qty"]
                    self.price = payload["price"]
                    self.candles = payload["candles"]
                    self.equity = payload["equity"]
                    self.pnl = payload["pnl"]
                    self.pnl_percent = payload["pnl_percent"]
                    self.trades.extend(payload["new_trades"])
                    self.status_text = payload["status"]
                    self._refresh_ui()
                    self.update_chart(self.candles)
                    if payload["new_trades"]:
                        self._render_trades(payload["new_trades"])
                elif kind == "error":
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
                candles = self._api["get_candles"](self.symbol, self.interval) or []
                trades_all = self._api["get_trades"]() or []

                new_trades = self._extract_new_trades(trades_all)

                equity = balance + asset_qty * price
                pnl = equity - self.initial_equity
                pnl_pct = (pnl / self.initial_equity) * 100 if self.initial_equity else 0.0

                self._ui_queue.put(
                    {
                        "kind": "update",
                        "balance": balance,
                        "asset_qty": asset_qty,
                        "price": price,
                        "candles": candles,
                        "equity": equity,
                        "pnl": pnl,
                        "pnl_percent": pnl_pct,
                        "new_trades": new_trades,
                        "status": "RUNNING" if self.bot_running else "PAUSED",
                    }
                )
            except Exception:
                self._ui_queue.put(
                    {"kind": "error", "message": f"Loop error:\n{traceback.format_exc(limit=2)}"}
                )
            time.sleep(1)

    def _extract_new_trades(self, trades_all: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not trades_all:
            return []
        if self._last_trade_id is None:
            self._last_trade_id = self._get_last_trade_id(trades_all)
            return trades_all[-20:]

        out: list[dict[str, Any]] = []
        for t in trades_all:
            tid = t.get("id") or t.get("trade_id") or t.get("time")
            if tid is None:
                continue
            if str(tid) == str(self._last_trade_id):
                out = []
            else:
                out.append(t)
        self._last_trade_id = self._get_last_trade_id(trades_all)
        return out[-20:]

    def start_bot(self) -> None:
        try:
            self._api["start_bot"]()
            self.bot_running = True
            self.status_text = "RUNNING"
            self._set_error("")
            self._refresh_ui()
        except Exception as exc:
            self.status_text = "ERROR"
            self._set_error(f"Error al iniciar bot: {exc}")

    def pause_bot(self) -> None:
        try:
            self._api["pause_bot"]()
            self.bot_running = False
            self.status_text = "PAUSED"
            self._set_error("")
            self._refresh_ui()
        except Exception as exc:
            self.status_text = "ERROR"
            self._set_error(f"Error al pausar bot: {exc}")

    def stop_bot(self) -> None:
        try:
            self._api["stop_bot"]()
            self.bot_running = False
            self.status_text = "STOPPED"
            self._set_error("")
            self._refresh_ui()
        except Exception as exc:
            self.status_text = "ERROR"
            self._set_error(f"Error al detener bot: {exc}")

    def _on_close(self) -> None:
        self._stop_event.set()
        self.destroy()


if __name__ == "__main__":
    app = TradingDashboard(symbol="SOLUSDT", interval="1h")
    app.mainloop()
