from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AppState:
    bot_state: str = "IDLE"
    symbol: str = "SOLUSDT"
    interval: str = "1h"
    live_mode: bool = True
    last_action: str = "init"
    errors: list[str] = field(default_factory=list)
    balances: dict[str, float] = field(default_factory=lambda: {"USDT": 1000.0})
    assets: list[dict[str, Any]] = field(default_factory=list)
    equity: float = 1000.0
    equity_initial: float = 1000.0
    pnl_usdt: float = 0.0
    pnl_pct: float = 0.0
    orders: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    candles: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)


class StateStore:
    def __init__(self, db_path: str = "mini_exchange.db"):
        self.state = AppState()
        self._lock = threading.RLock()
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS trades(ts INTEGER, payload TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS orders(ts INTEGER, payload TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS equity(ts INTEGER, equity REAL, pnl REAL)")
        conn.commit()
        conn.close()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "bot_state": self.state.bot_state,
                "symbol": self.state.symbol,
                "interval": self.state.interval,
                "live_mode": self.state.live_mode,
                "last_action": self.state.last_action,
                "errors": list(self.state.errors[-200:]),
                "account_state": {
                    "balances": self.state.balances,
                    "assets": self.state.assets,
                    "equity": self.state.equity,
                    "equity_initial": self.state.equity_initial,
                    "pnl_usdt": self.state.pnl_usdt,
                    "pnl_pct": self.state.pnl_pct,
                },
                "orders_state": self.state.orders[-300:],
                "trades_state": self.state.trades[-300:],
                "market_state": {
                    "candles": self.state.candles[-2000:],
                    "last_price": self.state.candles[-1]["close"] if self.state.candles else None,
                },
                "performance_state": {"equity_curve": self.state.equity_curve[-3000:]},
            }

    def log_error(self, msg: str):
        with self._lock:
            self.state.bot_state = "ERROR"
            self.state.errors.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}")

    def append_order(self, payload: dict[str, Any]):
        with self._lock:
            self.state.orders.append(payload)
        self._persist("orders", payload)

    def append_trade(self, payload: dict[str, Any]):
        with self._lock:
            self.state.trades.append(payload)
        self._persist("trades", payload)

    def append_equity(self, equity: float, pnl: float):
        ts = int(time.time() * 1000)
        row = {"time": ts, "equity": equity, "pnl": pnl}
        with self._lock:
            self.state.equity_curve.append(row)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("INSERT INTO equity(ts, equity, pnl) VALUES(?,?,?)", (ts, equity, pnl))
        conn.commit()
        conn.close()

    def _persist(self, table: str, payload: dict[str, Any]):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(f"INSERT INTO {table}(ts, payload) VALUES(?,?)", (int(time.time() * 1000), json.dumps(payload)))
        conn.commit()
        conn.close()
