from __future__ import annotations

import random
import time
from typing import Any


class DataLayer:
    """Binance wrapper with simple fallback data when API is unavailable."""

    def __init__(self, client=None):
        self.client = client

    def get_balances(self) -> dict[str, float]:
        if not self.client:
            return {"USDT": 1000.0, "SOL": 0.0}
        account = self.client.get_account()
        out = {}
        for b in account.get("balances", []):
            total = float(b.get("free", 0)) + float(b.get("locked", 0))
            if total > 0:
                out[b["asset"]] = total
        return out

    def get_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        if not self.client:
            return []
        return self.client.get_open_orders(symbol=symbol)

    def get_trades(self, symbol: str) -> list[dict[str, Any]]:
        if not self.client:
            return []
        return self.client.get_my_trades(symbol=symbol)

    def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: float | None = None) -> dict[str, Any]:
        if not self.client:
            return {
                "symbol": symbol,
                "side": side,
                "status": "FILLED",
                "executedQty": qty,
                "price": price or 0,
                "time": int(time.time() * 1000),
                "mock": True,
            }
        payload = {"symbol": symbol, "side": side, "type": order_type, "quantity": qty}
        if price is not None:
            payload["price"] = price
        return self.client.create_order(**payload)

    def get_historical_candles(self, symbol: str, interval: str, limit: int = 1000) -> list[dict[str, Any]]:
        if not self.client:
            return self._mock_candles(limit)
        rows = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        return [
            {
                "time": int(r[0] // 1000),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
                "close_time": int(r[6]),
            }
            for r in rows
        ]

    def _mock_candles(self, limit: int) -> list[dict[str, Any]]:
        out = []
        px = 100.0
        step = 3600
        now = int(time.time()) - limit * step
        for i in range(limit):
            o = px
            c = max(1.0, o + random.uniform(-1.5, 1.5))
            h = max(o, c) + random.uniform(0, 1.2)
            l = min(o, c) - random.uniform(0, 1.2)
            px = c
            out.append({"time": now + i * step, "open": o, "high": h, "low": l, "close": c, "volume": random.uniform(100, 1000), "close_time": (now + (i + 1) * step) * 1000})
        return out
