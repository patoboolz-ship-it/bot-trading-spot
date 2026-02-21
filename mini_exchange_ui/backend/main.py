from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from binance.client import Client
except Exception:
    Client = None

from .bot_engine import BotEngine
from .config import settings
from .data_layer import DataLayer
from .state_store import StateStore

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

app = FastAPI(title="Mini Exchange UI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class WSManager:
    def __init__(self):
        self.clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.clients:
            self.clients.remove(ws)

    async def broadcast(self, msg: dict[str, Any]):
        bad = []
        for ws in self.clients:
            try:
                await ws.send_json(msg)
            except Exception:
                bad.append(ws)
        for ws in bad:
            self.disconnect(ws)


client = Client(settings.api_key, settings.api_secret) if (Client and settings.api_key and settings.api_secret) else None
data = DataLayer(client)
store = StateStore()
wsman = WSManager()


async def emit(event_type: str, payload: dict[str, Any]):
    await wsman.broadcast({"type": event_type, "payload": payload})


def _emit_sync(msg: dict[str, Any]):
    asyncio.create_task(wsman.broadcast(msg))


engine = BotEngine(store, _emit_sync)


def recalc_account_with_last_price():
    balances = store.state.balances
    last_price = store.state.candles[-1]["close"] if store.state.candles else 0.0
    usdt = float(balances.get("USDT", 0.0))
    assets = []
    equity = usdt
    for asset, qty in balances.items():
        if asset == "USDT":
            continue
        est = float(qty) * float(last_price)
        assets.append({"asset": asset, "qty": qty, "est_usdt": est})
        equity += est
    store.state.assets = assets
    store.state.equity = equity
    store.state.pnl_usdt = equity - store.state.equity_initial
    base = store.state.equity_initial or 1.0
    store.state.pnl_pct = store.state.pnl_usdt / base * 100.0
    store.append_equity(equity, store.state.pnl_usdt)


async def market_loop():
    while True:
        try:
            symbol, interval = store.state.symbol, store.state.interval
            candles = data.get_historical_candles(symbol, interval, limit=500)
            store.state.candles = candles
            await emit("market.candle_update", {"candles": candles[-2:]})
            recalc_account_with_last_price()
            await emit("account.update", store.snapshot()["account_state"])
            await emit("performance.update", store.snapshot()["performance_state"])
        except Exception:
            store.log_error(traceback.format_exc())
            await emit("bot.state", {"state": "ERROR"})
        await asyncio.sleep(3)


@app.on_event("startup")
async def startup():
    store.state.symbol = settings.symbol
    store.state.interval = settings.interval
    store.state.balances = data.get_balances()
    recalc_account_with_last_price()
    asyncio.create_task(market_loop())


@app.get("/")
def index():
    return FileResponse(str(FRONTEND / "index.html"))


app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/state")
def get_state():
    return store.snapshot()


@app.post("/control/play")
def control_play():
    engine.start()
    return {"ok": True, "state": engine.get_state()}


@app.post("/control/pause")
def control_pause():
    engine.pause()
    return {"ok": True, "state": engine.get_state()}


@app.post("/control/stop")
def control_stop():
    engine.stop()
    return {"ok": True, "state": engine.get_state()}


@app.post("/symbol")
def set_symbol(payload: dict[str, Any]):
    store.state.symbol = payload.get("symbol", store.state.symbol)
    return {"ok": True, "symbol": store.state.symbol}


@app.post("/timeframe")
def set_timeframe(payload: dict[str, Any]):
    store.state.interval = payload.get("interval", store.state.interval)
    return {"ok": True, "interval": store.state.interval}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await wsman.connect(ws)
    await ws.send_json({"type": "snapshot", "payload": store.snapshot()})
    try:
        while True:
            _ = await ws.receive_text()
    except WebSocketDisconnect:
        wsman.disconnect(ws)
