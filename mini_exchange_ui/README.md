# Mini Exchange UI (Local)

Dashboard local tipo mini-exchange para controlar/monitorear bot spot.

## Qué incluye
- Play / Pause / Stop del motor.
- Cartera: USDT, activos spot, equity, PnL.
- Órdenes y trades.
- Gráfico de velas con TradingView Lightweight Charts.
- Selector símbolo + temporalidad.
- Modo LIVE vs HIST + botón **Volver a tiempo real**.
- WebSocket backend→frontend.
- Persistencia SQLite (`mini_exchange.db`) para órdenes/trades/equity.

## Instalación
```bash
cd mini_exchange_ui
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración
Variables `.env`/entorno:
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `APP_SYMBOL` (default `SOLUSDT`)
- `APP_INTERVAL` (default `1h`)

Si no hay keys, usa modo mock (igual se ve la UI viva).

## Ejecutar
```bash
cd mini_exchange_ui
uvicorn backend.main:app --reload --port 8085
```
Abrir `http://localhost:8085`.

## Controles
- Cambiar símbolo: selector superior.
- Cambiar temporalidad: selector superior.
- LIVE/HIST: toggle `LIVE`.
- Volver al presente: botón `Volver a tiempo real`.
- Estado bot: badge superior (`IDLE/RUNNING/PAUSED/ERROR`).

## Notas
- El módulo `backend/bot_engine.py` encapsula bot existente (`start/pause/stop/get_state`).
- Recomendado conectar aquí tu instancia real de `SpotBot` para ejecución operativa real.
