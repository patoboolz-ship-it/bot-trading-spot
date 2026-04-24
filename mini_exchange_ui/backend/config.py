from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    api_key: str = os.getenv("BINANCE_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    symbol: str = os.getenv("APP_SYMBOL", "SOLUSDT")
    interval: str = os.getenv("APP_INTERVAL", "1h")
    fee_rate: float = float(os.getenv("APP_FEE_RATE", "0.001"))
    equity_snapshot_sec: int = int(os.getenv("APP_EQUITY_SNAPSHOT_SEC", "10"))


settings = Settings()
