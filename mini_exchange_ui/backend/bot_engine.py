from __future__ import annotations

import traceback
from typing import Any, Callable


class BotEngine:
    def __init__(self, store, on_event: Callable[[dict[str, Any]], None]):
        self.store = store
        self.on_event = on_event
        self._state = "IDLE"
        self._delegate = None

    def bind_existing_bot(self, bot_obj):
        self._delegate = bot_obj

    def start(self):
        self._state = "RUNNING"
        self.store.state.bot_state = self._state
        self.store.state.last_action = "play"
        if self._delegate and hasattr(self._delegate, "start"):
            self._delegate.start()
        self.on_event({"type": "bot.state", "payload": {"state": self._state}})

    def pause(self):
        self._state = "PAUSED"
        self.store.state.bot_state = self._state
        self.store.state.last_action = "pause"
        if self._delegate and hasattr(self._delegate, "stop"):
            # bot original no tiene pause, usamos stop para congelar
            self._delegate.stop()
        self.on_event({"type": "bot.state", "payload": {"state": self._state}})

    def stop(self):
        self._state = "IDLE"
        self.store.state.bot_state = self._state
        self.store.state.last_action = "stop"
        if self._delegate and hasattr(self._delegate, "stop"):
            self._delegate.stop()
        self.on_event({"type": "bot.state", "payload": {"state": self._state}})

    def get_state(self) -> str:
        return self._state

    def on_candle(self, candle: dict[str, Any]):
        if self._state != "RUNNING":
            return
        try:
            if self._delegate and hasattr(self._delegate, "on_tick"):
                self._delegate.on_tick(candle)
        except Exception:
            self._state = "ERROR"
            self.store.log_error(traceback.format_exc())
            self.on_event({"type": "bot.state", "payload": {"state": self._state}})
