import importlib.util
import math

import pytest

if importlib.util.find_spec("binance") is None or importlib.util.find_spec("matplotlib") is None:
    pytest.skip("Optional dependencies not available", allow_module_level=True)

import bot


def make_candles(count: int = 260):
    candles = []
    base = 100.0
    for i in range(count):
        close = base + i * 0.1
        open_price = close - 0.05
        high = close + 0.05
        low = close - 0.1
        t = 1_700_000_000_000 + i * 60_000
        candles.append(
            {
                "t": t,
                "open_time": t,
                "close_time": t + 60_000,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            }
        )
    return candles


def test_scores_and_decisions_match_between_bot_and_simulator():
    candles = make_candles()
    params = bot.DEFAULT_GEN.copy()

    payload_params, close_params = bot.compute_score_snapshot_from_params(params, candles)
    ge = bot.Genome(**params)
    payload_genome, close_genome = bot.compute_score_snapshot_from_genome(ge, candles)

    assert math.isclose(close_params, close_genome, rel_tol=1e-9)
    assert math.isclose(payload_params["buy_score"], payload_genome["buy_score"], rel_tol=1e-9)
    assert math.isclose(payload_params["sell_score"], payload_genome["sell_score"], rel_tol=1e-9)

    for key in payload_params["signals"].keys():
        assert math.isclose(
            payload_params["signals"][key],
            payload_genome["signals"][key],
            rel_tol=1e-9,
        )

    buy_params = payload_params["buy_score"] >= float(params["buy_th"])
    sell_params = payload_params["sell_score"] >= float(params["sell_th"])
    buy_genome = payload_genome["buy_score"] >= ge.buy_th
    sell_genome = payload_genome["sell_score"] >= ge.sell_th

    assert buy_params == buy_genome
    assert sell_params == sell_genome
