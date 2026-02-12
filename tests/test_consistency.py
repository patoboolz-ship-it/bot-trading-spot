import importlib.util

import pytest

if importlib.util.find_spec("pandas") is None:
    pytest.skip("pandas not available", allow_module_level=True)

import pandas as pd

import bot


def make_candles(count: int = 120):
    candles = []
    base = 100.0
    for i in range(count):
        close = base + (i * 0.2) - (0.5 if i % 7 == 0 else 0.0)
        open_price = close - 0.1
        high = close + 0.15
        low = close - 0.2
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


def rsi_tv(series: pd.Series, period: int):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def macd_hist_tv(series: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - macd_signal


def test_simulator_matches_tradingview_formulas():
    candles = make_candles()
    df = pd.DataFrame(candles)
    close = df["close"]

    rsi_ref = rsi_tv(close, 14)
    rsi_sim = bot.rsi_wilder_series(close.tolist(), 14)
    for ref, sim in zip(rsi_ref.tolist(), rsi_sim):
        assert abs(ref - sim) < 1e-3

    macd_ref = macd_hist_tv(close, 12, 26, 9)
    macd_sim = bot.macd_hist_series(close.tolist(), 12, 26, 9)
    for ref, sim in zip(macd_ref.tolist(), macd_sim):
        assert abs(ref - sim) < 1e-6

    params = bot.DEFAULT_GEN.copy()
    params.update(
        {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        }
    )
    ge = bot.Genome(**params)

    rsi_vals = rsi_ref.tolist()
    macd_vals = macd_ref.tolist()
    for i in range(1, len(close) - 1):
        rsi_buy = 1.0 if rsi_vals[i] <= ge.rsi_oversold else 0.0
        rsi_sell = 1.0 if rsi_vals[i] >= ge.rsi_overbought else 0.0
        macd_buy = 1.0 if macd_vals[i] > 0 else 0.0
        macd_sell = 1.0 if macd_vals[i] < 0 else 0.0
        consec_buy = 1.0 if close[i] <= close[i - 1] else 0.0
        consec_sell = 1.0 if close[i] >= close[i - 1] else 0.0
        wbr, wbm, wbc = bot.normalize3(ge.w_buy_rsi, ge.w_buy_macd, ge.w_buy_consec)
        wsr, wsm, wsc = bot.normalize3(ge.w_sell_rsi, ge.w_sell_macd, ge.w_sell_consec)
        buy_ref = wbr * rsi_buy + wbm * macd_buy + wbc * consec_buy
        sell_ref = wsr * rsi_sell + wsm * macd_sell + wsc * consec_sell

        rsi_sim_val = rsi_sim[i]
        macd_sim_val = macd_sim[i]
        rsi_buy_sim = 1.0 if rsi_sim_val <= ge.rsi_oversold else 0.0
        rsi_sell_sim = 1.0 if rsi_sim_val >= ge.rsi_overbought else 0.0
        macd_buy_sim = 1.0 if macd_sim_val > 0 else 0.0
        macd_sell_sim = 1.0 if macd_sim_val < 0 else 0.0
        consec_buy_sim = 1.0 if close[i] <= close[i - 1] else 0.0
        consec_sell_sim = 1.0 if close[i] >= close[i - 1] else 0.0
        buy_sim = wbr * rsi_buy_sim + wbm * macd_buy_sim + wbc * consec_buy_sim
        sell_sim = wsr * rsi_sell_sim + wsm * macd_sell_sim + wsc * consec_sell_sim

        assert abs(buy_ref - buy_sim) < 1e-6
        assert abs(sell_ref - sell_sim) < 1e-6

        buy_ref_cond = buy_ref >= ge.buy_th
        sell_ref_cond = sell_ref >= ge.sell_th
        buy_sim_cond = buy_sim >= ge.buy_th
        sell_sim_cond = sell_sim >= ge.sell_th
        assert buy_ref_cond == buy_sim_cond
        assert sell_ref_cond == sell_sim_cond


def test_closed_candle_criterion_matches_spec():
    interval = "1h"
    interval_ms = bot.interval_to_ms(interval)
    server_time = 10_000_000
    candles = [
        {"close_time": server_time - (2 * interval_ms), "close": 1.0},
        {"close_time": server_time - interval_ms - 1, "close": 2.0},
        {"close_time": server_time - interval_ms, "close": 3.0},
        {"close_time": server_time - interval_ms + 1, "close": 4.0},
    ]

    closed = bot.filter_closed_candles(candles, interval, server_time)
    assert [c["close"] for c in closed] == [1.0, 2.0]


def test_bot_simulator_score_parity_on_same_closed_candle():
    candles = make_candles(300)
    ge = bot.Genome(**bot.DEFAULT_GEN)
    bot.assert_bot_simulator_score_parity(ge, candles, index=len(candles) - 2)
