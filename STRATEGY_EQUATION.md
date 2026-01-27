# Ecuación de scoring y consistencia

## Señales (rango)

Todas las señales se normalizan a rango **0..1**:

- `RSI_signal` (buy/sell) usa `buy_rsi_signal` / `sell_rsi_signal`.
- `MACD_signal` (buy/sell) usa `buy_macd_signal` / `sell_macd_signal`.
- `CONSEC_signal` (buy/sell) usa `consec_up` / `consec_down`.

## Ecuación (fuente de verdad)

```
BUY_SCORE  = Wbuy_rsi  * RSI_buy  + Wbuy_macd  * MACD_buy  + Wbuy_consec  * CONSEC_buy
SELL_SCORE = Wsell_rsi * RSI_sell + Wsell_macd * MACD_sell + Wsell_consec * CONSEC_sell
```

Los pesos se normalizan para que cada trío sume 1.

## Dónde se implementa

- **Bot (ejecución real)**: `compute_scores` delega en `compute_score_snapshot_from_params`, que llama a `score_components_at_index`.
- **Simulador/backtest**: `simulate_spot` usa `score_components_at_index` para cada vela cerrada.
- **Optimizador (GA)**: el fitness se basa en `simulate_spot`, por lo tanto usa la misma ecuación.
