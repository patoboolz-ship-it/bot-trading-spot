# Bloque de notas: Cómo replicar el bot en TradingView (RSI + MACD + Consecutivas)

Este documento está hecho para **revisar configuración** antes de tocar más código.
La idea es que puedas copiar estas reglas a TradingView y verificar por qué a veces “no opera”.

---

## 1) Qué estrategia usa realmente el bot

La estrategia no es “RSI solo” ni “MACD solo”.
Usa un **score ponderado** de 3 componentes para compra y 3 para venta:

- RSI
- MACD histograma
- Velas consecutivas (subidas/bajadas)

Ecuaciones exactas:

```python
BUY_SCORE  = Wbuy_rsi * RSI_buy + Wbuy_macd * MACD_buy + Wbuy_consec * CONSEC_buy
SELL_SCORE = Wsell_rsi * RSI_sell + Wsell_macd * MACD_sell + Wsell_consec * CONSEC_sell
```

Luego:

- Compra si `BUY_SCORE >= buy_th`
- Venta si `SELL_SCORE >= sell_th`

Además hay:

- `cooldown` por velas
- salida por `take_profit` o `stop_loss`
- opción `use_ha` (Heikin Ashi) para cálculo de señales

---

## 2) Señales base (0 o 1)

### 2.1 RSI

- `buy_rsi = 1` cuando RSI está en/por debajo de `rsi_oversold`
- `sell_rsi = 1` cuando RSI está en/por encima de `rsi_overbought`

> Nota: en el código hay una forma continua para RSI (normalizada 0..1), no solo binaria dura.

### 2.2 MACD (histograma)

Si `edge_trigger = 0`:

- `buy_macd = 1` cuando histograma > 0
- `sell_macd = 1` cuando histograma < 0

Si `edge_trigger = 1`:

- `buy_macd = 1` solo en cruce de histograma de <=0 a >0
- `sell_macd = 1` solo en cruce de histograma de >=0 a <0

### 2.3 Consecutivas

- `buy_consec = 1` cuando se cumplen `consec_green` cierres no-decrecientes seguidos
- `sell_consec = 1` cuando se cumplen `consec_red` cierres no-crecientes seguidos

---

## 3) Parámetros que DEBES igualar en TradingView

## 3.1 Timeframe

- Debe ser el mismo que `INTERVAL` del bot (ejemplo típico: `1h`).

## 3.2 Fuente de precio

Si `use_ha = 1`:

- Debes usar **Heikin Ashi** también para RSI/MACD/consecutivas.
- Si en TradingView usas velas normales pero el bot está en HA, te dará señales distintas.

Si `use_ha = 0`:

- usa velas normales en ambos lados.

## 3.3 RSI (TradingView)

Configurar RSI con:

- Length = `rsi_period`
- Source = close (o ha_close si estás en HA)
- Umbrales de estrategia:
  - Oversold = `rsi_oversold`
  - Overbought = `rsi_overbought`

## 3.4 MACD (TradingView)

Configurar MACD con:

- Fast Length = `macd_fast`
- Slow Length = `macd_slow`
- Signal Smoothing = `macd_signal`
- Señal usa el **histograma MACD**

## 3.5 Umbrales de decisión

- `buy_th` (umbral de compra del score)
- `sell_th` (umbral de venta del score)

## 3.6 Pesos (muy importante)

Compra:

- `w_buy_rsi`
- `w_buy_macd`
- `w_buy_consec`

Venta:

- `w_sell_rsi`
- `w_sell_macd`
- `w_sell_consec`

Estos pesos se normalizan internamente (suma 1 por lado), pero igual debes usar los mismos números del gen ganador.

---

## 4) Por qué puede “no operar” aunque parezca que está bien

1. **`buy_th` muy alto** para la combinación de señales/pesos.
2. **`sell_th` muy alto** y deja posiciones abiertas demasiado tiempo (o casi no vende).
3. **`edge_trigger=1`** reduce mucho señales MACD (solo cruces, no estado >0/<0).
4. **`cooldown`** bloquea entradas consecutivas.
5. **`use_ha` desalineado** entre bot y TradingView.
6. **Timeframe distinto** entre bot y gráfico.
7. En bot real, señales sólo en **vela cerrada**, no intrabar.

---

## 5) Checklist rápido para copiar a TradingView

1. Mismo símbolo (ej. `SOLUSDT`).
2. Mismo timeframe (`INTERVAL`).
3. Misma fuente de precio (`use_ha`).
4. RSI: period/overbought/oversold exactos.
5. MACD: fast/slow/signal exactos.
6. Misma lógica de score ponderado + umbrales `buy_th`/`sell_th`.
7. Evaluar solo al cierre de vela.
8. Aplicar `cooldown`, TP y SL.

---

## 6) Parámetros base actuales (DEFAULT_GEN en el código)

```text
use_ha=1
rsi_period=10
rsi_oversold=40.0
rsi_overbought=58.0
macd_fast=40
macd_slow=83
macd_signal=46
consec_red=3
consec_green=4
w_buy_rsi=0.51
w_buy_macd=0.49
w_buy_consec=0.00
buy_th=0.74
w_sell_rsi=0.21
w_sell_macd=0.46
w_sell_consec=0.33
sell_th=0.62
take_profit=0.109
stop_loss=0.012
cooldown=1
edge_trigger=0
```

Si el optimizador te entrega otro GEN, ese reemplaza estos valores.

---

## 7) Fragmentos de código clave (explicados)

### Criterio de vela cerrada

```python
return int(close_time_ms) < int(server_time_ms) - int(interval_ms)
```

Interpretación: una vela se considera cerrada para señales sólo si su `close_time` está al menos un intervalo detrás del tiempo de servidor.

### Cálculo de score (fuente de verdad)

```python
payload = score_components_at_index(...)
bs = payload["buy_score"]
ss = payload["sell_score"]
```

Interpretación: tanto bot como simulador deben pasar por esta misma función para no divergir.

---

## 8) Prompt sugerido para pedir validación automática a ChatGPT

“Te paso estos parámetros del GEN y quiero que me construyas un Pine Script Strategy que replique EXACTAMENTE:
- score ponderado compra/venta,
- RSI/MACD/consecutivas,
- use_ha,
- edge_trigger,
- cooldown,
- TP/SL,
- evaluación sólo en vela cerrada.
Luego compara señales barra por barra con estos datos exportados.”

---

## 9) Recomendación operativa

Antes de retocar el código de optimización:

1. Toma 1 GEN que “sí operaba”.
2. Replícalo exacto en TradingView con este checklist.
3. Verifica señales por barra (al menos 200 velas).
4. Recién después ajusta código si aún hay diferencias.

