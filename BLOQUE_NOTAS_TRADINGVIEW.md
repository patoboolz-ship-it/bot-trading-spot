# Bloque de notas: cómo replicar el bot en TradingView (RSI + MACD + Consecutivas)

Este documento explica **exactamente** cómo decide entradas/salidas el bot para que lo puedas clonar en TradingView sin “configuraciones ocultas”.

---

## 1) Lógica real del bot (resumen corto)

La estrategia es un **score ponderado** por lado:

- Compra = mezcla de señal RSI + señal MACD + señal de velas consecutivas.
- Venta = mezcla de señal RSI + señal MACD + señal de velas consecutivas.

Luego se compara con umbrales:

- si `BUY_SCORE >= buy_th` ⇒ condición de compra;
- si `SELL_SCORE >= sell_th` ⇒ condición de venta.

Además intervienen:

- `cooldown` (espera mínima entre entradas),
- `take_profit` y `stop_loss`,
- `edge_trigger` (MACD por estado o por cruce),
- `use_ha` (si se calcula sobre Heikin-Ashi o vela normal),
- y **evaluación solo al cierre de vela cerrada**.

---

## 2) Ecuaciones exactas (pesos + umbrales)

### 2.1 Score de compra

```text
BUY_SCORE = w_buy_rsi   * RSI_buy_component
          + w_buy_macd  * MACD_buy_component
          + w_buy_consec* CONSEC_buy_component
```

Disparo de compra:

```text
BUY_SCORE >= buy_th
```

### 2.2 Score de venta

```text
SELL_SCORE = w_sell_rsi    * RSI_sell_component
           + w_sell_macd   * MACD_sell_component
           + w_sell_consec * CONSEC_sell_component
```

Disparo de venta:

```text
SELL_SCORE >= sell_th
```

### 2.3 Qué significan los pesos

- `w_buy_rsi`, `w_buy_macd`, `w_buy_consec`: cuánto “aporta” cada familia al score de compra.
- `w_sell_rsi`, `w_sell_macd`, `w_sell_consec`: cuánto “aporta” cada familia al score de venta.
- Internamente se normalizan por lado (la suma efectiva de pesos por compra y por venta queda 1), pero debes usar los valores exactos del GEN ganador para que el balance relativo quede igual.

---

## 3) Componentes de señal (RSI/MACD/Consecutivas)

## 3.1 RSI

Parámetros:

- `rsi_period`
- `rsi_oversold`
- `rsi_overbought`

Interpretación operativa:

- lado compra se activa cuando RSI entra zona baja (oversold),
- lado venta se activa cuando RSI entra zona alta (overbought).

> En el código, la contribución puede trabajar en forma continua (no siempre 0/1 duro), así que en TradingView conviene replicar fórmula, no simplificarla.

## 3.2 MACD (histograma)

Parámetros:

- `macd_fast`
- `macd_slow`
- `macd_signal`
- `edge_trigger`

Si `edge_trigger = 0` (modo estado):

- compra favorecida cuando histograma > 0,
- venta favorecida cuando histograma < 0.

Si `edge_trigger = 1` (modo cruce):

- compra favorecida **solo en la barra del cruce** de hist ≤ 0 a hist > 0,
- venta favorecida **solo en la barra del cruce** de hist ≥ 0 a hist < 0.

Este switch cambia muchísimo la frecuencia de entradas.

## 3.3 Velas consecutivas

Parámetros:

- `consec_green` para compra,
- `consec_red` para venta.

Interpretación:

- componente de compra sube cuando se cumplen `consec_green` cierres consecutivos alcistas/no-decrecientes,
- componente de venta sube cuando se cumplen `consec_red` cierres consecutivos bajistas/no-crecientes.

---

## 4) Cierre de vela (clave para que TradingView coincida)

El bot/simulador evalúa señales **solo con vela cerrada**, no intrabar.

Criterio implementado:

```python
close_time < server_time - interval_ms
```

O sea, una vela entra al motor de señales únicamente cuando ya quedó atrás del reloj de servidor por al menos 1 intervalo.

En TradingView debes forzar evaluación al cierre de barra (`barstate.isconfirmed`) para evitar entradas fantasmas que luego desaparecen.

---

## 5) Orden de decisión por vela (pipeline mental)

En cada vela cerrada:

1. Se calcula fuente de precio (normal o HA según `use_ha`).
2. Se actualizan RSI/MACD/consecutivas.
3. Se construye `BUY_SCORE` y `SELL_SCORE` con los pesos.
4. Se comparan scores con `buy_th` y `sell_th`.
5. Se aplican filtros de estado (`cooldown`, posición abierta/cerrada).
6. Si hay posición abierta, pueden salir por `SELL_SCORE`, `take_profit`, `stop_loss`.

Si tú en TradingView cambias el orden, o evalúas intrabar, o no respetas cooldown, ya no coincide.

---

## 6) Qué debes configurar en TradingView (checklist exacto)

1. Mismo símbolo (ejemplo `SOLUSDT`).
2. Mismo timeframe (`INTERVAL`).
3. Misma fuente de precio:
   - `use_ha = 1` ⇒ indicadores y lógica sobre Heikin-Ashi.
   - `use_ha = 0` ⇒ indicadores y lógica sobre velas normales.
4. RSI:
   - Length = `rsi_period`
   - Oversold = `rsi_oversold`
   - Overbought = `rsi_overbought`
5. MACD:
   - Fast = `macd_fast`
   - Slow = `macd_slow`
   - Signal = `macd_signal`
   - usar histograma y respetar `edge_trigger`
6. Consecutivas:
   - `consec_green`
   - `consec_red`
7. Pesos compra:
   - `w_buy_rsi`, `w_buy_macd`, `w_buy_consec`
8. Pesos venta:
   - `w_sell_rsi`, `w_sell_macd`, `w_sell_consec`
9. Umbrales:
   - `buy_th`
   - `sell_th`
10. Gestión:
   - `cooldown`
   - `take_profit`
   - `stop_loss`
11. Ejecutar señales solo al cierre de vela.

---

## 7) Por qué a veces “no opera nada”

Causas típicas:

- `buy_th` demasiado alto para la combinación de pesos.
- `w_buy_consec` o `w_buy_macd` casi en 0 deja una pata “apagada”.
- `edge_trigger=1` reduce mucho disparos MACD.
- `consec_green`/`consec_red` altos para ese timeframe.
- `cooldown` demasiado largo.
- `use_ha` diferente entre bot y TradingView.
- Estás mirando señales intrabar y no al cierre.

---

## 8) Mapa rápido de parámetros del GEN → panel TradingView

- RSI Length ← `rsi_period`
- RSI Oversold ← `rsi_oversold`
- RSI Overbought ← `rsi_overbought`
- MACD Fast ← `macd_fast`
- MACD Slow ← `macd_slow`
- MACD Signal ← `macd_signal`
- Buy weight RSI ← `w_buy_rsi`
- Buy weight MACD ← `w_buy_macd`
- Buy weight Consecutive ← `w_buy_consec`
- Sell weight RSI ← `w_sell_rsi`
- Sell weight MACD ← `w_sell_macd`
- Sell weight Consecutive ← `w_sell_consec`
- Buy threshold ← `buy_th`
- Sell threshold ← `sell_th`
- Consecutive green bars ← `consec_green`
- Consecutive red bars ← `consec_red`
- MACD edge mode ← `edge_trigger`
- Use Heikin-Ashi ← `use_ha`
- Cooldown bars ← `cooldown`
- Take Profit ← `take_profit`
- Stop Loss ← `stop_loss`

---

## 9) Parámetros base actuales (DEFAULT_GEN de referencia)

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

Si el optimizador entrega otro GEN, usa ese (este bloque solo es punto de partida).

---

## 10) Mini plantillas de validación (para comparar con TradingView)

### 10.1 Validación de score por barra

Para una barra histórica X, registra y compara:

- `RSI_buy_component`, `MACD_buy_component`, `CONSEC_buy_component`
- `RSI_sell_component`, `MACD_sell_component`, `CONSEC_sell_component`
- `BUY_SCORE`, `SELL_SCORE`
- decisión final (buy/sell/hold)

Si estos números empatan barra por barra, la implementación está alineada.

### 10.2 Validación de cierre de vela

Comprobar que la barra usada por estrategia cumple criterio de cerrada y que no se usa la vela en formación.

---

## 11) Prompt sugerido para pedir Pine Script exacto

"Construye una estrategia Pine v5 que replique 1:1 esta lógica:
- score ponderado compra/venta con RSI, MACD-hist y consecutivas,
- pesos `w_buy_*` y `w_sell_*`,
- umbrales `buy_th` y `sell_th`,
- modo `edge_trigger`,
- `use_ha`,
- `cooldown`, `take_profit`, `stop_loss`,
- ejecución solo en barra cerrada (`barstate.isconfirmed`).
Luego imprime en tabla por barra los componentes y scores para compararlos con mi bot." 

---

## 12) Si no te coincide con Python: 6 diferencias típicas (aplica a tu script)

Si usas una versión como la que compartiste, hay varios puntos que suelen romper la paridad:

1. **RSI en binario vs RSI continuo**
   - En Python, `buy_rsi_signal` y `sell_rsi_signal` son continuas (0..1 lineal entre oversold/overbought).
   - Si en Pine pones RSI como 0/1 duro, cambian `BUY_SCORE`/`SELL_SCORE` y la frecuencia de trades.

2. **Pesos sin normalizar por lado**
   - En Python se normalizan siempre (`normalize3`) antes de combinar.
   - Si en Pine usas pesos en crudo, las escalas no coinciden.

3. **Modo de ejecución de órdenes**
   - Simulador Python entra/sale en el **open de la vela siguiente** (`process_orders_on_close=false` en TradingView aproxima mejor esto).
   - Con `process_orders_on_close=true` te estás llenando al cierre de la misma vela de señal y eso altera resultados.

4. **TP/SL intrabar y prioridad**
   - En Python, al evaluar la vela siguiente: primero SL (si low toca), luego TP (si high toca), luego señal de salida.
   - En TradingView `strategy.exit` puede resolver distinto cuando una barra toca ambos extremos.

5. **Consecutivas exactas**
   - Python usa condición no estricta (`>=` para verdes y `<=` para rojas) sobre los últimos `n` saltos.
   - Un contador distinto o con resets distintos cambia bastante.

6. **Fuente HA**
   - Python usa `haClose=(o+h+l+c)/4` para señales cuando `use_ha=1`.
   - Si en TradingView tomas otra variante de HA (o `close` normal), ya no es 1:1.

> Para acelerar, usa `tradingview_parity_template.pine` como base y solo cambia parámetros del GEN.

