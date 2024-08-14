---
marp: true
theme: default
class: invert
paginate: true
math: katex
---

# Análisis de Precios de Bitcoin

### Paquetes Utilizados

- **quantmod**: Para obtener y manejar datos financieros.
- **strucchange**: Para realizar pruebas de cambio estructural.
- **urca**: Para análisis de raíces unitarias.
- **changepoint**: Para detectar puntos de cambio en series temporales.
- **WaveletComp**: Para análisis de espectros de potencia utilizando wavelets.

---

## Obtener Datos de Bitcoin

```r
getSymbols("BTC-USD", src = "yahoo", from = "2014-01-01", to = "2024-01-01")
btc_prices <- Cl(`BTC-USD`)
plot.ts(btc_prices)
```

- **getSymbols()**: Obtiene datos históricos de precios de Bitcoin desde Yahoo Finance.
- **Cl()**: Extrae los precios de cierre ajustados.
- **plot.ts()**: Grafica la serie temporal de precios de Bitcoin.

---

## Prueba CUSUM

```r
cusum_test <- efp(btc_prices ~ lag(btc_prices),type = "OLS-CUSUM")
plot(cusum_test, main = "Prueba CUSUM en Precios de Bitcoin")
```

- **CUSUM (Cumulative Sum Control Chart)**: Detecta cambios graduales en la media de la serie temporal.
- **efp()**: Realiza la prueba CUSUM para detectar cambios en la estructura de la regresión.
- **plot()**: Grafica los resultados de la prueba.

---

## Prueba de Chow

```r
breakpoint_index <- floor(length(btc_prices) / 2)
chow_test <- sctest(btc_prices ~ lag(btc_prices), type = "Chow", point = breakpoint_index)
chow_test_result <- chow_test$p.value
print(paste("Chow Test p-value:", chow_test_result))
```

- **Prueba de Chow**: Detecta si existe un cambio estructural en un punto específico de la serie.
- **breakpoint_index**: Se selecciona el punto medio de la serie como posible punto de cambio.
- **sctest()**: Realiza la prueba de Chow.
- **p-value**: Indica la significancia del cambio detectado.

---

## Prueba Bai-Perron

```r
bp_test <- breakpoints(btc_prices ~ lag(btc_prices), breaks = 5)
summary(bp_test)
plot(bp_test, main = "Prueba Bai-Perron en Precios de Bitcoin")
```

- **Prueba Bai-Perron**: Identifica múltiples puntos de cambio en la serie temporal.
- **breakpoints()**: Estima los puntos de cambio óptimos para hasta 5 segmentos.
- **plot()**: Grafica los puntos de cambio detectados en la serie.

---

## Análisis Wavelet

```r
btc_df <- data.frame(date = index(btc_prices), returns = as.vector(btc_prices))
wavelet_analysis <- analyze.wavelet(btc_df, my.series = "returns", loess.span = 0)
wt.image(wavelet_analysis, 
         main = "Espectro de Potencia de las Retornos de Bitcoin",
         spec.time.axis = list(
           at = c(which(index(btc_prices)=="2016-07-09"),
                  which(index(btc_prices)=="2020-05-11"),
                  which(index(btc_prices)=="2023-04-23")
                  ), 
           labels = c("2016-07-09","2020-05-11","2023-04-23")))
```

- **analyze.wavelet()**: Realiza un análisis de espectro de potencia utilizando wavelets.
- **wt.image()**: Grafica el espectro de potencia de los retornos de Bitcoin, mostrando cómo las frecuencias dominantes cambian a lo largo del tiempo.

---

## Detección de Puntos de Quiebre con `cpt.meanvar`

```r
mv_pelt <- cpt.meanvar(as.vector(btc_prices), method = "BinSeg",Q=5)
ts.plot(btc_prices, main = "Bitcoin con Puntos de Quiebre Detectados", ylab = "Precio (USD)", xlab = "Fecha")
abline(v = cpts(mv_pelt), col = "red", lty = 2)
abline(v=c(which(index(btc_prices)=="2016-07-09"),
           which(index(btc_prices)=="2020-05-11"),
           which(index(btc_prices)=="2023-04-23")),
       col="steelblue")
```

- **cpt.meanvar()**: Detecta cambios en la media y varianza utilizando el método BinSeg.
- **abline()**: Añade líneas verticales para indicar los puntos de cambio detectados.
- **ts.plot()**: Grafica los precios de Bitcoin junto con los puntos de quiebre identificados.

---

## Conclusión

- Este análisis aplica varias pruebas y métodos para detectar cambios estructurales en los precios de Bitcoin.
- Se utilizan diferentes enfoques para identificar puntos de cambio significativos en la serie temporal.
- Las pruebas incluyen CUSUM, Chow, Bai-Perron y análisis de espectro de potencia con wavelets, proporcionando una visión completa de los cambios en la dinámica de los precios de Bitcoin a lo largo del tiempo.
