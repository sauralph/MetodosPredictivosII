---
marp: true
theme: default
class: invert
paginate: true
math: katex
---

# Aplicaciones de Datos Sintéticos en la Creación y Validación de Modelos Predictivos

---
## Repaso

---
## 


---
# Introducción a Dynamic Time Warping (DTW)

---
## Dynamic Time Warping (DTW)

- **Objetivo:** Introducir DTW y su aplicación en la alineación de series temporales.
- **Teoría:**
  - Definición y formulación matemática de DTW.
  - Casos de uso en finanzas para alinear movimientos de precios.
- **Implementación:**

---

```python
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

distance, path = fastdtw(series1, series2, dist=euclidean)
plt.plot(series1)
plt.plot([x[1] for x in path])
```
---

## Métricas Basadas en Alineación

- **Objetivo:** Evaluar la similitud entre series temporales.
- **Definición:** Una serie temporal es una secuencia de características: $X = (x_1, x_2, \ldots, x_T)$.
- **Métrica de alineación:** Utiliza una alineación temporal para evaluar la similitud.

---

## Comparación entre DTW y Distancia Euclidiana

- **Distancia Euclidiana:** Calcula la suma de las distancias entre puntos con el mismo índice temporal.
- **Dynamic Time Warping (DTW):** Busca la alineación temporal que minimiza la distancia entre series.

---

---

## Problema de Formulación de DTW

- **Series temporales:** $X = (x_1, x_2, \ldots, x_T)$ y $Y = (y_1, y_2, \ldots, y_U)$.
- **Objetivo:** Encontrar la alineación que minimice la distancia Euclidiana acumulada.
- **Función de costo:**
  $$
  DTW(X, Y) = \min_{\pi} \sum_{(i,j) \in \pi} d(x_i, y_j)
  $$
  donde $\pi$ es una secuencia de pares de índices que representa la alineación.

---

## Solución Algorítmica

- **Programación dinámica:** Resuelve el problema de alineación de forma eficiente.
- **Recurrencia:**
  $$
  DTW(i, j) = d(x_i, y_j) + \min \begin{cases}
    DTW(i-1, j) \\
    DTW(i, j-1) \\
    DTW(i-1, j-1)
  \end{cases}
  $$
- **Implementación en Python:**

```python
def dtw(x, y):
    n, m = len(x), len(y)
    dtw_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            cost = abs(x[i] - y[j])
            if i == 0 and j == 0:
                dtw_matrix[i, j] = cost
            elif i == 0:
                dtw_matrix[i, j] = cost + dtw_matrix[i, j-1]
            elif j == 0:
                dtw_matrix[i, j] = cost + dtw_matrix[i-1, j]
            else:
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])

    return dtw_matrix[-1, -1]
```

---

## Propiedades de DTW

- **Invarianza a desplazamientos temporales:** DTW puede alinear patrones similares que ocurren en diferentes momentos.
- **Alineación flexible:** Puede manejar series de diferentes longitudes y velocidades.

---

## Restricciones Adicionales

- **Banda de Sakoe-Chiba:** Limita la alineación a una banda alrededor de la diagonal.
- **Paralelogramo de Itakura:** Limita la pendiente máxima de la alineación.

---

## Conclusión

- **Uso de DTW:** Útil para comparar series temporales con desplazamientos y deformaciones temporales.
- **Ventajas sobre la Distancia Euclidiana:** Proporciona una evaluación de similitud más precisa para series temporales.

---

---
## Análisis de Transformada Wavelet

- **Objetivo:** Introducir la Transformada Wavelet y su aplicación en el análisis de series temporales.
- **Teoría:**
  - Transformada Wavelet Continua (CWT) vs. Transformada Wavelet Discreta (DWT).
  - Casos de uso en finanzas para detectar patrones en diferentes escalas.
- **Implementación:**

```python
import pywt
import numpy as np
import matplotlib.pyplot as plt

coeffs, freqs = pywt.cwt(prices, scales, 'morl')
plt.imshow(coeffs, extent=[0, len(prices), 1, 31], cmap='PRGn', aspect='auto', vmax=abs(coeffs).max(), vmin=-abs(coeffs).max())
plt.show()
```

---

## Estrategias de Trading

- **Objetivo:** Introducir e implementar diversas estrategias de trading.

### Estrategia HODL
- **Código:**

```r
hodl_return <- (as.numeric(prices[length(prices)]) - as.numeric(prices[1])) / as.numeric(prices[1])
cat("HODL Cumulative Return: ", hodl_return * 100, "%\n")
```

### Estrategia de Cruce de Medias Móviles (SMA)
- **Código:**

```r
short_ma <- SMA(prices, n = 2)
long_ma <- SMA(prices, n = 30)
long_signals_sma <- which(short_ma > long_ma & lag(short_ma, 1) <= lag(long_ma, 1))
...
```

### Estrategia RSI
- **Código:**

```r
rsi <- RSI(prices, n = 14)
long_signals_rsi <- which(rsi < 30 & lag(rsi, 1) >= 30)
...
```

### Estrategia de Bandas de Bollinger
- **Código:**

```r
bbands <- BBands(prices, n = 20, sd = 2)
long_signals_bbands <- which(prices < bbands[, "dn"] & lag(prices, 1) >= lag(bbands[, "dn"], 1))
...
```

### Estrategia de Filtro CUSUM
- **Código:**

```r
cusum_filter <- function(prices, threshold) {
  ...
}
long_signals_cusum <- cusum_filter(prices, threshold)
...
```

---

## Optimización de Portafolio con Métodos Numéricos

- **Objetivo:** Introducir e implementar la optimización de portafolios usando algoritmos genéticos.
- **Teoría:**
  - Definir la función objetivo para la optimización.
  - Uso de la biblioteca DEAP para algoritmos evolutivos.
- **Implementación:**

```python
from deap import base, creator, tools, algorithms
...
def run():
    ...
result = run()

w0 = weights(result)

print("Optimal weights:", w0)
print("Expected return:", portfolio_return(w0))
print("Expected volatility:", portfolio_volatility(w0))
```

---

## Conclusión

- **Recapitulación:** Resumir los puntos clave sobre la generación de datos sintéticos, DTW, Transformada Wavelet, estrategias de trading y optimización de portafolio.
- **Q&A:** Espacio para preguntas y aclaraciones.

---

# Recursos Adicionales

- **Libros:**
  - *Advances in Financial Machine Learning* por Marcos López de Prado (Capítulo 13)
- **Online:**
  - [Repositorio en Github](https://github.com/sauralph/MetodosPredictivosII/tree/main/Synthetic%20Data%20and%20Validation) para código adicional y ejemplos.
  - Romain Tavenard, "An introduction to Dynamic Time Warping", 2021. [Link](https://rtavenar.github.io/blog/dtw.html)

---