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
- Se definieron dos modelos de simulacion para series de tiempo, uno con deriva y otro sin ella (**Geometric Brownian Motion** y **Proceso de Ornstein-Uhlenbeck**).
- Se simularion diferentes puntos de TP y SL para un modelo de trading sobre datos simulados.
- Se observo que los parametros obtenidos eran optimos para la simulacion, y funcionaron bien frente a los datos reales.

---
## Repaso
- Se simulo una serie de precios con su correspondiente deriva y se verifico que el comportamiento promedio coincidia con la ecuacion de Black-Scholes.
- Se realizo una validacion cruzada de la estrategia de trading sobre un Block_bootstrap y se obtuvieron resultados similares a los obtenidos con los datos reales.
- Se utilizaron los datos simulados y bootrap para generear intervalos de confianza.
---
# Bootstrap y Simulaciones para Cálculo de P-valores

---

## Introducción

- **Objetivo:** Explicar cómo el Bootstrap y las simulaciones se utilizan para calcular p-valores cuando no se cumplen los supuestos de normalidad.
- **Importancia:** En muchos casos, las distribuciones de los datos no son normales y los métodos paramétricos no son adecuados.

---

## Bootstrap

- **Definición:** Método de remuestreo que permite estimar la distribución de una estadística.
- **Proceso:**
  1. Generar múltiples muestras aleatorias con reemplazo del conjunto de datos original.
  2. Calcular la estadística de interés para cada muestra.
  3. Usar la distribución de estas estadísticas para hacer inferencias.

---
# Solución del Problema de los Tanques Alemanes con Bootstrap

---

## Introducción

- **Problema:** Durante la Segunda Guerra Mundial, los Aliados querían estimar el número total de tanques alemanes basándose en números de serie observados en tanques capturados.
- **Objetivo:** Estimar el número total de tanques (N) a partir de una muestra de números de serie.

---
## Enfoque Clásico

- **Estimación de N:** La fórmula clásica para estimar el número total de tanques es:
  $$
  \hat{N} = m + \frac{m}{k} - 1
  $$
  donde:
  - $m$ es el número máximo observado en la muestra.
  - $k$ es el tamaño de la muestra.

---

## Enfoque Bootstrap

- **Bootstrap:** Técnica de remuestreo que nos permite estimar la distribución de una estadística sin hacer supuestos paramétricos.
- **Proceso:**
  1. Generar múltiples muestras bootstrap a partir de la muestra original.
  2. Calcular la estadística de interés (estimación de N) para cada muestra.
  3. Usar la distribución de estas estimaciones para hacer inferencias.

---

## Paso 1: Datos de la Muestra

- **Supongamos:** Observamos los siguientes números de serie de tanques capturados: [48, 84, 39, 54, 77].

```python
# Generar datos de muestra simulando números de serie de tanques capturados
np.random.seed(1234)  # Para reproducibilidad
sample_size = 5
true_N = 100
sample_data = np.random.randint(1, true_N + 1, size=sample_size)
print(f"Datos de la muestra: {sample_data}")
```

---

## Paso 2: Estadística Observada

- **Estimación de N:** La fórmula clásica para estimar el número total de tanques es:
  $$
  \hat{N} = m + \frac{m}{k} - 1
  $$
  donde:
  - $m$ es el número máximo observado en la muestra.
  - $k$ es el tamaño de la muestra.

```python
def estimate_total_tanks(data):
    m = np.max(data)
    k = len(data)
    N_hat = m + (m / k) - 1
    return N_hat

# Estimación observada
observed_estimate = estimate_total_tanks(sample_data)
print(f"Estimación observada de N: {observed_estimate}")
```
```
Estimación observada de N: 99.8
```
---

## Paso 3: Generación de Muestras Bootstrap

- **Remuestreo:** Generar múltiples muestras con reemplazo a partir de los datos originales.

```python
# Número de muestras bootstrap
n_bootstrap = 1000

# Generar muestras bootstrap y calcular estimaciones
bootstrap_estimates = np.array([estimate_total_tanks(
  np.random.choice(
    sample_data, 
    size=len(sample_data), 
    replace=True)) 
  for _ in range(n_bootstrap)])
```

---

## Paso 4: Inferencias Estadísticas

- **Distribución de Bootstrap:** Usar la distribución de las estimaciones bootstrap para calcular el intervalo de confianza.

```python
# Intervalo de confianza del 95%
ci_lower = np.percentile(bootstrap_estimates, 2.5)
ci_upper = np.percentile(bootstrap_estimates, 97.5)

print(f"Intervalo de confianza del 95%: [{ci_lower}, {ci_upper}]")
```
```
Intervalo de confianza del 95%: [63.8, 99.8]
```
---

## Paso 5: Prueba de Hipótesis

- **P-valor:** Calcular el p-valor para la hipótesis nula de que el número total de tanques es 95 o mas.


```python
p_value = np.sum(bootstrap_estimates < 95) / n_bootstrap
print(f"P-valor: {p_value}")

```
```
P-valor: 0.326
```
No existe evidencia significativa para rechazar la hipótesis nula de que el número total de tanques es 95 o más.

---

## Conclusión

- **Bootstrap**: Método efectivo para estimar la distribución de una estadística cuando los supuestos paramétricos no son aplicables.
- **Ventaja**: No requiere suposiciones sobre la distribución subyacente de los datos.
- **Aplicación**: Útil en problemas de estimación donde se dispone de muestras limitadas.

---

## Recapitulación

- Bootstrap y simulacion son herramientas para la generacion de datos sinteticos y la validacion de modelos.
- Se pueden utilizar para calcular intervalos de confianza y p-valores.
- Se pueden aplicar a problemas de estimacion y optimizacion numerica.

---

# Introducción a Dynamic Time Warping (DTW)

---
## Métricas Basadas en Alineación

- **Objetivo:** Evaluar la similitud entre series temporales.
- **Definición:** Una serie temporal es una secuencia de características: $X = (x_1, x_2, \ldots, x_T)$.
- **Métrica de alineación:** Utiliza una alineación temporal para evaluar la similitud.

---

## Comparación entre DTW y Distancia Euclidiana

- **Distancia Euclidiana:** Calcula la suma de las distancias entre puntos con el mismo índice temporal.
- **Dynamic Time Warping (DTW):** Busca la alineación temporal que minimiza la distancia entre series.

![dtw_vs_euclidean](dtw_vs_euclidean.png)

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