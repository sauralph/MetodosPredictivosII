---
marp: true
theme: default
class: invert
paginate: true
size: 16:9
---

# Modelos de Simulación Estocástica en Finanzas

---

## Objetivo

- Presentar los principales **procesos estocásticos** utilizados en simulación de precios de activos.
- Clasificar los modelos según sus propiedades.
- Dar ejemplos de aplicación.

---

## Clasificación General

### 1. Procesos Estocásticos Continuos  
- Basados en ecuaciones diferenciales estocásticas (SDEs).  
- Incluyen GBM, Ornstein-Uhlenbeck, CIR, Heston, etc.

### 2. Modelos Discretos y Empíricos  
- Modelos ARIMA/GARCH, Bootstrap, Copulas, etc.

### 3. Basados en Reglas o Agentes  
- Modelos basados en agentes (ABM), árboles binomiales.

### 4. Modelos Avanzados o con Saltos  
- Jumps de Merton, procesos Lévy, movimientos fraccionarios.

---

## Movimiento Browniano Geométrico (GBM)

**Ecuación:**

$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

- Modela el precio de una acción en Black-Scholes.
- Siempre positivo.
- No permite reversión a la media.

✅ Simple  
❌ No capta volatilidad variable ni shocks

---

## Proceso de Ornstein-Uhlenbeck

**Ecuación:**

$$
dX_t = \theta(\mu - X_t) dt + \sigma dW_t
$$

- Reversión a la media.
- Usado en tasas de interés, commodities, pares de trading.

✅ Permite reversión  
❌ Puede tomar valores negativos

---

## Proceso CIR (Cox-Ingersoll-Ross)

**Ecuación:**

$$
dX_t = \theta(\mu - X_t) dt + \sigma \sqrt{X_t} dW_t
$$

- Similar al OU pero **siempre positivo**.
- Usado en tasas de interés.

✅ Reversión + Positividad  
❌ Requiere parámetros válidos para estabilidad

---

## Modelo de Heston

**Modelo de volatilidad estocástica**

$$
\begin{aligned}
dS_t &= \mu S_t dt + \sqrt{v_t} S_t dW_t^S \\
dv_t &= \kappa(\theta - v_t) dt + \sigma \sqrt{v_t} dW_t^v
\end{aligned}
$$

- La volatilidad también sigue un proceso CIR.
- Útil para modelar _smile de volatilidad_ en opciones.

---

## Modelos con Saltos (Merton)

**Modelo con saltos de Poisson:**

$$
dS_t = \mu S_t dt + \sigma S_t dW_t + J_t dN_t
$$

- \( J_t \): cambio porcentual durante un salto  
- \( N_t \): proceso de Poisson

✅ Crisis / eventos  
❌ Más difícil de calibrar

---

## Modelos Econométricos

- **ARIMA / ARMA**: Autoregresivos, útiles en series estacionarias.
- **GARCH / EGARCH**: Capturan volatilidad condicional.

✅ Capturan patrones temporales  
❌ No modelan trayectorias realistas de precios

---

## Bootstrap

- **Histórico puro**: reordenamiento aleatorio de rendimientos.
- **Block Bootstrap**: bloques para mantener autocorrelación.
- **Circular Bootstrap**: preserva continuidad en bordes.

✅ No asume distribución  
❌ No genera escenarios extremos nuevos

---

## Copulas

- Modelan la **dependencia conjunta** entre activos.
- Separan la dependencia de la distribución marginal.

$$
F(x, y) = C(F_X(x), F_Y(y))
$$

✅ Útil en portafolios  
❌ Calibración compleja

---

## Simulación con Árboles

- Árboles binomiales o trinomiales discretizan posibles precios futuros.

Usos:
- Valuación de opciones
- Análisis de decisiones

✅ Intuitivos  
❌ Crecen exponencialmente con el tiempo

---

## Modelos Basados en Agentes (ABM)

- Cada “agente” sigue reglas (trader, arbitrajista, etc).
- Interacciones generan precios emergentes.

✅ Explora dinámicas realistas  
❌ No siempre reproducibles ni calibrables

---

## Modelos Avanzados

### 1. Procesos Lévy

- Generalizan el Browniano. Permiten colas pesadas, skew.

### 2. Movimiento Browniano Fraccionario

- Introduce memoria (autocorrelación) en los pasos.

✅ Capturan propiedades empíricas  
❌ Incompatibles con algunos modelos (ej. Black-Scholes)

---

## Comparación Rápida

| Modelo   | Reversión | Volatilidad | Saltos | Positivo |
|----------|-----------|-------------|--------|----------|
| GBM      | ❌        | Constante   | ❌     | ✅       |
| OU       | ✅        | Constante   | ❌     | ❌       |
| CIR      | ✅        | √X_t        | ❌     | ✅       |
| Heston   | ❌        | Estocástica | ❌     | ✅       |
| Merton   | ❌        | Constante   | ✅     | ✅       |

---

## Conclusión

- No hay modelo perfecto. Cada uno **simula ciertos fenómenos**.
- La elección del modelo depende de:
  - El activo
  - El objetivo del análisis (opciones, portafolios, stress test)
  - La disponibilidad de datos

---

## ¿Preguntas?

Gracias por su atención  
👨‍🏫

---
