# Paquetes
library(quantmod)
library(dplyr)
library(ggplot2)
library(tidyr)

# Descargar datos BTC últimos 365 días
getSymbols("BTC-USD", src = "yahoo", from = Sys.Date() - 365, auto.assign = TRUE)
btc <- Cl(`BTC-USD`)
btc <- na.omit(btc)

# Calcular retornos logarítmicos diarios
log_ret <- dailyReturn(btc, type = "log")
log_ret <- na.omit(log_ret)

# Dividir en entrenamiento (días 1-180) y test (días 181-360)
ret_train <- head(log_ret, 180)
btc_test <- tail(btc, 180)

# Estimar parámetros GBM
mu <- mean(ret_train)
sigma <- sd(ret_train)

# Usar precio inicial para simulación: último día del set de entrenamiento
S0 <- as.numeric(btc[index(ret_train)[180]])

# Simulación GBM: 10.000 caminos de 180 días
set.seed(42)
n_days <- 180
n_sims <- 1000
dt <- 1

simulate_gbm_prices <- function(S0, mu, sigma, n_days, n_sims, dt) {
  drift <- (mu - 0.5 * sigma^2) * dt
  diffusion <- sigma * sqrt(dt)
  Z <- matrix(rnorm(n_days * n_sims), nrow = n_days, ncol = n_sims)
  log_returns <- drift + diffusion * Z
  log_prices <- apply(log_returns, 2, cumsum)
  prices <- S0 * exp(log_prices)
  prices <- rbind(rep(S0, n_sims), prices)  # incluir día 0
  return(prices)
}

gbm_prices <- simulate_gbm_prices(S0, mu, sigma, n_days, n_sims, dt)

# Estadísticas por día (media y percentiles 2.5% y 97.5%)
gbm_stats <- apply(gbm_prices, 1, function(x) {
  c(mean = mean(x),
    lower = quantile(x, 0.025),
    upper = quantile(x, 0.975))
})
gbm_stats_df <- as.data.frame(t(gbm_stats))
gbm_stats_df$day <- 0:n_days

# Traer trayectoria real de precios test (últimos 180 días)
btc_real_vec <- as.numeric(btc_test)
btc_real_df <- data.frame(day = 0:(length(btc_real_vec) - 1), price = btc_real_vec)

colnames(gbm_stats_df)<-c("mean","lower","upper","day")

# Gráfico
ggplot(gbm_stats_df, aes(x = day)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.4) +
  geom_line(aes(y = mean), color = "blue", size = 1) +
  geom_line(data = btc_real_df, aes(x = day, y = price), color = "red", size = 1.2) +
  labs(title = "Simulación de Precios BTC con GBM desde el día 180",
       subtitle = "Rojo: trayectoria real — Azul: media simulada — Banda: IC 95%",
       x = "Día desde simulación", y = "Precio BTC (USD)") +
  theme_minimal()





