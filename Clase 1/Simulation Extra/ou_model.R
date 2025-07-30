# Load packages
library(quantmod)
library(dplyr)
library(ggplot2)
library(tidyr)

# Download BTC-USD data from last 365 days
getSymbols("BTC-USD", src = "yahoo", from = Sys.Date() - 365, auto.assign = TRUE)
btc <- Cl(`BTC-USD`)
btc <- na.omit(btc)

# Calculate log-returns
log_ret <- dailyReturn(btc, type = "log")
log_ret <- na.omit(log_ret)

# Split: First 180 days for parameter estimation, last 180 for test
ret_train <- head(log_ret, 180)
btc_test <- tail(btc, 180)
log_price_train <- log(Cl(head(btc, 180)))

# Estimate OU parameters (X_t = log(price))
X <- as.numeric(log_price_train)
dt <- 1

# Estimate theta and mu using linear regression (discretized OU)
X_t <- X[-length(X)]
X_t1 <- X[-1]
model <- lm(X_t1 ~ X_t)

theta <- -log(coef(model)[2]) / dt
mu_hat <- coef(model)[1] / (1 - exp(-theta * dt))
residuals <- X_t1 - predict(model)
sigma_hat <- sd(residuals) * sqrt(2 * theta / (1 - exp(-2 * theta * dt)))

cat("Estimated OU parameters:\n")
cat("theta =", theta, "\nmu =", mu_hat, "\nsigma =", sigma_hat, "\n")

# Initial log-price (end of training period)
X0 <- as.numeric(log(Cl(btc[index(ret_train)[180]])))
n_days <- 180
n_sims <- 10000

# Simulate OU log-prices
set.seed(42)
simulate_ou_paths <- function(X0, mu, theta, sigma, n_days, n_sims, dt) {
  paths <- matrix(NA, nrow = n_days + 1, ncol = n_sims)
  paths[1, ] <- X0
  for (t in 2:(n_days + 1)) {
    Z <- rnorm(n_sims)
    paths[t, ] <- paths[t - 1, ] + theta * (mu - paths[t - 1, ]) * dt + sigma * sqrt(dt) * Z
  }
  return(paths)
}

log_paths <- simulate_ou_paths(X0, mu_hat, theta, sigma_hat, n_days, n_sims, dt)
price_paths <- exp(log_paths)  # back to price space

# Compute mean and 95% CI at each time step
stats <- apply(price_paths, 1, function(x) {
  c(mean = mean(x),
    lower = quantile(x, 0.025),
    upper = quantile(x, 0.975))
})
stats_df <- as.data.frame(t(stats))
stats_df$day <- 0:n_days

# Real BTC trajectory (test)
btc_real_vec <- as.numeric(btc_test)
btc_real_df <- data.frame(day = 0:(length(btc_real_vec) - 1), price = btc_real_vec)

# Plot
ggplot(stats_df, aes(x = day)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.4) +
  geom_line(aes(y = mean), color = "blue", size = 1) +
  geom_line(data = btc_real_df, aes(x = day, y = price), color = "red", size = 1.2) +
  labs(title = "Simulación de precios BTC con Proceso Ornstein-Uhlenbeck",
       subtitle = "Rojo: trayectoria real — Azul: promedio simulado — Banda: IC 95%",
       x = "Día desde simulación", y = "Precio BTC (USD)") +
  theme_minimal()
