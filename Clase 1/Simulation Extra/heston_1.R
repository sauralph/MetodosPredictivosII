# Packages
library(quantmod)
library(dplyr)
library(ggplot2)
library(tidyr)

# Download BTC data
getSymbols("BTC-USD", src = "yahoo", from = Sys.Date() - 365, auto.assign = TRUE)
btc <- Cl(`BTC-USD`)
btc <- na.omit(btc)

# Log returns
log_ret <- dailyReturn(btc, type = "log")
log_ret <- na.omit(log_ret)

# Training: first 180 days
ret_train <- head(log_ret, 180)
btc_test <- tail(btc, 180)

# Estimate constant drift from training
mu <- mean(ret_train)

# Estimate initial volatility
v0 <- var(ret_train)

# Heston parameters (assumed or rough estimates)
kappa <- 3         # speed of reversion
theta <- v0        # long-term variance
sigma_v <- 0.5     # volatility of volatility
rho <- -0.7        # correlation between returns and variance
S0 <- as.numeric(Cl(btc[index(ret_train)[180]]))

# Simulation settings
n_days <- 180
n_sims <- 10000
dt <- 1
set.seed(42)

# Simulate Heston paths
simulate_heston <- function(S0, v0, mu, kappa, theta, sigma_v, rho, n_days, n_sims, dt) {
  S <- matrix(NA, nrow = n_days + 1, ncol = n_sims)
  v <- matrix(NA, nrow = n_days + 1, ncol = n_sims)
  S[1, ] <- S0
  v[1, ] <- v0
  
  for (t in 2:(n_days + 1)) {
    Z1 <- rnorm(n_sims)
    Z2 <- rnorm(n_sims)
    W1 <- Z1
    W2 <- rho * Z1 + sqrt(1 - rho^2) * Z2
    
    # Variance: CIR process
    v[t, ] <- abs(v[t - 1, ] + kappa * (theta - v[t - 1, ]) * dt + sigma_v * sqrt(pmax(v[t - 1, ], 0)) * sqrt(dt) * W2)
    
    # Price process
    S[t, ] <- S[t - 1, ] * exp((mu - 0.5 * v[t - 1, ]) * dt + sqrt(v[t - 1, ]) * sqrt(dt) * W1)
  }
  
  return(S)
}

S_matrix <- simulate_heston(S0, v0, mu, kappa, theta, sigma_v, rho, n_days, n_sims, dt)

# Stats
S_stats <- apply(S_matrix, 1, function(x) {
  c(mean = mean(x),
    lower = quantile(x, 0.025),
    upper = quantile(x, 0.975))
})
S_df <- as.data.frame(t(S_stats))
S_df$day <- 0:n_days

# Real BTC trajectory
btc_real_vec <- as.numeric(btc_test)
btc_real_df <- data.frame(day = 0:(length(btc_real_vec) - 1), price = btc_real_vec)
colnames(S_df)<-c("mean","lower","upper","day")

# Plot
ggplot(S_df, aes(x = day)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.4) +
  geom_line(aes(y = mean), color = "blue", size = 1) +
  geom_line(data = btc_real_df, aes(x = day, y = price), color = "red", size = 1.2) +
  labs(title = "Simulación de precios BTC con modelo de Heston",
       subtitle = "Rojo: trayectoria real — Azul: media simulada — Banda: IC 95%",
       x = "Día desde simulación", y = "Precio BTC (USD)") +
  theme_minimal()
