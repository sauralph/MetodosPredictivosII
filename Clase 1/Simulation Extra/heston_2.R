simulate_heston_mean_path <- function(S0, v0, mu, kappa, theta, sigma_v, rho, n_days, n_sims, dt = 1) {
  # Initialize matrices
  S <- matrix(NA, nrow = n_days + 1, ncol = n_sims)
  v <- matrix(NA, nrow = n_days + 1, ncol = n_sims)
  S[1, ] <- S0
  v[1, ] <- v0
  
  # Simulate Heston paths
  for (t in 2:(n_days + 1)) {
    Z1 <- rnorm(n_sims)
    Z2 <- rnorm(n_sims)
    W1 <- Z1
    W2 <- rho * Z1 + sqrt(1 - rho^2) * Z2
    
    v[t, ] <- abs(v[t - 1, ] + kappa * (theta - v[t - 1, ]) * dt +
                    sigma_v * sqrt(pmax(v[t - 1, ], 0)) * sqrt(dt) * W2)
    
    S[t, ] <- S[t - 1, ] * exp((mu - 0.5 * v[t - 1, ]) * dt +
                                 sqrt(v[t - 1, ]) * sqrt(dt) * W1)
  }
  
  # Compute mean path
  mean_path <- rowMeans(S)
  return(mean_path)
}

mean_path <- simulate_heston_mean_path(
  S0 = btc[1],
  v0 = 0.0004,
  mu = 0.0003,
  kappa = 3,
  theta = 0.0004,
  sigma_v = 0.5,
  rho = -0.7,
  n_days = 180,
  n_sims = 10000
)

heston_mse <- function(mu, v0, kappa, theta, sigma_v, rho,
                       S0, n_days, n_sims, btc_real_prices, dt = 1) {
  
  # Simulate Heston paths and compute mean path
  mean_path <- simulate_heston_mean_path(
    S0 = S0,
    v0 = v0,
    mu = mu,
    kappa = kappa,
    theta = theta,
    sigma_v = sigma_v,
    rho = rho,
    n_days = n_days,
    n_sims = n_sims,
    dt = dt
  )
  
  # Align lengths
  if (length(btc_real_prices) != length(mean_path)) {
    stop("Length mismatch between real prices and mean simulated path.")
  }
  
  # Compute Mean Squared Error (MSE)
  mse <- mean((btc_real_prices - mean_path)^2)
  return(mse)
}

mse <- heston_mse(
  mu = 0.0003,
  v0 = 0.0004,
  kappa = 3,
  theta = 0.0004,
  sigma_v = 0.5,
  rho = -0.7,
  S0 = btc[1],
  n_days = length(btc) - 1,
  n_sims = 10000,
  btc_real_prices = btc
)

cat("MSE between Heston mean path and real BTC:", round(mse, 2), "\n")
#install.packages("GA")  # Only if not already installed
library(GA)
# Fitness function for GA (to be minimized, so we return -MSE)
heston_fitness <- function(params, S0, n_days, n_sims, btc_real_prices) {
  mu       <- params[1]
  v0       <- params[2]
  kappa    <- params[3]
  theta    <- params[4]
  sigma_v  <- params[5]
  rho      <- params[6]
  
  # Penalize invalid parameter combinations (e.g. negative variance)
  if (v0 <= 0 || theta <= 0 || sigma_v <= 0 || kappa <= 0 || abs(rho) > 1) {
    return(Inf)
  }
  
  mse <- tryCatch(
    heston_mse(mu, v0, kappa, theta, sigma_v, rho, S0, n_days, n_sims, btc_real_prices),
    error = function(e) Inf
  )
  
  return(-mse)  # maximize negative MSE = minimize MSE
}

# BTC test data (real prices for comparison)
# Should already be defined: btc_real_prices

# Set simulation configuration
S0 <- btc[1]
n_days <- length(btc) - 1
n_sims <- 1000  # can reduce for speed during optimization
btc_real_prices<-btc
heston_fitness(c(  mu = 0.0003,
                   v0 = 0.0004,
                   kappa = 3,
                   theta = 0.0004,
                   sigma_v = 0.5,
                   rho = -0.7),S0,n_days,n_sims,btc_real_prices)
# Run GA
ga_result <- ga(
  type = "real-valued",
  fitness = function(params) heston_fitness(params, S0, n_days, n_sims, btc_real_prices),
  lower = c( 0.0001, 0.00001, 0.1, 0.00001, 0.01, -0.99),   # lower bounds for mu, v0, kappa, theta, sigma_v, rho
  upper = c( 0.01,    0.01,    10,  0.01,    2.0,   0.99),   # upper bounds
  suggestions = ga_result@solution,
  popSize = 500,
  maxiter = 100,
  run = 20,
  seed = 42,
  parallel = FALSE  # can use TRUE if multicore
)

summary(ga_result)


# Second attempt ----------------------------------------------------------
mu <- ga_result@solution[1]

# Estimate initial volatility
v0 <- ga_result@solution[2]

# Heston parameters (assumed or rough estimates)
kappa <- ga_result@solution[3]         # speed of reversion
theta <- ga_result@solution[4]        # long-term variance
sigma_v <- ga_result@solution[5]     # volatility of volatility
rho <- ga_result@solution[6]        # correlation between returns and variance
S0 <- btc_real_df$price[1]

# Simulation settings
n_days <- 180
n_sims <- 1000
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
  #geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.4) +
  geom_line(aes(y = mean), color = "blue", size = 1) +
  geom_line(data = btc_real_df, aes(x = day, y = price), color = "red", size = 1.2) +
  labs(title = "Simulación de precios BTC con modelo de Heston",
       subtitle = "Rojo: trayectoria real — Azul: media simulada — Banda: IC 95%",
       x = "Día desde simulación", y = "Precio BTC (USD)") +
  theme_minimal()
