# Load libraries
library(quantmod)
library(GA)
library(ggplot2)
library(dplyr)

# 1. Download BTC data
getSymbols("BTC-USD", src = "yahoo", from = Sys.Date() - 365, auto.assign = TRUE)
btc <- na.omit(Cl(`BTC-USD`))
btc_train <- as.numeric(head(btc, 180))
S0 <- btc_train[1]
n_days <- length(btc_train)
n_agents <- 500

# 2. Agent-Based Simulation Function
simulate_abm <- function(alpha, beta, noise_sd, liquidity, fundamental_value,
                         n_f, n_c, n_n, S0, n_days) {
  price <- numeric(n_days)
  price[1] <- S0
  
  for (t in 2:n_days) {
    delta <- ifelse(t == 2, 0, price[t - 1] - price[t - 2])
    excess_f <- alpha * (fundamental_value - price[t - 1]) * n_f
    excess_c <- beta * delta * n_c
    excess_n <- rnorm(1, mean = 0, sd = noise_sd) * n_n
    excess_total <- excess_f + excess_c + excess_n
    price[t] <- price[t - 1] + liquidity * (excess_total / (n_f + n_c + n_n))
  }
  
  return(price)
}

# 3. MSE Loss Function
abm_mse <- function(params, S0, btc_train, n_days, n_agents) {
  alpha <- params[1]
  beta <- params[2]
  noise_sd <- params[3]
  liquidity <- params[4]
  fundamental_value <- params[5]
  f_ratio <- params[6]
  c_ratio <- params[7]
  n_ratio <- params[8]
  
  if (any(c(alpha, beta, noise_sd, liquidity) < 0) ||
      any(c(f_ratio, c_ratio, n_ratio) < 0)) return(Inf)
  
  total_ratio <- f_ratio + c_ratio + n_ratio
  n_f <- round(n_agents * f_ratio / total_ratio)
  n_c <- round(n_agents * c_ratio / total_ratio)
  n_n <- n_agents - n_f - n_c  # ensure sum = n_agents
  
  sim_prices <- simulate_abm(
    alpha, beta, noise_sd, liquidity, fundamental_value,
    n_f, n_c, n_n, S0, n_days
  )
  
  mse <- mean((btc_train - sim_prices)^2)
  return(mse)
}

# 4. Fitness Function for GA
abm_fitness <- function(params) {
  -abm_mse(params, S0, btc_train, n_days, n_agents)
}

# 5. Run GA Optimization
set.seed(42)
ga_result <- ga(
  type = "real-valued",
  fitness = abm_fitness,
  lower = c(0.0001, 0.0001, 0.0001, 1, 10000,   1, 1, 1),     # alpha, beta, noise_sd, liquidity, fundamental_value, f_ratio, c_ratio, n_ratio
  upper = c(1,      1,      1,      500, 60000,  100, 100, 100),
  popSize = 100,
  maxiter = 100,
  run = 20,
  seed = 42,
  parallel = FALSE
)

# 6. Simulate using best params
best_params <- as.numeric(ga_result@solution)
names(best_params) <- c("alpha", "beta", "noise_sd", "liquidity", "fundamental_value", "f_ratio", "c_ratio", "n_ratio")
print(best_params)

# Normalize agent types
total_ratio <- best_params["f_ratio"] + best_params["c_ratio"] + best_params["n_ratio"]
n_f <- round(n_agents * best_params["f_ratio"] / total_ratio)
n_c <- round(n_agents * best_params["c_ratio"] / total_ratio)
n_n <- n_agents - n_f - n_c

# Simulate with best parameters
simulated_price <- simulate_abm(
  alpha = best_params["alpha"],
  beta = best_params["beta"],
  noise_sd = best_params["noise_sd"],
  liquidity = best_params["liquidity"],
  fundamental_value = best_params["fundamental_value"],
  n_f = n_f, n_c = n_c, n_n = n_n,
  S0 = S0, n_days = n_days
)

# 7. Plot real vs simulated
df_plot <- data.frame(
  day = 1:n_days,
  real = btc_train,
  simulated = simulated_price
)

ggplot(df_plot, aes(x = day)) +
  geom_line(aes(y = real), color = "red", size = 1.2) +
  geom_line(aes(y = simulated), color = "blue", size = 1) +
  labs(title = "BTC Price Simulation Using Optimized ABM",
       subtitle = "Red: real BTC — Blue: ABM simulated",
       x = "Day", y = "Price (USD)") +
  theme_minimal()

