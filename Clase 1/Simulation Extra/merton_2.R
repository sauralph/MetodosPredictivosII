simulate_merton_mean_path <- function(S0, mu, sigma, lambda, mu_j, sigma_j, n_days, n_sims, dt = 1) {
  prices <- matrix(NA, nrow = n_days + 1, ncol = n_sims)
  prices[1, ] <- S0
  
  for (t in 2:(n_days + 1)) {
    Z <- rnorm(n_sims)
    dW <- sqrt(dt) * Z
    dN <- rpois(n_sims, lambda * dt)
    J <- rnorm(n_sims, mean = mu_j, sd = sigma_j)
    jump <- dN * J  # log-space jumps
    
    drift <- (mu - 0.5 * sigma^2) * dt
    diffusion <- sigma * dW
    dlogS <- drift + diffusion + jump
    
    prices[t, ] <- prices[t - 1, ] * exp(dlogS)
  }
  
  mean_path <- rowMeans(prices)
  return(mean_path)
}

merton_mse <- function(mu, sigma, lambda, mu_j, sigma_j,
                       S0, n_days, n_sims, btc_real_prices, dt = 1) {
  
  mean_path <- simulate_merton_mean_path(
    S0 = S0,
    mu = mu,
    sigma = sigma,
    lambda = lambda,
    mu_j = mu_j,
    sigma_j = sigma_j,
    n_days = n_days,
    n_sims = n_sims,
    dt = dt
  )
  
  if (length(btc_real_prices) != length(mean_path)) {
    stop("Length mismatch between real prices and simulated path.")
  }
  
  mse <- max(abs(btc_real_prices - mean_path))
  return(mse)
}

merton_fitness <- function(params, S0, n_days, n_sims, btc_real_prices) {
  mu       <- params[1]
  sigma    <- params[2]
  lambda   <- params[3]
  mu_j     <- params[4]
  sigma_j  <- params[5]
  
  # Penalize invalid combinations
  if (sigma <= 0 || sigma_j <= 0 || lambda < 0) {
    return(Inf)
  }
  
  mse <- tryCatch(
    merton_mse(mu, sigma, lambda, mu_j, sigma_j, S0, n_days, n_sims, btc_real_prices),
    error = function(e) Inf
  )
  
  return(-mse)
}

library(GA)

# BTC preparation
btc_real_prices <- as.numeric(tail(btc, 181))  # includes initial point
S0 <- btc_real_prices[1]
n_days <- length(btc_real_prices) - 1
n_sims <- 1000  # for performance

set.seed(42)
ga_result_merton <- ga(
  type = "real-valued",
  fitness = function(params) merton_fitness(params, S0, n_days, n_sims, btc_real_prices),
  lower = c(0.0001, 0.001,  0.01, -0.1,  0.001),  # mu, sigma, lambda, mu_j, sigma_j
  upper = c(0.01,   0.1,    0.5,   0.1,  0.1),
  popSize = 100,
  maxiter = 100,
  run = 20,
  seed = 42,
  parallel = FALSE
)

summary(ga_result_merton)

best_params <- ga_result_merton@solution
names(best_params) <- c("mu", "sigma", "lambda", "mu_j", "sigma_j")
print(best_params)

# Final simulation with best parameters
best_mean_path <- simulate_merton_mean_path(
  S0 = S0,
  mu = best_params["mu"],
  sigma = best_params["sigma"],
  lambda = best_params["lambda"],
  mu_j = best_params["mu_j"],
  sigma_j = best_params["sigma_j"],
  n_days = n_days,
  n_sims = 10000
)

plot(0:n_days, btc_real_prices, type = "l", col = "red", lwd = 2,
     ylab = "Price", xlab = "Day", main = "BTC Real vs Optimized Merton Path")
lines(0:n_days, best_mean_path, col = "blue", lwd = 2)
legend("topright", legend = c("Real", "Merton Simulated"), col = c("red", "blue"), lwd = 2)
