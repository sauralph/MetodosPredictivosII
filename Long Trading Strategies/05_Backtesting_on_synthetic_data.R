# Load necessary libraries
library(MASS)
library(quantmod)
# Get Bitcoin historical data from Yahoo Finance
getSymbols("BTC-USD", src = "yahoo", from = "2020-01-01", to = Sys.Date())
prices <- `BTC-USD`$`BTC-USD.Close`

# Interpolate NA values
prices <- na.approx(prices)


# Step 1: Estimate input parameters {sigma, phi}
estimate_parameters <- function(prices, E0) {
  I <- ncol(prices)
  T_max <- nrow(prices)
  
  X <- matrix(nrow = (T_max - 1) * I, ncol = 1)
  Y <- matrix(nrow = (T_max - 1) * I, ncol = 1)
  
  idx <- 1
  for (i in 1:I) {
    for (t in 1:(T_max - 1)) {
      X[idx] <- prices[t, i] - E0[i]
      Y[idx] <- prices[t + 1, i]
      idx <- idx + 1
    }
  }
  
  phi_hat <- as.vector(cov(Y, X) / cov(X, X))
  Z <- matrix(E0, nrow = (T_max - 1) * I, ncol = 1, byrow = TRUE)
  residuals <- Y - Z - phi_hat * X
  sigma_hat <- sqrt(var(residuals))
  
  return(list(sigma = sigma_hat, phi = phi_hat))
}

# Step 2: Construct mesh of stop-loss and profit-taking pairs
construct_mesh <- function(sigma) {
  pi <- seq(-12 * sigma, -sigma, length.out = 10)
  pi_bar <- seq(sigma, 12 * sigma, length.out = 10)
  mesh <- expand.grid(pi, pi_bar)
  colnames(mesh) <- c("pi", "pi_bar")
  return(mesh)
}

# Step 3: Generate paths
generate_paths <- function(N, T_max, sigma, phi, initial_price, E0) {
  paths <- matrix(nrow = N, ncol = T_max)
  for (i in 1:N) {
    paths[i, 1] <- initial_price
    for (t in 2:T_max) {
      paths[i, t] <- E0 + phi * (paths[i, t - 1] - E0) + rnorm(1, mean = 0, sd = sigma)
    }
  }
  return(paths)
}

# Step 4: Apply stop-loss and profit-taking logic
apply_trading_logic <- function(paths, mesh, T_max) {
  N <- nrow(paths)
  results <- data.frame(pi = numeric(), pi_bar = numeric(), sharpe_ratio = numeric())
  
  for (i in 1:nrow(mesh)) {
    pi <- mesh$pi[i]
    pi_bar <- mesh$pi_bar[i]
    final_pnl <- numeric(N)
    
    for (j in 1:N) {
      for (t in 1:T_max) {
        pnl <- paths[j, t] - paths[j, 1]
        if (pnl <= pi || pnl >= pi_bar) {
          final_pnl[j] <- pnl
          break
        }
        if (t == T_max) {
          final_pnl[j] <- pnl
        }
      }
    }
    
    sharpe_ratio <- mean(final_pnl) / sd(final_pnl)
    results <- rbind(results, data.frame(pi = pi, pi_bar = pi_bar, sharpe_ratio = sharpe_ratio))
  }
  
  return(results)
}

# Step 5: Determine optimal trading rule
determine_optimal_rule <- function(results) {
  optimal_rule <- results[which.max(results$sharpe_ratio), ]
  return(optimal_rule)
}

# Example usage
# Define input data
E0 <- mean(prices)

# Step 1: Estimate parameters
params <- estimate_parameters(prices, E0)
sigma_hat <- params$sigma
phi_hat <- params$phi

# Step 2: Construct mesh
mesh <- construct_mesh(sigma_hat)

# Step 3: Generate paths
N <- 100000
T_max <- 100
initial_price <- as.vector(prices[1, ])
paths <- generate_paths(N, T_max, sigma_hat, phi_hat, initial_price, E0)

# Step 4: Apply trading logic
results <- apply_trading_logic(paths, mesh, T_max)

# Step 5: Determine optimal rule
optimal_rule <- determine_optimal_rule(results)

print(optimal_rule)

i<-order(results$pi,results$pi_bar)

contour(results$pi[i],results$pi_bar[i],results$sharpe_ratio[i])

ggplot(results, aes(x = pi, y = pi_bar, z = sharpe_ratio)) +
  geom_contour_filled() +
  labs(
    title = "Contour Plot of Sharpe Ratio",
    x = expression(pi),
    y = expression(pi_bar),
    fill = "Sharpe Ratio"
  ) +
  theme_minimal()
