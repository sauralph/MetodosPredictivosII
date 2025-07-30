# Load libraries
library(quantmod)
library(ggplot2)
library(dplyr)
library(tidyr)

# 1. Get BTC price data
getSymbols("BTC-USD", src = "yahoo", from = Sys.Date() - 365, auto.assign = TRUE)
btc <- na.omit(Cl(`BTC-USD`))

# 2. Compute log-returns and split
log_ret <- dailyReturn(btc, type = "log")
ret_train <- head(log_ret, 180)
btc_test <- tail(btc, 180)
btc_real_vec <- as.numeric(btc_test)
S0 <- as.numeric(btc[180])

# 3. Estimate parameters
mu <- mean(ret_train)
sigma <- sd(ret_train)
dt <- 1  # 1 day

n_steps <- 180

# 4. Compute binomial parameters
u <- exp(sigma * sqrt(dt))
d <- exp(-sigma * sqrt(dt))
p <- (exp(mu * dt) - d) / (u - d)

# Ensure p is between 0 and 1
p <- max(min(p, 1), 0)

# 5. Generate tree paths
# We'll build all terminal nodes using combinatorics
library(data.table)
tree_df <- data.frame()

for (i in 0:n_steps) {
  up_moves <- i
  down_moves <- n_steps - i
  prob <- choose(n_steps, up_moves) * (p^up_moves) * ((1 - p)^down_moves)
  price <- S0 * u^up_moves * d^down_moves
  
  tree_df <- rbind(tree_df, data.frame(
    up_moves = up_moves,
    down_moves = down_moves,
    final_price = price,
    prob = prob
  ))
}

# 6. Simulate sample paths from root (optional visualization)
n_paths <- 1000
set.seed(42)
simulate_binomial_paths <- function(S0, u, d, p, n_steps, n_paths) {
  steps <- matrix(rbinom(n_paths * n_steps, 1, p), ncol = n_steps)
  log_path <- apply(steps, 1, function(x) {
    cumsum(log(ifelse(x == 1, u, d)))
  })
  prices <- S0 * exp(rbind(rep(0, n_paths), log_path))
  prices
}

tree_paths <- simulate_binomial_paths(S0, u, d, p, n_steps, 1000)

# 7. Compute mean and CI
stats_df <- apply(tree_paths, 1, function(x) {
  c(mean = mean(x), lower = quantile(x, 0.025), upper = quantile(x, 0.975))
})
stats_df <- as.data.frame(t(stats_df))
stats_df$day <- 0:n_steps
colnames(stats_df) <- c("mean", "lower", "upper", "day")

# 8. Actual BTC path
btc_real_df <- data.frame(day = 0:n_steps, price = c(S0, btc_real_vec))

# 9. Plot
ggplot(stats_df, aes(x = day)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.3) +
  geom_line(aes(y = mean), color = "blue", size = 1) +
  geom_line(data = btc_real_df, aes(x = day, y = price), color = "red", size = 1.2) +
  labs(title = "BTC Price Simulation Using Binomial Tree",
       subtitle = paste("Blue: simulated mean — Red: BTC real — CI 95%"),
       x = "Day", y = "Price (USD)") +
  theme_minimal()


# Extract two sample paths
sample_paths_df <- data.frame(
  day = 0:n_steps,
  path1 = tree_paths[, 1],
  path2 = tree_paths[, 2]
)

# Reshape to long format for ggplot
sample_paths_long <- pivot_longer(
  sample_paths_df,
  cols = c("path1", "path2"),
  names_to = "path",
  values_to = "price"
)

ggplot(stats_df, aes(x = day)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.3) +
  geom_line(aes(y = mean), color = "blue", size = 1) +
  geom_line(data = btc_real_df, aes(x = day, y = price), color = "red", size = 1.2) +
  geom_line(data = sample_paths_long, aes(x = day, y = price, group = path), 
            color = "gray40", size = 0.8) +
  labs(title = "BTC Price Simulation Using Binomial Tree",
       subtitle = "Blue: mean path — Red: real BTC — Gray dashed: sample paths — CI 95%",
       x = "Day", y = "Price (USD)") +
  theme_minimal()
