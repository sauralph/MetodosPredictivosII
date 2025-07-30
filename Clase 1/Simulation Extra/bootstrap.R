# Load libraries
library(quantmod)
library(dplyr)
library(ggplot2)
library(tidyr)

# 1. Get BTC data
getSymbols("BTC-USD", src = "yahoo", from = Sys.Date() - 365, auto.assign = TRUE)
btc <- na.omit(Cl(`BTC-USD`))

# 2. Compute log-returns
log_ret <- dailyReturn(btc, type = "log")
log_ret <- na.omit(log_ret)

# 3. Split into training and test
ret_train <- head(log_ret, 180)
btc_test <- tail(btc, 180)
btc_real_vec <- as.numeric(btc_test)
S0 <- as.numeric(Cl(btc)[180])

n_days <- 180
n_sims <- 10000
set.seed(42)

# 4. Bootstrap simulation of log-returns
bootstrap_prices <- matrix(NA, nrow = n_days + 1, ncol = n_sims)
bootstrap_prices[1, ] <- S0

for (j in 1:n_sims) {
  sampled_returns <- sample(ret_train, size = n_days, replace = TRUE)
  log_prices <- cumsum(sampled_returns)
  bootstrap_prices[-1, j] <- S0 * exp(log_prices)
}

# 5. Stats (mean path, 95% CI)
stats <- apply(bootstrap_prices, 1, function(x) {
  c(mean = mean(x), lower = quantile(x, 0.025), upper = quantile(x, 0.975))
})
stats_df <- as.data.frame(t(stats))
stats_df$day <- 0:n_days

# 6. Real BTC path
btc_real_df <- data.frame(day = 0:n_days, price = c(S0, btc_real_vec))
colnames(stats_df)<-c("mean","lower","upper","day")

# 7. Plot
ggplot(stats_df, aes(x = day)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.4) +
  geom_line(aes(y = mean), color = "blue", size = 1) +
  geom_line(data = btc_real_df, aes(x = day, y = price), color = "red", size = 1.2) +
  labs(title = "BTC Price Simulation Using Bootstrap",
       subtitle = "Red: actual BTC — Blue: mean path — Blue band: 95% CI",
       x = "Day", y = "Price (USD)") +
  theme_minimal()

# Extract two sample paths
sample_paths_df <- data.frame(
  day = 0:n_steps,
  path1 = bootstrap_prices[, 5],
  path2 = bootstrap_prices[, 4]
)

# Reshape to long format for ggplot
sample_paths_long <- pivot_longer(
  sample_paths_df,
  cols = c("path1", "path2"),
  names_to = "path",
  values_to = "price"
)

ggplot(stats_df, aes(x = day)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.4) +
  geom_line(aes(y = mean), color = "blue", size = 1) +
  geom_line(data = btc_real_df, aes(x = day, y = price), color = "red", size = 1.2) +
  geom_line(data = sample_paths_long, aes(x = day, y = price, group = path), 
            color = "gray40", size = 0.8) +
  labs(title = "BTC Price Simulation Using Bootstrap",
       subtitle = "Red: actual BTC — Blue: mean path — Blue band: 95% CI",
       x = "Day", y = "Price (USD)") +
  theme_minimal()

