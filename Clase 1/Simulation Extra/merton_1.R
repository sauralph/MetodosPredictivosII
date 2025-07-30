# Load packages
library(quantmod)
library(dplyr)
library(ggplot2)
library(tidyr)

# 1. Get BTC prices
getSymbols("BTC-USD", src = "yahoo", from = Sys.Date() - 365, auto.assign = TRUE)
btc <- na.omit(Cl(`BTC-USD`))

# 2. Compute log-returns
log_ret <- dailyReturn(btc, type = "log")
log_ret <- na.omit(log_ret)

# 3. Split data
ret_train <- head(log_ret, 180)
btc_test <- tail(btc, 180)
S0 <- as.numeric(Cl(btc)[180])
n_days <- 180
n_sims <- 10000
dt <- 1

# 4. Estimate GBM part
mu <- mean(ret_train)
sigma <- sd(ret_train)

# 5. Merton jump parameters (can be tuned)
lambda <- 0.1          # average 1 jump every 10 days
mu_j <- -0.02          # average jump size (in log-space)
sigma_j <- 0.05        # jump volatility

# 6. Simulate Merton paths
simulate_merton <- function(S0, mu, sigma, lambda, mu_j, sigma_j, n_days, n_sims, dt) {
  prices <- matrix(NA, nrow = n_days + 1, ncol = n_sims)
  prices[1, ] <- S0
  
  for (t in 2:(n_days + 1)) {
    Z <- rnorm(n_sims)
    dW <- sqrt(dt) * Z
    dN <- rpois(n_sims, lambda * dt)
    J <- rnorm(n_sims, mean = mu_j, sd = sigma_j)
    jump <- dN * J  # log(jump multiplier)
    
    drift <- (mu - 0.5 * sigma^2) * dt
    diffusion <- sigma * dW
    dlogS <- drift + diffusion + jump
    
    prices[t, ] <- prices[t - 1, ] * exp(dlogS)
  }
  
  return(prices)
}

set.seed(42)
merton_paths <- simulate_merton(S0, mu, sigma, lambda, mu_j, sigma_j, n_days, n_sims, dt)

# 7. Stats
stats <- apply(merton_paths, 1, function(x) {
  c(mean = mean(x), lower = quantile(x, 0.025), upper = quantile(x, 0.975))
})
stats_df <- as.data.frame(t(stats))
stats_df$day <- 0:n_days

# 8. Real BTC trajectory (test)
btc_real_vec <- as.numeric(btc_test)
btc_real_df <- data.frame(day = 0:(length(btc_real_vec) - 1), price = btc_real_vec)
colnames(stats_df)<-c("mean","lower","upper","day")

# 9. Plot
ggplot(stats_df, aes(x = day)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.4) +
  geom_line(aes(y = mean), color = "blue", size = 1) +
  geom_line(data = btc_real_df, aes(x = day, y = price), color = "red", size = 1.2) +
  labs(title = "BTC Price Simulation with Merton Jump-Diffusion",
       subtitle = "Red: actual BTC — Blue: mean path — Blue band: 95% CI",
       x = "Day", y = "Price (USD)") +
  theme_minimal()
