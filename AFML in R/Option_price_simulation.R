# Load necessary packages
#install.packages("tidyverse")
library(tidyverse)

# Set parameters
S0 <- 100      # Initial stock price
K <- 105       # Strike price
T <- 1         # Time to maturity in years
r <- 0.05      # Risk-free interest rate
sigma <- 0.2   # Volatility of the stock
M <- 100       # Number of time steps
N <- 10000     # Number of simulation paths

# Simulate stock price paths
set.seed(123)
dt <- T / M
S <- matrix(0, nrow=N, ncol=M+1)
S[,1] <- S0

for (i in 2:(M+1)) {
  Z <- rnorm(N)
  S[,i] <- S[,i-1] * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
}

# Calculate 95% confidence intervals
S_mean <- apply(S, 2, mean)
#S_sd <- apply(S, 2, sd)
S_ci_upper <- apply(S, 2, quantile, probs = .975)
S_ci_lower <- apply(S, 2, quantile, probs = .025)

# Monte Carlo option pricing
payoffs <- pmax(S[,M+1] - K, 0)
discounted_payoffs <- exp(-r * T) * payoffs
mc_option_price <- mean(discounted_payoffs)

# Black-Scholes option pricing
d1 <- (log(S0 / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
d2 <- d1 - sigma * sqrt(T)
bs_option_price <- S0 * pnorm(d1) - K * exp(-r * T) * pnorm(d2)

# Print option prices
print(paste("Monte Carlo Option Price:", round(mc_option_price, 2)))
print(paste("Black-Scholes Option Price:", round(bs_option_price, 2)))

# Plot all paths
plot(S[1,], type="n", xlim=c(0, M), ylim=range(S), frame.plot=F, xlab="Time Steps", ylab="Stock Price", main="Monte Carlo Simulation of Stock Prices")
#apply(S, 1, function(x) {
#  lines(0:M, x, col=rgb(0.1, 0.1, 0.1, 0.1))
#})

# Plot confidence intervals
lines(0:M, S_ci_upper, col="red", lwd=2)
lines(0:M, S_ci_lower, col="red", lwd=2)

# Shade area between highest and lowest path
S_max <- apply(S, 2, max)
S_min <- apply(S, 2, min)
polygon(c(0:M, rev(0:M)), c(S_max, rev(S_min)), col=rgb(0.5, 0.5, 0.5, 0.5), border=NA)

# Add legend
legend("topleft", legend=c(
  "95% CI", "Min-Max Range"), 
  col=c("red",
        rgb(0.5, 0.5, 0.5, 0.5)), 
  lwd=2, fill=c(NA, rgb(0.5, 0.5, 0.5, 0.5)), 
  bty="n", lty=c(1, 1))

