library(quantmod)
library(TTR)

# Get Bitcoin historical data from Yahoo Finance
getSymbols("BTC-USD", src = "yahoo", from = "2020-01-01", to = Sys.Date())

# Store the data in a variable
bitcoin_data <- `BTC-USD`
prices <- bitcoin_data$`BTC-USD.Close`

# HODL strategy -----------------------------------------------------------
hodl_return <- (as.numeric(prices[length(prices)]) - as.numeric(prices[1])) / as.numeric(prices[1])
cat("HODL Cumulative Return: ", hodl_return * 100, "%\n")

# SMA Strategy ------------------------------------------------------------
short_ma <- SMA(prices, n = 2)
long_ma <- SMA(prices, n = 30)
long_signals_sma <- which(short_ma > long_ma & lag(short_ma, 1) <= lag(long_ma, 1))

# Plot the SMA strategy
plot(prices, type = "l", main = "Moving Average Crossover Strategy", xlab = "Date", ylab = "Price")
lines(short_ma, col = "blue")
lines(long_ma, col = "red")
points(index(prices)[long_signals_sma], prices[long_signals_sma], col = "green", pch = 19)

# Simulate the SMA returns
returns_sma <- numeric(length(long_signals_sma) - 1)
for (i in 1:(length(long_signals_sma) - 1)) {
  entry_price <- as.numeric(prices[long_signals_sma[i]])
  exit_price <- as.numeric(prices[long_signals_sma[i + 1]])
  returns_sma[i] <- (exit_price - entry_price) / entry_price
}
cumulative_return_sma <- prod(1 + returns_sma) - 1
cat("SMA Cumulative Return: ", cumulative_return_sma * 100, "%\n")

# RSI Strategy ------------------------------------------------------------
rsi <- RSI(prices, n = 14)
long_signals_rsi <- which(rsi < 30 & lag(rsi, 1) >= 30)

# Plot the RSI strategy
plot(prices, type = "l", main = "RSI Strategy", xlab = "Date", ylab = "Price")
points(index(prices)[long_signals_rsi], prices[long_signals_rsi], col = "green", pch = 19)

# Simulate the RSI returns
returns_rsi <- numeric(length(long_signals_rsi) - 1)
for (i in 1:(length(long_signals_rsi) - 1)) {
  entry_price <- as.numeric(prices[long_signals_rsi[i]])
  exit_price <- as.numeric(prices[long_signals_rsi[i + 1]])
  returns_rsi[i] <- (exit_price - entry_price) / entry_price
}
cumulative_return_rsi <- prod(1 + returns_rsi) - 1
cat("RSI Cumulative Return: ", cumulative_return_rsi * 100, "%\n")

# Bollinger Bands Strategy ------------------------------------------------
bbands <- BBands(prices, n = 20, sd = 2)
long_signals_bbands <- which(prices < bbands[, "dn"] & lag(prices, 1) >= lag(bbands[, "dn"], 1))

# Plot the Bollinger Bands strategy
plot(prices, type = "l", main = "Bollinger Bands Strategy", xlab = "Date", ylab = "Price")
lines(bbands[, "dn"], col = "red")
points(index(prices)[long_signals_bbands], prices[long_signals_bbands], col = "green", pch = 19)

# Simulate the Bollinger Bands returns
returns_bbands <- numeric(length(long_signals_bbands) - 1)
for (i in 1:(length(long_signals_bbands) - 1)) {
  entry_price <- as.numeric(prices[long_signals_bbands[i]])
  exit_price <- as.numeric(prices[long_signals_bbands[i + 1]])
  returns_bbands[i] <- (exit_price - entry_price) / entry_price
}
cumulative_return_bbands <- prod(1 + returns_bbands) - 1
cat("Bollinger Bands Cumulative Return: ", cumulative_return_bbands * 100, "%\n")

# CUSUM Filter Strategy ---------------------------------------------------
cusum_filter <- function(prices, threshold) {
  prices_ <- as.numeric(prices)
  n <- length(prices_)
  S <- rep(0, n)
  long_signals <- rep(0, n)
  
  for (t in 2:n) {
    S[t] <- max(0, S[t-1] + prices_[t] - prices_[t-1])
    if (S[t] >= threshold) {
      long_signals[t] <- 1  # Generate a long signal
      S[t] <- 0  # Reset S[t]
    }
  }
  
  return(which(long_signals == 1))
}

threshold <- 1000
long_signals_cusum <- cusum_filter(prices, threshold)

# Plot the CUSUM Filter strategy
plot(prices, type = "l", main = "CUSUM Filter Strategy", xlab = "Date", ylab = "Price")
points(index(prices)[long_signals_cusum], prices[long_signals_cusum], col = "green", pch = 19)

# Simulate the CUSUM Filter returns
returns_cusum <- numeric(length(long_signals_cusum) - 1)
for (i in 1:(length(long_signals_cusum) - 1)) {
  entry_price <- as.numeric(prices[long_signals_cusum[i]])
  exit_price <- as.numeric(prices[long_signals_cusum[i + 1]])
  returns_cusum[i] <- (exit_price - entry_price) / entry_price
}
cumulative_return_cusum <- prod(1 + returns_cusum) - 1
cat("CUSUM Filter Cumulative Return: ", cumulative_return_cusum * 100, "%\n")

# Create a markdown table with the cumulative returns --------------------
cat("\n# Cumulative Returns for All Strategies\n")
cat("| Strategy | Cumulative Return (%) |\n")
cat("|----------|-----------------------|\n")
cat("| HODL |", hodl_return * 100, "|\n")
cat("| SMA |", cumulative_return_sma * 100, "|\n")
cat("| RSI |", cumulative_return_rsi * 100, "|\n")
cat("| Bollinger Bands |", cumulative_return_bbands * 100, "|\n")
cat("| CUSUM Filter |", cumulative_return_cusum * 100, "|\n")

