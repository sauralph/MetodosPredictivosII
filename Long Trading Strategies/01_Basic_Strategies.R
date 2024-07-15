library(quantmod)

# Get Bitcoin historical data from Yahoo Finance
getSymbols("BTC-USD", src = "yahoo", from = "2020-01-01", to = Sys.Date())

# Store the data in a variable
bitcoin_data <- `BTC-USD`

# Display the first few rows of the data
head(bitcoin_data)

# Plot the data
plot(bitcoin_data)
prices <- bitcoin_data$`BTC-USD.Close`


# HODL strategy -----------------------------------------------------------

(as.numeric(prices[length(prices)]) - as.numeric(prices[1])) / as.numeric(prices[1])


# SMA Strategy ------------------------------------------------------------

# Calculate moving averages
short_ma <- SMA(prices, n = 2)
long_ma <- SMA(prices, n = 30)

# Determine long entry points
long_signals <- which(short_ma > long_ma & lag(short_ma, 1) <= lag(long_ma, 1))

# Plot the prices and signals
plot(prices, type = "l", main = "Moving Average Crossover Strategy", xlab = "Date", ylab = "Price")
lines(short_ma, col = "blue")
lines(long_ma, col = "red")
points(prices[long_signals], col = "green", pch = 19)  # Long signals

# Simulate the returns
returns <- numeric(length(long_signals) - 1)
for (i in 1:(length(long_signals) - 1)) {
  entry_price <- as.numeric(prices[long_signals[i]])
  exit_price <- as.numeric(prices[long_signals[i + 1]])
  returns[i] <- (exit_price - entry_price) / entry_price
}

# Calculate cumulative return
cumulative_return <- prod(1 + returns) - 1

# Display results
cat("Cumulative Return: ", cumulative_return * 100, "%\n")

# Plot cumulative returns
cumulative_returns <- cumprod(1 + returns) - 1
plot(index(prices)[long_signals[-length(long_signals)]], cumulative_returns, type = "l", col = "green", lwd = 2, main = "Cumulative Returns of Moving Average Crossover Strategy", xlab = "Date", ylab = "Cumulative Return")

# RSI Strategy ------------------------------------------------------------

# Calculate RSI
rsi <- RSI(prices, n = 14)

# Determine long entry points
long_signals <- which(rsi < 30 & lag(rsi, 1) >= 30)

# Plotting the prices and signals
plot(prices, type = "l", main = "RSI Strategy")
points(prices[long_signals], col = "green", pch = 19)  # Long signals

# Simulate the returns
returns <- numeric(length(long_signals) - 1)
for (i in 1:(length(long_signals) - 1)) {
  entry_price <- as.numeric(prices[long_signals[i]])
  exit_price <- as.numeric(prices[long_signals[i + 1]])
  returns[i] <- (exit_price - entry_price) / entry_price
}

# Calculate cumulative return
cumulative_return <- prod(1 + returns) - 1

# Display results
cat("Cumulative Return: ", cumulative_return * 100, "%\n")

# Plot cumulative returns
cumulative_returns <- cumprod(1 + returns) - 1
plot(index(prices)[long_signals[-length(long_signals)]], cumulative_returns, type = "l", col = "green", lwd = 2, main = "Cumulative Returns of RSI Strategy", xlab = "Date", ylab = "Cumulative Return")

# Bollinger Bands ---------------------------------------------------------

# Calculate Bollinger Bands
bbands <- BBands(prices, n = 20, sd = 2)
# Determine long entry points
long_signals <- which(prices < bbands[,"dn"] & lag(prices, 1) >= lag(bbands[,"dn"], 1))

# Plotting the prices and signals
plot(prices, type = "l", main = "Bollinger Bands Strategy", xlab = "Time", ylab = "Price")
lines(bbands[,"dn"], col = "red")
points(prices[long_signals], col = "green", pch = 19)  # Long signals

# Simulate the returns
returns <- numeric(length(long_signals) - 1)
for (i in 1:(length(long_signals) - 1)) {
  entry_price <- as.numeric(prices[long_signals[i]])
  exit_price <- as.numeric(prices[long_signals[i + 1]])
  returns[i] <- (exit_price - entry_price) / entry_price
}

# Handle the case where there's only one long signal
if (length(long_signals) > 1) {
  # Calculate cumulative return
  cumulative_return <- prod(1 + returns) - 1
  
  # Display results
  cat("Cumulative Return: ", cumulative_return * 100, "%\n")
  
  # Plot cumulative returns
  cumulative_returns <- cumprod(1 + returns) - 1
  plot(long_signals[-length(long_signals)], cumulative_returns, type = "l", col = "green", lwd = 2, main = "Cumulative Returns of Bollinger Bands Strategy", xlab = "Time", ylab = "Cumulative Return")
} else {
  cat("Not enough long signals to simulate returns.\n")
}


# CUMSUM Filter -----------------------------------------------------------

# Define the CUSUM filter function
cusum_filter <- function(prices, threshold) {
  prices_<-as.numeric(prices)
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


# Set the threshold for the CUSUM filter
threshold <- 1000

# Get long signals
long_signals <- cusum_filter(prices, threshold)

# Plotting the prices and signals
plot(prices, type = "l", main = "CUSUM Filter Strategy", xlab = "Date", ylab = "Price")
points(index(prices)[long_signals], prices[long_signals], col = "green", pch = 19)  # Long signals

# Simulate the returns
returns <- numeric(length(long_signals) - 1)
for (i in 1:(length(long_signals) - 1)) {
  entry_price <- as.numeric(prices[long_signals[i]])
  exit_price <- as.numeric(prices[long_signals[i + 1]])
  returns[i] <- (exit_price - entry_price) / entry_price
}

# Handle the case where there's only one long signal
if (length(long_signals) > 1) {
  # Calculate cumulative return
  cumulative_return <- prod(1 + returns) - 1
  
  # Display results
  cat("Cumulative Return: ", cumulative_return * 100, "%\n")
  
  # Plot cumulative returns
  cumulative_returns <- cumprod(1 + returns) - 1
  plot(index(prices)[long_signals[-length(long_signals)]], cumulative_returns, type = "l", col = "green", lwd = 2, main = "Cumulative Returns of CUSUM Filter Strategy", xlab = "Date", ylab = "Cumulative Return")
} else {
  cat("Not enough long signals to simulate returns.\n")
}

