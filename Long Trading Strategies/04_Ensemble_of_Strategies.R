# Load necessary libraries
library(quantmod)
library(TTR)
library(randomForest)
library(reshape2)
library(ggplot2)
library(zoo)

# Get Bitcoin historical data from Yahoo Finance
getSymbols("BTC-USD", src = "yahoo", from = "2020-01-01", to = Sys.Date())
prices <- `BTC-USD`$`BTC-USD.Close`

# Interpolate NA values
prices <- na.approx(prices)

# Split data into training and testing sets
n <- length(prices)
last <- ceiling(n * 0.3)
idx <- seq(n - last + 1, n)
train_prices <- prices[-idx]
test_prices <- prices[idx]

# HODL strategy
hodl_return <- (as.numeric(test_prices[length(test_prices)]) - as.numeric(test_prices[1])) / as.numeric(test_prices[1])
cat("HODL Cumulative Return: ", hodl_return * 100, "%\n")

# Calculate SMA signals
calculate_sma_signals <- function(short_n, long_n, prices) {
  short_ma <- SMA(prices, n = short_n)
  long_ma <- SMA(prices, n = long_n)
  signals <- ifelse(short_ma > long_ma, 1, 0)
  signals[is.na(signals)] <- 0
  return(signals)
}

short_n <- 7
long_n <- 30
sma_signals_train <- calculate_sma_signals(short_n, long_n, train_prices)
sma_signals_test <- calculate_sma_signals(short_n, long_n, test_prices)

# Calculate RSI signals
calculate_rsi_signals <- function(n, rsi_limit, prices) {
  rsi <- RSI(prices, n = n)
  signals <- ifelse(rsi < rsi_limit, 1, 0)
  signals[is.na(signals)] <- 0
  return(signals)
}

rsi_n <- 14
rsi_limit <- 30
rsi_signals_train <- calculate_rsi_signals(rsi_n, rsi_limit, train_prices)
rsi_signals_test <- calculate_rsi_signals(rsi_n, rsi_limit, test_prices)

# Calculate Bollinger Bands signals
calculate_bbands_signals <- function(n, sd, prices) {
  bbands <- BBands(prices, n = n, sd = sd)
  signals <- ifelse(prices < bbands[, "dn"], 1, 0)
  signals[is.na(signals)] <- 0
  return(signals)
}

bbands_n <- 20
bbands_sd <- 2
bbands_signals_train <- calculate_bbands_signals(bbands_n, bbands_sd, train_prices)
bbands_signals_test <- calculate_bbands_signals(bbands_n, bbands_sd, test_prices)

# Calculate CUSUM signals
cusum_filter <- function(prices, threshold) {
  prices_ <- as.numeric(prices)
  n <- length(prices_)
  S <- rep(0, n)
  signals <- rep(0, n)
  
  for (t in 2:n) {
    S[t] <- max(0, S[t-1] + prices_[t] - prices_[t-1])
    if (S[t] >= threshold) {
      signals[t] <- 1  # Generate a long signal
      S[t] <- 0  # Reset S[t]
    }
  }
  
  return(signals)
}

cusum_threshold <- 500
cusum_signals_train <- cusum_filter(train_prices, cusum_threshold)
cusum_signals_test <- cusum_filter(test_prices, cusum_threshold)

# Create feature matrix and target vector for training
train_features <- data.frame(SMA = sma_signals_train, RSI = rsi_signals_train, BBands = bbands_signals_train, CUSUM = cusum_signals_train)
train_target <- ifelse(diff(train_prices) > 0, 1, 0)
train_features <- train_features[-1, ]  # Remove first row to match target length
train_target <- train_target[-1]  # Remove first element

# Train Random Forest model
set.seed(123)
rf_model <- randomForest(x = train_features, y = as.factor(train_target), ntree = 100)

# Create feature matrix for testing
test_features <- data.frame(SMA = sma_signals_test, RSI = rsi_signals_test, BBands = bbands_signals_test, CUSUM = cusum_signals_test)

# Make predictions
predictions <- predict(rf_model, test_features)

# Simulate returns based on predictions
test_prices <- as.numeric(test_prices)
returns <- numeric(length(predictions) - 1)
for (i in 1:(length(predictions) - 1)) {
  if (predictions[i] == 1) {
    entry_price <- test_prices[i]
    exit_price <- test_prices[i + 1]
    returns[i] <- (exit_price - entry_price) / entry_price
  }
}

# Calculate cumulative return
cumulative_return <- prod(1 + returns, na.rm = TRUE) - 1
cat("Cumulative Return over the test set: ", cumulative_return * 100, "%\n")

# Plot signals
plot(test_prices, type = "l", main = "Random Forest Buy/Sell Signals", xlab = "Date", ylab = "Price")
buy_signals <- which(predictions == 1)
sell_signals <- which(predictions == 0)
points(buy_signals, test_prices[buy_signals], col = "green", pch = 19)
points(sell_signals, test_prices[sell_signals], col = "red", pch = 19)

# Plot cumulative returns
cumulative_returns <- cumprod(1 + returns) - 1
plot(seq_along(cumulative_returns), cumulative_returns, type = "l", col = "blue", lwd = 2, 
     main = "Cumulative Returns of Random Forest Strategy", xlab = "Time", ylab = "Cumulative Return")

