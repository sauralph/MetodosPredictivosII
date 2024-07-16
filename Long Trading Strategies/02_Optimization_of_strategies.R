library(quantmod)
library(TTR)

# Get Bitcoin historical data from Yahoo Finance
getSymbols("BTC-USD", src = "yahoo", from = "2020-01-01", to = Sys.Date())

# Store the data in a variable
bitcoin_data <- `BTC-USD`
prices <- bitcoin_data$`BTC-USD.Close`


# Split -------------------------------------------------------------------
n<-length(prices)
last<-ceiling(n * .3)
idx<-seq(n-last,n)
train_prices<-prices[-idx]
test_prices<-prices[idx]

# HODL strategy -----------------------------------------------------------
hodl_return <- (as.numeric(test_prices[length(test_prices)]) - as.numeric(test_prices[1])) / as.numeric(test_prices[1])
cat("HODL Cumulative Return: ", hodl_return * 100, "%\n")

# SMA Strategy ------------------------------------------------------------
calculate_cumulative_return <- function(short_n, long_n, prices) {
  short_ma <- SMA(prices, n = short_n)
  long_ma <- SMA(prices, n = long_n)
  long_signals <- which(short_ma > long_ma & lag(short_ma, 1) <= lag(long_ma, 1))
  
  if (length(long_signals) < 2) return(NA)
  
  returns <- numeric(length(long_signals) - 1)
  for (i in 1:(length(long_signals) - 1)) {
    entry_price <- as.numeric(prices[long_signals[i]])
    exit_price <- as.numeric(prices[long_signals[i + 1]])
    returns[i] <- (exit_price - entry_price) / entry_price
  }
  
  cumulative_return <- prod(1 + returns) - 1
  return(cumulative_return)
}

short_n_values <- seq(2, 20, by = 2)
long_n_values <- seq(30, 100, by = 10)


cumulative_returns_matrix <- matrix(NA, nrow = length(short_n_values), ncol = length(long_n_values))
for (i in 1:length(short_n_values)) {
  for (j in 1:length(long_n_values)) {
    cumulative_returns_matrix[i, j] <- calculate_cumulative_return(short_n_values[i], long_n_values[j], train_prices)
  }
}

# Convert the matrix to a data frame for plotting
cumulative_returns_df <- melt(cumulative_returns_matrix)
colnames(cumulative_returns_df) <- c("Short_N", "Long_N", "Cumulative_Return")
cumulative_returns_df$Short_N <- short_n_values[cumulative_returns_df$Short_N]
cumulative_returns_df$Long_N <- long_n_values[cumulative_returns_df$Long_N]

contour(cumulative_returns_matrix)

# Plot the contour plot using ggplot2
ggplot(cumulative_returns_df, aes(x = Short_N, y = Long_N, z = Cumulative_Return)) +
  geom_contour_filled() +
  labs(title = "Contour Plot of Cumulative Returns",
       x = "Short-Term MA (n)",
       y = "Long-Term MA (n)",
       fill = "Cumulative Return") +
  theme_minimal()

calculate_cumulative_return(7, 30, train_prices)

# RSI Strategy ------------------------------------------------------------

# Function to calculate cumulative returns for given n values and RSI limit
calculate_cumulative_return_rsi <- function(n, rsi_limit, prices) {
  rsi <- RSI(prices, n = n)
  long_signals <- which(rsi < rsi_limit & lag(rsi, 1) >= rsi_limit)
  
  if (length(long_signals) < 2) return(NA)
  
  returns <- numeric(length(long_signals) - 1)
  for (i in 1:(length(long_signals) - 1)) {
    entry_price <- as.numeric(prices[long_signals[i]])
    exit_price <- as.numeric(prices[long_signals[i + 1]])
    returns[i] <- (exit_price - entry_price) / entry_price
  }
  
  cumulative_return <- prod(1 + returns) - 1
  return(cumulative_return)
}

# Define range of n values and RSI limits
n_values <- seq(10, 30, by = 2)
rsi_limits <- seq(20, 40, by = 2)

# Initialize matrix to store cumulative returns
cumulative_returns_matrix <- matrix(NA, nrow = length(n_values), ncol = length(rsi_limits))

# Calculate cumulative returns for each combination of n and RSI limit
for (i in 1:length(n_values)) {
  for (j in 1:length(rsi_limits)) {
    cumulative_returns_matrix[i, j] <- calculate_cumulative_return_rsi(n_values[i], rsi_limits[j], train_prices)
  }
}

# Convert the matrix to a data frame for plotting
cumulative_returns_df <- melt(cumulative_returns_matrix)
colnames(cumulative_returns_df) <- c("RSI_N", "RSI_Limit", "Cumulative_Return")
cumulative_returns_df$RSI_N <- n_values[cumulative_returns_df$RSI_N]
cumulative_returns_df$RSI_Limit <- rsi_limits[cumulative_returns_df$RSI_Limit]

# Plot the contour plot with height bands using ggplot2
ggplot(cumulative_returns_df, aes(x = RSI_N, y = RSI_Limit, z = Cumulative_Return)) +
  geom_contour_filled() +
  labs(title = "Contour Plot of Cumulative Returns for RSI Strategy",
       x = "RSI Period (n)",
       y = "RSI Limit",
       fill = "Cumulative Return") +
  theme_minimal()

calculate_cumulative_return_rsi(10, 20, test_prices)

# Bollinger Bands ---------------------------------------------------------

# Function to calculate cumulative returns for given n values and sd
calculate_cumulative_return_bbands <- function(n, sd, prices) {
  bbands <- BBands(prices, n = n, sd = sd)
  long_signals <- which(prices < bbands[, "dn"] & lag(prices, 1) >= lag(bbands[, "dn"], 1))
  
  if (length(long_signals) < 2) return(NA)
  
  returns <- numeric(length(long_signals) - 1)
  for (i in 1:(length(long_signals) - 1)) {
    entry_price <- as.numeric(prices[long_signals[i]])
    exit_price <- as.numeric(prices[long_signals[i + 1]])
    returns[i] <- (exit_price - entry_price) / entry_price
  }
  
  cumulative_return <- prod(1 + returns) - 1
  return(cumulative_return)
}

# Define range of n values and sd values
n_values <- seq(10, 30, by = 2)
sd_values <- seq(1, 3, by = 0.5)

# Initialize matrix to store cumulative returns
cumulative_returns_matrix <- matrix(NA, nrow = length(n_values), ncol = length(sd_values))

# Calculate cumulative returns for each combination of n and sd
for (i in 1:length(n_values)) {
  for (j in 1:length(sd_values)) {
    cumulative_returns_matrix[i, j] <- calculate_cumulative_return_bbands(n_values[i], sd_values[j], train_prices)
  }
}

# Convert the matrix to a data frame for plotting
cumulative_returns_df <- melt(cumulative_returns_matrix)
colnames(cumulative_returns_df) <- c("BBands_N", "BBands_SD", "Cumulative_Return")
cumulative_returns_df$BBands_N <- n_values[cumulative_returns_df$BBands_N]
cumulative_returns_df$BBands_SD <- sd_values[cumulative_returns_df$BBands_SD]

# Plot the contour plot with height bands using ggplot2
ggplot(cumulative_returns_df, aes(x = BBands_N, y = BBands_SD, z = Cumulative_Return)) +
  geom_contour_filled() +
  labs(title = "Contour Plot of Cumulative Returns for Bollinger Bands Strategy",
       x = "Bollinger Bands Period (n)",
       y = "Bollinger Bands SD",
       fill = "Cumulative Return") +
  theme_minimal()

calculate_cumulative_return_bbands(17, 2.5, test_prices)


# CUSUM -------------------------------------------------------------------

# Define the CUSUM filter function
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

# Function to calculate cumulative returns for a given threshold
calculate_cumulative_return_cusum <- function(threshold, prices) {
  long_signals <- cusum_filter(prices, threshold)
  
  if (length(long_signals) < 2) return(NA)
  
  returns <- numeric(length(long_signals) - 1)
  for (i in 1:(length(long_signals) - 1)) {
    entry_price <- as.numeric(prices[long_signals[i]])
    exit_price <- as.numeric(prices[long_signals[i + 1]])
    returns[i] <- (exit_price - entry_price) / entry_price
  }
  
  cumulative_return <- prod(1 + returns) - 1
  return(cumulative_return)
}

# Define range of threshold values
threshold_values <- seq(100, 2000, by = 100)

# Initialize vector to store cumulative returns
cumulative_returns <- numeric(length(threshold_values))

# Calculate cumulative returns for each threshold value
for (i in 1:length(threshold_values)) {
  cumulative_returns[i] <- calculate_cumulative_return_cusum(threshold_values[i], train_prices)
}

# Create a data frame for plotting
cumulative_returns_df <- data.frame(Threshold = threshold_values, Cumulative_Return = cumulative_returns)

# Plot the contour plot with height bands using ggplot2
ggplot(cumulative_returns_df, aes(x = Threshold, y = Cumulative_Return)) +
  geom_line() +
  geom_point() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Cumulative Returns for CUSUM Filter Strategy",
       x = "CUSUM Threshold",
       y = "Cumulative Return") +
  theme_minimal()

calculate_cumulative_return_cusum(500,test_prices)
