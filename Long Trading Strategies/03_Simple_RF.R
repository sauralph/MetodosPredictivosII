# Load necessary libraries
library(quantmod)
library(randomForest)
library(caret)

# Get historical data
getSymbols("BTC-USD", src = "yahoo", from = "2020-01-01", to = Sys.Date())
prices <- `BTC-USD`$`BTC-USD.Close`

prices <- na.approx(prices)


# Feature engineering
lagged_prices <- Lag(prices, k = 1:5)  # Lagged values
rolling_mean <- rollapply(prices, width = 5, FUN = mean, align = "right", fill = NA)
rolling_mean_30 <- rollapply(prices, width = 5, FUN = mean, align = "right", fill = NA)
rolling_sd <- rollapply(prices, width = 5, FUN = sd, align = "right", fill = NA)
rolling_sd_30 <- rollapply(prices, width = 5, FUN = sd, align = "right", fill = NA)

# Create dataset
features <- data.frame(lagged_prices, rolling_mean, rolling_sd,rolling_sd_30,rolling_mean_30)
features <- na.omit(features)
target <- ifelse(diff(prices, lag = 1) > 0, 1, 0)  # Binary target: 1 for up, 0 for down
target <- target[6:length(target)]  # Align with features

# Split into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(features), 0.7 * nrow(features))
train_features <- features[train_indices, ]
test_features <- features[-train_indices, ]
train_target <- target[train_indices]
test_target <- target[-train_indices]

# Train Random Forest model
rf_model <- randomForest(x = train_features, y = as.factor(train_target), ntree = 100)

# Make predictions
predictions <- predict(rf_model, test_features)

# Evaluate model
caret::confusionMatrix(as.factor(test_target),predictions,positive = "1")

# Generate buy/sell signals
signals <- data.frame(Date = index(prices)[-c(1:5)], Price = prices[-c(1:5)], Signal = NA)
signals$Signal[train_indices] <- as.numeric(as.character(predict(rf_model, features[train_indices, ])))
signals$Signal[-train_indices] <- as.numeric(as.character(predictions))

# Plot signals
plot(prices, type = "l", main = "Random Forest Buy/Sell Signals", xlab = "Date", ylab = "Price")
buy_signals <- signals$Signal == 1
sell_signals <- signals$Signal == 0
points(prices[buy_signals], col = "green", pch = 19)
points(prices[sell_signals], col = "red", pch = 19)

# Simulate the returns
returns <- numeric(length(test_target) - 1)
for (i in 1:(length(test_target) - 1)) {
  if (signals$Signal[train_indices[length(train_indices)] + i] == 1) {
    entry_price <- as.numeric(prices[train_indices[length(train_indices)] + i])
    exit_price <- as.numeric(prices[train_indices[length(train_indices)] + i + 1])
    returns[i] <- (exit_price - entry_price) / entry_price
  }
}

# Calculate cumulative return
cumulative_return <- prod(1 + returns, na.rm = TRUE) - 1
cat("Cumulative Return over the test set: ", cumulative_return * 100, "%\n")
