import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample

# Get Bitcoin historical data from Yahoo Finance
data = yf.download("BTC-USD", start="2020-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))

# Store the data in a variable
prices = data['Close']

# Interpolate missing values
prices = prices.interpolate()

# Calculate returns and create labels
returns = prices.pct_change().shift(-1)
buy_signals = (returns > 0).astype(int)

# Create features
window_sizes = [5, 10, 15, 20]
features = pd.DataFrame(index=prices.index)
for window in window_sizes:
    features[f'rolling_mean_{window}'] = prices.rolling(window=window).mean()
    features[f'rolling_std_{window}'] = prices.rolling(window=window).std()
    features[f'rolling_max_{window}'] = prices.rolling(window=window).max()
    features[f'rolling_min_{window}'] = prices.rolling(window=window).min()

# Drop rows with NaN values
features = features.dropna()
buy_signals = buy_signals.loc[features.index]

# Split the data into training (pre-2024) and testing (2024) sets
train_features = features[:'2023-12-31']
train_labels = buy_signals[:'2023-12-31']
test_features = features['2024-01-01':]
test_labels = buy_signals['2024-01-01':]

# Bootstrap the training data
n_bootstrap_samples = 100
bootstrap_features = []
bootstrap_labels = []
for _ in range(n_bootstrap_samples):
    boot_features, boot_labels = resample(train_features, train_labels)
    bootstrap_features.append(boot_features)
    bootstrap_labels.append(boot_labels)

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search to optimize for cumulative returns
best_model = None
best_return = -np.inf
for params in ParameterGrid(param_grid):
    cumulative_returns = []
    for boot_features, boot_labels in zip(bootstrap_features, bootstrap_labels):
        rf = RandomForestClassifier(**params, random_state=42)
        rf.fit(boot_features, boot_labels)
        # Predict buy signals
        boot_predictions = rf.predict(boot_features)
        # Calculate returns
        boot_prices = prices.loc[boot_features.index]
        returns_rf = (boot_prices.shift(-1) - boot_prices)[boot_predictions == 1] / boot_prices[boot_predictions == 1]
        cumulative_return_rf = np.prod(1 + returns_rf) - 1
        cumulative_returns.append(cumulative_return_rf)
    avg_cumulative_return = np.mean(cumulative_returns)
    if avg_cumulative_return > best_return:
        best_return = avg_cumulative_return
        best_model = rf

# Evaluate the model on the test set
test_predictions = best_model.predict(test_features)

# Calculate test returns
test_prices = prices.loc[test_features.index]
returns_rf_test = (test_prices.shift(-1) - test_prices)[test_predictions == 1] / test_prices[test_predictions == 1]
cumulative_return_rf_test = np.prod(1 + returns_rf_test) - 1

print("Random Forest Test Cumulative Return: {:.2f}%".format(cumulative_return_rf_test * 100))
print("\nClassification Report:\n", classification_report(test_labels, test_predictions))

# Plot the test buy signals
plt.figure(figsize=(14, 7))
plt.plot(test_prices, label='BTC Prices', color='black')
plt.scatter(test_prices.index[test_predictions == 1], test_prices[test_predictions == 1], label='Buy Signal', color='green', marker='o')
plt.title('Random Forest Buy Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Create a markdown table with the cumulative returns --------------------
print("\n# Cumulative Returns for All Strategies\n")
print("| Strategy | Cumulative Return (%) |")
print("|----------|-----------------------|")
print("| Random Forest | {:.2f} |".format(cumulative_return_rf_test * 100))
