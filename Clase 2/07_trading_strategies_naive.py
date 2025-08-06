import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from sklearn.model_selection import ParameterGrid

# Get Bitcoin historical data from Yahoo Finance
data = yf.download("BTC-USD", start="2020-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))

# Store the data in a variable
prices = data['Close']

# Interpolate missing values
prices = prices.interpolate()

# Split the data into training (pre-2024) and testing (2024) sets
train_prices = prices[:'2023-12-31']
test_prices = prices['2024-01-01':]

# HODL strategy -----------------------------------------------------------
hodl_return = (test_prices[-1] - test_prices[0]) / test_prices[0]
print("HODL Cumulative Return: {:.2f}%".format(hodl_return * 100))

# SMA Strategy ------------------------------------------------------------
def optimize_sma(train_prices):
    best_return = -np.inf
    best_short_window = 0
    best_long_window = 0
    param_grid = ParameterGrid({'short_window': range(2, 10), 'long_window': range(20, 50, 5)})
    
    for params in param_grid:
        short_ma = train_prices.rolling(window=params['short_window']).mean()
        long_ma = train_prices.rolling(window=params['long_window']).mean()
        signals = np.where((short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1)))[0]
        
        if len(signals) < 2:
            continue
        
        returns = [(train_prices.iloc[signals[i + 1]] - train_prices.iloc[signals[i]]) / train_prices.iloc[signals[i]] for i in range(len(signals) - 1)]
        cumulative_return = np.prod([1 + r for r in returns]) - 1
        
        if cumulative_return > best_return:
            best_return = cumulative_return
            best_short_window = params['short_window']
            best_long_window = params['long_window']
    
    return best_short_window, best_long_window

best_short_window, best_long_window = optimize_sma(train_prices)
short_ma = test_prices.rolling(window=best_short_window).mean()
long_ma = test_prices.rolling(window=best_long_window).mean()
long_signals_sma = np.where((short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1)))[0]

# Plot the SMA strategy
plt.figure(figsize=(14, 7))
plt.plot(test_prices, label='BTC Prices', color='black')
plt.plot(short_ma, label=f'{best_short_window}-day SMA', color='blue')
plt.plot(long_ma, label=f'{best_long_window}-day SMA', color='red')
plt.scatter(test_prices.index[long_signals_sma], test_prices[long_signals_sma], label='Buy Signal', color='green', marker='o')
plt.title('Moving Average Crossover Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Simulate the SMA returns
returns_sma = []
for i in range(len(long_signals_sma) - 1):
    entry_price = test_prices.iloc[long_signals_sma[i]]
    exit_price = test_prices.iloc[long_signals_sma[i + 1]]
    returns_sma.append((exit_price - entry_price) / entry_price)
cumulative_return_sma = np.prod([1 + r for r in returns_sma]) - 1
print("SMA Cumulative Return: {:.2f}%".format(cumulative_return_sma * 100))

# RSI Strategy ------------------------------------------------------------
def optimize_rsi(train_prices):
    best_return = -np.inf
    best_window = 0
    param_grid = ParameterGrid({'window': range(10, 20)})
    
    for params in param_grid:
        rsi = RSIIndicator(train_prices, window=params['window']).rsi()
        signals = np.where((rsi < 30) & (rsi.shift(1) >= 30))[0]
        
        if len(signals) < 2:
            continue
        
        returns = [(train_prices.iloc[signals[i + 1]] - train_prices.iloc[signals[i]]) / train_prices.iloc[signals[i]] for i in range(len(signals) - 1)]
        cumulative_return = np.prod([1 + r for r in returns]) - 1
        
        if cumulative_return > best_return:
            best_return = cumulative_return
            best_window = params['window']
    
    return best_window

best_rsi_window = optimize_rsi(train_prices)
rsi = RSIIndicator(test_prices, window=best_rsi_window).rsi()
long_signals_rsi = np.where((rsi < 30) & (rsi.shift(1) >= 30))[0]

# Plot the RSI strategy
plt.figure(figsize=(14, 7))
plt.plot(test_prices, label='BTC Prices', color='black')
plt.scatter(test_prices.index[long_signals_rsi], test_prices[long_signals_rsi], label='Buy Signal', color='green', marker='o')
plt.title('RSI Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Simulate the RSI returns
returns_rsi = []
for i in range(len(long_signals_rsi) - 1):
    entry_price = test_prices.iloc[long_signals_rsi[i]]
    exit_price = test_prices.iloc[long_signals_rsi[i + 1]]
    returns_rsi.append((exit_price - entry_price) / entry_price)
cumulative_return_rsi = np.prod([1 + r for r in returns_rsi]) - 1
print("RSI Cumulative Return: {:.2f}%".format(cumulative_return_rsi * 100))

# Bollinger Bands Strategy ------------------------------------------------
def optimize_bbands(train_prices):
    best_return = -np.inf
    best_window = 0
    param_grid = ParameterGrid({'window': range(10, 30, 2), 'window_dev': [1, 2, 3]})
    
    for params in param_grid:
        bbands = BollingerBands(close=train_prices, window=params['window'], window_dev=params['window_dev'])
        signals = np.where((train_prices < bbands.bollinger_lband()) & (train_prices.shift(1) >= bbands.bollinger_lband().shift(1)))[0]
        
        if len(signals) < 2:
            continue
        
        returns = [(train_prices.iloc[signals[i + 1]] - train_prices.iloc[signals[i]]) / train_prices.iloc[signals[i]] for i in range(len(signals) - 1)]
        cumulative_return = np.prod([1 + r for r in returns]) - 1
        
        if cumulative_return > best_return:
            best_return = cumulative_return
            best_window = params['window']
            best_window_dev = params['window_dev']
    
    return best_window, best_window_dev

best_bbands_window, best_bbands_window_dev = optimize_bbands(train_prices)
bbands = BollingerBands(close=test_prices, window=best_bbands_window, window_dev=best_bbands_window_dev)
long_signals_bbands = np.where((test_prices < bbands.bollinger_lband()) & (test_prices.shift(1) >= bbands.bollinger_lband().shift(1)))[0]

# Plot the Bollinger Bands strategy
plt.figure(figsize=(14, 7))
plt.plot(test_prices, label='BTC Prices', color='black')
plt.plot(bbands.bollinger_lband(), label='Lower Band', color='red')
plt.scatter(test_prices.index[long_signals_bbands], test_prices[long_signals_bbands], label='Buy Signal', color='green', marker='o')
plt.title('Bollinger Bands Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Simulate the Bollinger Bands returns
returns_bbands = []
for i in range(len(long_signals_bbands) - 1):
    entry_price = test_prices.iloc[long_signals_bbands[i]]
    exit_price = test_prices.iloc[long_signals_bbands[i + 1]]
    returns_bbands.append((exit_price - entry_price) / entry_price)
cumulative_return_bbands = np.prod([1 + r for r in returns_bbands]) - 1
print("Bollinger Bands Cumulative Return: {:.2f}%".format(cumulative_return_bbands * 100))

# CUSUM Filter Strategy ---------------------------------------------------
def cusum_filter(prices, threshold):
    n = len(prices)
    S = np.zeros(n)
    long_signals = np.zeros(n)
    for t in range(1, n):
        S[t] = max(0, S[t-1] + prices[t] - prices[t-1])
        if S[t] >= threshold:
            long_signals[t] = 1  # Generate a long signal
            S[t] = 0  # Reset S[t]
    return np.where(long_signals == 1)[0]

# Optimize CUSUM threshold
def optimize_cusum(train_prices):
    best_return = -np.inf
    best_threshold = 0
    param_grid = ParameterGrid({'threshold': range(500, 1500, 100)})
    
    for params in param_grid:
        signals = cusum_filter(train_prices.values, params['threshold'])
        
        if len(signals) < 2:
            continue
        
        returns = [(train_prices.iloc[signals[i + 1]] - train_prices.iloc[signals[i]]) / train_prices.iloc[signals[i]] for i in range(len(signals) - 1)]
        cumulative_return = np.prod([1 + r for r in returns]) - 1
        
        if cumulative_return > best_return:
            best_return = cumulative_return
            best_threshold = params['threshold']
    
    return best_threshold

best_cusum_threshold = optimize_cusum(train_prices)
long_signals_cusum = cusum_filter(test_prices.values, best_cusum_threshold)

# Plot the CUSUM Filter strategy
plt.figure(figsize=(14, 7))
plt.plot(test_prices, label='BTC Prices', color='black')
plt.scatter(test_prices.index[long_signals_cusum], test_prices[long_signals_cusum], label='Buy Signal', color='green', marker='o')
plt.title('CUSUM Filter Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Simulate the CUSUM Filter returns
returns_cusum = []
for i in range(len(long_signals_cusum) - 1):
    entry_price = test_prices[long_signals_cusum[i]]
    exit_price = test_prices[long_signals_cusum[i + 1]]
    returns_cusum.append((exit_price - entry_price) / entry_price)
cumulative_return_cusum = np.prod([1 + r for r in returns_cusum]) - 1
print("CUSUM Filter Cumulative Return: {:.2f}%".format(cumulative_return_cusum * 100))

# Create a markdown table with the cumulative returns --------------------
print("\n# Cumulative Returns for All Strategies\n")
print("| Strategy | Cumulative Return (%) |")
print("|----------|-----------------------|")
print("| HODL | {:.2f} |".format(hodl_return * 100))
print("| SMA | {:.2f} |".format(cumulative_return_sma * 100))
print("| RSI | {:.2f} |".format(cumulative_return_rsi * 100))
print("| Bollinger Bands | {:.2f} |".format(cumulative_return_bbands * 100))
print("| CUSUM Filter | {:.2f} |".format(cumulative_return_cusum * 100))
