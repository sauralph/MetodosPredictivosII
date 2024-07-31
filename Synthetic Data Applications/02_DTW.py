import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Get Bitcoin and Ethereum price data from Yahoo Finance
btc_data = yf.download("BTC-USD", start="2024-01-01", end="2024-07-31")
eth_data = yf.download("ETH-USD", start="2024-01-01", end="2024-07-31")

# Calculate daily closing prices
btc_prices = btc_data['Close']
eth_prices = eth_data['Close']

# Align the time series by merging them
merged_returns = np.vstack((btc_prices, eth_prices)).T

# Normalize the prices
btc_aligned = btc_prices / np.mean(btc_prices)
eth_aligned = eth_prices / np.mean(eth_prices)

btc_array = np.array(list(enumerate(btc_aligned, start=1)))
eth_array = np.array(list(enumerate(eth_aligned, start=1)))

# Perform DTW
distance, path = fastdtw(btc_array, eth_array, dist=euclidean)

# Print the DTW distance
print(f"Distancia DTW: {distance}")


# Plot the alignment
fig, ax = plt.subplots()
for (map_btc, map_eth) in path:
    ax.plot([btc_aligned.index[map_btc], eth_aligned.index[map_eth]], [btc_aligned[map_btc], eth_aligned[map_eth]], color='gray')
ax.plot(btc_aligned, label='BTC', color='steelblue')
ax.plot(eth_aligned, label='ETH', color='orange')
ax.legend()
plt.title('DTW entre BTC y ETH')
plt.show()

# Plot the original time series
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(btc_aligned, label='BTC', color='steelblue')
ax1.set_title('Precio Bitcoin (BTC)')
ax1.set_ylabel('Precio Normalizado')

ax2.plot(eth_aligned, label='ETH', color='orange')
ax2.set_title('Precio Ethereum (ETH)')
ax2.set_ylabel('Precio Normalizado')
ax2.set_xlabel('Fecha')

plt.tight_layout()
plt.show()
