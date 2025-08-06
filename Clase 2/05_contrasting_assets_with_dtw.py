import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# List of stocks
stocks = ["SAMI.BA", "SUPV.BA", "BTC-USD", "YPFD.BA"]

start_date = "2011-01-02"
end_date = datetime.now().strftime('%Y-%m-%d')

# Fetch historical stock data
data = yf.download(stocks, start=start_date, end=end_date)['Close']
data.interpolate(method='linear', inplace=True)

# drop na
data = data.dropna()


# Normalize the prices
normalized_data = data.div(data.mean())

# Function to plot DTW alignment
def plot_dtw_alignment(stock1, stock2):
    stock1_prices = normalized_data[stock1]
    stock2_prices = normalized_data[stock2]
    
    # Cast the normalized data to arrays with enumerated indices
    stock1_array = np.array(list(enumerate(stock1_prices, start=1)))
    stock2_array = np.array(list(enumerate(stock2_prices, start=1)))
    
    # Perform DTW
    distance, path = fastdtw(stock1_array, stock2_array, dist=euclidean)
    
    # Plot the alignment
    fig, ax = plt.subplots()
    for (map_stock1, map_stock2) in path:
        ax.plot([stock1_prices.index[map_stock1-1], stock2_prices.index[map_stock2-1]], [stock1_prices.iloc[map_stock1-1], stock2_prices.iloc[map_stock2-1]], color='gray')
    ax.plot(stock1_prices, label=stock1, color='steelblue')
    ax.plot(stock2_prices, label=stock2, color='orange')
    ax.legend()
    plt.title(f'DTW entre {stock1} y {stock2}')
    plt.show()

# Plot DTW alignment for SAMI.BA vs SUPV.BA
plot_dtw_alignment("SAMI.BA", "SUPV.BA")

# Plot DTW alignment for BTC-USD vs YPFD.BA
plot_dtw_alignment("BTC-USD", "YPFD.BA")
