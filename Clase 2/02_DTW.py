import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pandas as pd
import requests
from datetime import datetime, timedelta

# Option 1: Using CoinGecko API (free, no API key required)
def get_cryptocurrency_data_gecko(symbol, start_date, end_date):
    """
    Download cryptocurrency data from CoinGecko API
    """
    # Convert symbol to CoinGecko format
    symbol_map = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'BTC-USD': 'bitcoin',
        'ETH-USD': 'ethereum'
    }
    
    coin_id = symbol_map.get(symbol, symbol.lower())
    
    # Convert dates to Unix timestamps
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        'vs_currency': 'usd',
        'from': start_ts,
        'to': end_ts
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract prices and timestamps
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df = df[['price']]
        df.columns = ['Close']
        
        return df
    except Exception as e:
        print(f"Error downloading data from CoinGecko: {e}")
        return None

# Option 2: Using Alpha Vantage API (requires free API key)
def get_cryptocurrency_data_alphavantage(symbol, api_key):
    """
    Download cryptocurrency data from Alpha Vantage API
    Requires free API key from https://www.alphavantage.co/
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": symbol,
        "market": "USD",
        "apikey": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Time Series (Digital Currency Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (Digital Currency Daily)"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df = df[['4a. close (USD)']]
            df.columns = ['Close']
            return df
        else:
            print(f"Error: {data.get('Note', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Error downloading data from Alpha Vantage: {e}")
        return None

# Option 3: Using FRED (Federal Reserve Economic Data) for some crypto ETFs
def get_crypto_etf_data_fred(symbol):
    """
    Download crypto ETF data from FRED
    """
    try:
        import pandas_datareader.data as web
        # Some crypto-related ETFs available on FRED
        etf_map = {
            'BTC': 'GBTC',  # Grayscale Bitcoin Trust
            'ETH': 'ETHE'   # Grayscale Ethereum Trust
        }
        
        etf_symbol = etf_map.get(symbol, symbol)
        df = web.DataReader(etf_symbol, 'fred', start='2024-01-01', end='2024-07-31')
        df.columns = ['Close']
        return df
    except Exception as e:
        print(f"Error downloading data from FRED: {e}")
        return None

# Option 4: Using Binance API (free, no API key required for public data)
def get_cryptocurrency_data_binance(symbol, start_date, end_date):
    """
    Download cryptocurrency data from Binance API
    """
    # Convert symbol to Binance format
    symbol_map = {
        'BTC': 'BTCUSDT',
        'ETH': 'ETHUSDT',
        'BTC-USD': 'BTCUSDT',
        'ETH-USD': 'ETHUSDT'
    }
    
    binance_symbol = symbol_map.get(symbol, symbol)
    
    # Convert dates to milliseconds
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': binance_symbol,
        'interval': '1d',
        'startTime': start_ts,
        'endTime': end_ts
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df['Close'] = df['close'].astype(float)
        df = df[['Close']]
        
        return df
    except Exception as e:
        print(f"Error downloading data from Binance: {e}")
        return None

# Choose your preferred data source:
# Option 1: CoinGecko (recommended - free, no API key)
print("Downloading data from CoinGecko...")
btc_data = get_cryptocurrency_data_gecko("BTC", "2024-01-01", "2024-07-31")
eth_data = get_cryptocurrency_data_gecko("ETH", "2024-01-01", "2024-07-31")

# Option 2: Binance (alternative - free, no API key)
if btc_data is None or eth_data is None:
    print("Trying Binance API...")
    btc_data = get_cryptocurrency_data_binance("BTC", "2024-01-01", "2024-07-31")
    eth_data = get_cryptocurrency_data_binance("ETH", "2024-01-01", "2024-07-31")

# Option 3: Alpha Vantage (requires API key)
# Uncomment and add your API key if you want to use Alpha Vantage
# ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY_HERE"
# btc_data = get_cryptocurrency_data_alphavantage("BTC", ALPHA_VANTAGE_API_KEY)
# eth_data = get_cryptocurrency_data_alphavantage("ETH", ALPHA_VANTAGE_API_KEY)

# Option 4: Fallback to Yahoo Finance if other sources fail
if btc_data is None or eth_data is None:
    print("Falling back to Yahoo Finance...")
    btc_data = yf.download("BTC-USD", start="2024-01-01", end="2024-07-31")
    eth_data = yf.download("ETH-USD", start="2024-01-01", end="2024-07-31")

# Check if data was successfully downloaded
if btc_data is None or eth_data is None:
    print("Error: Could not download data from any source")
    exit()

print(f"Successfully downloaded data:")
print(f"BTC data shape: {btc_data.shape}")
print(f"ETH data shape: {eth_data.shape}")

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
