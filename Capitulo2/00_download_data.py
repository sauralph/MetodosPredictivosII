import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np



# Your Alpha Vantage API key
# Load API key from .env file
load_dotenv('.env')
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

# Initialize the TimeSeries object
ts = TimeSeries(key=api_key, output_format='pandas')

# Get intraday data for SPY (S&P 500 ETF) - interval can be '1min', '5min', '15min', '30min', or '60min'
data, meta_data = ts.get_intraday(symbol='SPY', interval='1min', outputsize='full')

# Display the first few rows
print(data.head())

# Rename columns for convenience
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Plot the closing prices
data['Close'].plot(title='Intraday Time Series for the SPY stock (1 min)')
plt.show()
# Save the dataframe as a parquet file
data.to_parquet('spy_future.parquet')

# Get intraday data for Eurostoxx 50 futures
# The symbol for Eurostoxx 50 on Yahoo Finance is '^STOXX50E'
data_esx = yf.download(tickers='^STOXX50E', interval='1m', period='5d')
print(data_esx.head())

# Simulate volume data
np.random.seed(42)  # For reproducibility
n = len(data_esx)
mean_volume = 1000 
volume_noise = np.random.normal(0, 200, n)
base_volume = np.full(n, mean_volume)
simulated_volume = base_volume + np.cumsum(volume_noise)
simulated_volume[simulated_volume < 0] = mean_volume
hours = data_esx.index.hour
daily_pattern = np.sin((hours - 8) * (np.pi / 8)) * 200 + mean_volume  # Arbitrary pattern

# Combine the random walk and daily pattern
final_volume = (simulated_volume + daily_pattern) / 2

# Add the simulated volume to the dataframe
data_esx['Volume'] = final_volume

fig, ax1 = plt.subplots()
ax1.set_xlabel('DateTime')
ax1.set_ylabel('Close', color='tab:blue')
ax1.plot(data_esx.index, data_esx['Close'], color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Volume', color='tab:green')
ax2.plot(data_esx.index, data_esx['Volume'], color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')

fig.tight_layout()
plt.title('Intraday Time Series for Eurostoxx 50 Index with Simulated Volume (1 min)')
plt.show()

# Save the Eurostoxx 50 futures dataframe as a parquet file
data_esx.to_parquet('eurostoxx50_future.parquet')


# Download EUR/USD FX data
fx_data = yf.download('EURUSD=X', interval='1d')
fx_data = fx_data[['Close']]
fx_data.columns = ['EURUSD']
fx_data.to_parquet('eur_to_usd.parquet')
