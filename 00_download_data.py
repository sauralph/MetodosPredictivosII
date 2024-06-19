import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import yfinance as yf



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

# Display the first few rows
print(data_esx.head())

# Plot the closing prices for Eurostoxx 50 futures
data_esx['Close'].plot(title='Intraday Time Series for the Eurostoxx 50 futures (1 min)')
plt.show()

# Save the Eurostoxx 50 futures dataframe as a parquet file
data_esx.to_parquet('eurostoxx50_future.parquet')


# Download EUR/USD FX data
fx_data = yf.download('EURUSD=X', interval='1d')
fx_data = fx_data[['Close']]
fx_data.columns = ['EURUSD']
fx_data.to_parquet('eur_to_usd.parquet')
