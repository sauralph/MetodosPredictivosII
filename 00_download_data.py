import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

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
