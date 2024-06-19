import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt



tick_data_spy = pd.read_parquet('spy_future.parquet')
tick_data_eurstoxx50 = pd.read_parquet('eurostoxx50_future.parquet')
eur_to_usd = pd.read_parquet('eur_to_usd.parquet')


print(tick_data_spy.head())
print(tick_data_eurstoxx50.head())
print(eur_to_usd.tail())

tick_data_eurstoxx50.index = tick_data_eurstoxx50.index.tz_localize(None)
eur_to_usd_resampled = eur_to_usd.resample('T').ffill()

# Adjust tick_data_eurstoxx50 with EUR/USD exchange rates
tick_data_eurstoxx50 = tick_data_eurstoxx50.merge(eur_to_usd_resampled, left_index=True, right_index=True, how='left')
tick_data_eurstoxx50['EURUSD'].fillna(method='ffill', inplace=True)
tick_data_eurstoxx50['Close_USD'] = tick_data_eurstoxx50['Close'] * tick_data_eurstoxx50['EURUSD']

# Compute dollar bars for both futures
def compute_dollar_bars(df, dollar_threshold):
    dollar_bars = []
    bar = {'Open': None, 'High': -float('inf'), 'Low': float('inf'), 'Close': None, 'Volume': 0}
    dollar_sum = 0

    for index, row in df.iterrows():
        trade_dollar_value = row['Close'] * row['Volume']
        dollar_sum += trade_dollar_value

        if bar['Open'] is None:
            bar['Open'] = row['Open']

        bar['High'] = max(bar['High'], row['High'])
        bar['Low'] = min(bar['Low'], row['Low'])
        bar['Close'] = row['Close']
        bar['Volume'] += row['Volume']

        if dollar_sum >= dollar_threshold:
            dollar_bars.append(bar)
            bar = {'Open': None, 'High': -float('inf'), 'Low': float('inf'), 'Close': None, 'Volume': 0}
            dollar_sum = 0

    return pd.DataFrame(dollar_bars)


# Step 1: Calculate the Dollar Volume
tick_data_spy['DollarVolume'] = tick_data_spy['Close'] * tick_data_spy['Volume']
mean_dollar_volume = tick_data_spy['DollarVolume'].mean()
threshold_multiple = 2  # This can be adjusted based on your preference
dollar_threshold_spy = threshold_multiple * mean_dollar_volume

tick_data_eurstoxx50['DollarVolume'] = tick_data_eurstoxx50['Close'] * tick_data_eurstoxx50['Volume']
mean_dollar_volume = tick_data_eurstoxx50['Close_USD'].mean()
threshold_multiple = 2  # This can be adjusted based on your preference
dollar_threshold_eurstoxx50 = threshold_multiple * mean_dollar_volume

# Convert OHLC columns to USD
tick_data_eurstoxx50['Open_USD'] = tick_data_eurstoxx50['Open'] * tick_data_eurstoxx50['EURUSD']
tick_data_eurstoxx50['High_USD'] = tick_data_eurstoxx50['High'] * tick_data_eurstoxx50['EURUSD']
tick_data_eurstoxx50['Low_USD'] = tick_data_eurstoxx50['Low'] * tick_data_eurstoxx50['EURUSD']
tick_data_eurstoxx50['Close_USD'] = tick_data_eurstoxx50['Close'] * tick_data_eurstoxx50['EURUSD']

# Create a new dataframe with the relevant columns
eurostoxx50_usd = tick_data_eurstoxx50[['Open_USD', 'High_USD', 'Low_USD', 'Close_USD', 'Volume']]
eurostoxx50_usd.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

dollar_bars_spy = compute_dollar_bars(tick_data_spy, dollar_threshold_spy)
dollar_bars_eurstoxx50 = compute_dollar_bars(eurostoxx50_usd, dollar_threshold_eurstoxx50)

# Apply the ETF trick to compute the {𝜔̂t} vector
dollar_bars_spy['Adjusted_Close'] = dollar_bars_spy['Close']
dollar_bars_eurstoxx50['Adjusted_Close'] = dollar_bars_eurstoxx50['Close']
omega_t = dollar_bars_spy['Adjusted_Close'] / dollar_bars_eurstoxx50['Adjusted_Close']

# Derive the time series of the S&P 500/Eurostoxx 50 spread
spread = dollar_bars_spy['Adjusted_Close'] - dollar_bars_eurstoxx50['Adjusted_Close']

# Confirm the series is stationary with an ADF test
result = adfuller(spread.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Plot the spread
spread.plot(title='S&P 500 / Eurostoxx 50 Spread')
plt.show()

# Check if the series is stationary
if result[1] <= 0.05:
    print("The series is stationary.")
else:
    print("The series is not stationary.")