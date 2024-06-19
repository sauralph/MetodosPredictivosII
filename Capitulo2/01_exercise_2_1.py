# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera


# Read the parquet file
tick_data = pd.read_parquet('spy_future.parquet')

# Display the first few rows of the data
print(tick_data.head())

# Ensure the 'date' column is a datetime type
tick_data.index = pd.to_datetime(tick_data.index)

# Form tick bars
def form_tick_bars(df, tick_size=100):
    df['tick_group'] = (np.arange(len(df)) // tick_size) + 1
    tick_bars = df.groupby('tick_group').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).reset_index(drop=True)
    tick_bars['date'] = df.index[::tick_size]
    return tick_bars

# Form volume bars
def form_volume_bars(df, volume_threshold=1000):
    df['cumulative_volume'] = df['Volume'].cumsum()
    df['volume_group'] = (df['cumulative_volume'] // volume_threshold) + 1
    volume_bars = df.groupby('volume_group').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).reset_index(drop=True)
    volume_bars['date'] = df.groupby('volume_group').apply(lambda x: x.index[-1]).values
    return volume_bars

# Form dollar bars
def form_dollar_bars(df, dollar_threshold=100000):
    df['dollar_value'] = df['Close'] * df['Volume']
    df['cumulative_dollar'] = df['dollar_value'].cumsum()
    df['dollar_group'] = (df['cumulative_dollar'] // dollar_threshold) + 1
    dollar_bars = df.groupby('dollar_group').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).reset_index(drop=True)
    dollar_bars['date'] = df.groupby('dollar_group').apply(lambda x: x.index[-1]).values
    return dollar_bars

tick_bars = form_tick_bars(tick_data)
volume_bars = form_volume_bars(tick_data)
dollar_bars = form_dollar_bars(tick_data)

# Ensure 'date' column is datetime
tick_bars['date'] = pd.to_datetime(tick_bars['date'])
volume_bars['date'] = pd.to_datetime(volume_bars['date'])
dollar_bars['date'] = pd.to_datetime(dollar_bars['date'])

# Count the number of bars produced weekly
tick_bars['week'] = tick_bars['date'].dt.to_period('W')
volume_bars['week'] = volume_bars['date'].dt.to_period('W')
dollar_bars['week'] = dollar_bars['date'].dt.to_period('W')

tick_weekly_count = tick_bars['week'].value_counts().sort_index()
volume_weekly_count = volume_bars['week'].value_counts().sort_index()
dollar_weekly_count = dollar_bars['week'].value_counts().sort_index()

# Plot the time series of the bar count
plt.figure(figsize=(14, 7))
plt.plot(tick_weekly_count.index.to_timestamp(), tick_weekly_count.values, label='Tick Bars')
plt.plot(volume_weekly_count.index.to_timestamp(), volume_weekly_count.values, label='Volume Bars')
plt.plot(dollar_weekly_count.index.to_timestamp(), dollar_weekly_count.values, label='Dollar Bars')
plt.legend()
plt.title('Weekly Bar Count for Tick, Volume, and Dollar Bars')
plt.xlabel('Week')
plt.ylabel('Bar Count')
plt.show()

# Compute the serial correlation of returns for the three bar types
def compute_serial_correlation(df):
    df['returns'] = df['Close'].pct_change()
    serial_corr = df['returns'].autocorr()
    return serial_corr

tick_serial_corr = compute_serial_correlation(tick_bars)
volume_serial_corr = compute_serial_correlation(volume_bars)
dollar_serial_corr = compute_serial_correlation(dollar_bars)

print(f"Serial Correlation of Tick Bars: {tick_serial_corr}")
print(f"Serial Correlation of Volume Bars: {volume_serial_corr}")
print(f"Serial Correlation of Dollar Bars: {dollar_serial_corr}")

# Determine the bar type with the lowest serial correlation
bar_types = ['Tick Bars', 'Volume Bars', 'Dollar Bars']
serial_correlations = [tick_serial_corr, volume_serial_corr, dollar_serial_corr]
lowest_serial_corr = min(serial_correlations)
best_bar_type = bar_types[serial_correlations.index(lowest_serial_corr)]

print(f"The bar type with the lowest serial correlation is: {best_bar_type}")

# Function to compute monthly variances of returns
def compute_monthly_variances(df):
    df['returns'] = df['Close'].pct_change()
    df['month'] = df['date'].dt.to_period('M')
    monthly_variances = df.groupby('month')['returns'].var().dropna()
    return monthly_variances

tick_monthly_variances = compute_monthly_variances(tick_bars)
volume_monthly_variances = compute_monthly_variances(volume_bars)
dollar_monthly_variances = compute_monthly_variances(dollar_bars)

# Compute the variance of variances for each bar type
tick_variance_of_variances = tick_monthly_variances.var()
volume_variance_of_variances = volume_monthly_variances.var()
dollar_variance_of_variances = dollar_monthly_variances.var()

print(f"Variance of variances for Tick Bars: {tick_variance_of_variances}")
print(f"Variance of variances for Volume Bars: {volume_variance_of_variances}")
print(f"Variance of variances for Dollar Bars: {dollar_variance_of_variances}")

# Determine which bar type exhibits the smallest variance of variances
variance_of_variances = {
    'Tick Bars': tick_variance_of_variances,
    'Volume Bars': volume_variance_of_variances,
    'Dollar Bars': dollar_variance_of_variances
}
smallest_variance_of_variances = min(variance_of_variances, key=variance_of_variances.get)
print(f"The bar type with the smallest variance of variances is: {smallest_variance_of_variances}")

# Apply the Jarque-Bera normality test on returns from the three bar types
tick_jb_stat, tick_jb_p = jarque_bera(tick_bars['returns'].dropna())
volume_jb_stat, volume_jb_p = jarque_bera(volume_bars['returns'].dropna())
dollar_jb_stat, dollar_jb_p = jarque_bera(dollar_bars['returns'].dropna())

print(f"Jarque-Bera test statistic for Tick Bars: {tick_jb_stat}, p-value: {tick_jb_p}")
print(f"Jarque-Bera test statistic for Volume Bars: {volume_jb_stat}, p-value: {volume_jb_p}")
print(f"Jarque-Bera test statistic for Dollar Bars: {dollar_jb_stat}, p-value: {dollar_jb_p}")

# Determine which bar type achieves the lowest Jarque-Bera test statistic
jb_statistics = {
    'Tick Bars': tick_jb_stat,
    'Volume Bars': volume_jb_stat,
    'Dollar Bars': dollar_jb_stat
}
lowest_jb_statistic = min(jb_statistics, key=jb_statistics.get)
print(f"The bar type with the lowest Jarque-Bera test statistic is: {lowest_jb_statistic}")
