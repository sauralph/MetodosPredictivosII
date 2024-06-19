import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tick_data_spy = pd.read_parquet('spy_future.parquet')
tick_data_spy



# Assuming tick_data_spy is your dataframe

# Part (a): Compute Bollinger Bands
def compute_bollinger_bands(df, window=20, width=0.05):
    df['Rolling_Mean'] = df['Close'].rolling(window=window).mean()
    df['Rolling_Std'] = df['Close'].rolling(window=window).std()
    df['Upper_Band'] = df['Rolling_Mean'] * (1 + width)
    df['Lower_Band'] = df['Rolling_Mean'] * (1 - width)
    
    # Count the number of times prices cross the bands
    df['Cross_Upper'] = ((df['Close'].shift(1) <= df['Upper_Band'].shift(1)) & (df['Close'] > df['Upper_Band'])).astype(int)
    df['Cross_Lower'] = ((df['Close'].shift(1) >= df['Lower_Band'].shift(1)) & (df['Close'] < df['Lower_Band'])).astype(int)
    
    num_cross_upper = df['Cross_Upper'].sum()
    num_cross_lower = df['Cross_Lower'].sum()
    
    return df, num_cross_upper, num_cross_lower

# Apply Bollinger Bands calculation
tick_data_spy, num_cross_upper, num_cross_lower = compute_bollinger_bands(tick_data_spy)

print(f'Number of times prices crossed the upper band: {num_cross_upper}')
print(f'Number of times prices crossed the lower band: {num_cross_lower}')

# Part (b): Sample using a CUSUM Filter
def cusum_filter(df, h=0.05):
    t_events = []
    s_pos, s_neg = 0, 0
    
    diff = df['Close'].diff()
    for i in range(1, len(diff)):
        s_pos = max(0, s_pos + diff.iloc[i])
        s_neg = min(0, s_neg + diff.iloc[i])
        
        if s_pos > h:
            s_pos = 0
            t_events.append(diff.index[i])
        elif s_neg < -h:
            s_neg = 0
            t_events.append(diff.index[i])
    
    return pd.DatetimeIndex(t_events)

# Apply CUSUM filter
cusum_events = cusum_filter(tick_data_spy, h=0.05)
cusum_filtered_data = tick_data_spy.loc[cusum_events]

print(f'Number of samples after CUSUM filter: {len(cusum_filtered_data)}')

# Part (c): Compute rolling standard deviation
def compute_rolling_std(df, window=20):
    return df['Close'].rolling(window=window).std()

# Compute rolling standard deviation for dollar bars and CUSUM-filtered series
rolling_std_dollar_bars = compute_rolling_std(tick_data_spy)
rolling_std_cusum_filtered = compute_rolling_std(cusum_filtered_data)

# Determine which series is least heteroscedastic
heteroscedasticity_dollar_bars = rolling_std_dollar_bars.var()
heteroscedasticity_cusum_filtered = rolling_std_cusum_filtered.var()

print(f'Heteroscedasticity of dollar bars series: {heteroscedasticity_dollar_bars}')
print(f'Heteroscedasticity of CUSUM-filtered series: {heteroscedasticity_cusum_filtered}')

if heteroscedasticity_dollar_bars < heteroscedasticity_cusum_filtered:
    print("Dollar bars series is least heteroscedastic.")
else:
    print("CUSUM-filtered series is least heteroscedastic.")

fig, ax1 = plt.subplots()

ax1.plot(tick_data_spy.index, tick_data_spy['Close'], label='Close')
ax1.plot(tick_data_spy.index, tick_data_spy['Rolling_Mean'], label='Rolling Mean')
ax1.plot(tick_data_spy.index, tick_data_spy['Upper_Band'], label='Upper Band')
ax1.plot(tick_data_spy.index, tick_data_spy['Lower_Band'], label='Lower Band')

ax1.legend()
plt.title('Bollinger Bands and Close Prices')
plt.show()

# Plotting rolling standard deviation for comparison
fig, ax2 = plt.subplots()
ax2.plot(rolling_std_dollar_bars.index, rolling_std_dollar_bars, label='Dollar Bars Rolling Std')
ax2.plot(rolling_std_cusum_filtered.index, rolling_std_cusum_filtered, label='CUSUM-filtered Rolling Std')

ax2.legend()
plt.title('Rolling Standard Deviation Comparison')
plt.show()
