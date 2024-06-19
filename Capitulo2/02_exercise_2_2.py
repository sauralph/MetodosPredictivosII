import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf



tick_data = pd.read_parquet('spy_future.parquet')

print(tick_data.head())

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



def compute_dollar_imbalance_bars(df, dollar_threshold):
    imbalance_bars = []
    bar = {'Open': None, 'High': -float('inf'), 'Low': float('inf'), 'Close': None, 'Volume': 0}
    dollar_sum = 0
    imbalance_sum = 0
    
    for index, row in df.iterrows():
        trade_dollar_value = row['Close'] * row['Volume']
        dollar_sum += trade_dollar_value
        
        if bar['Open'] is None:
            bar['Open'] = row['Open']
        
        bar['High'] = max(bar['High'], row['High'])
        bar['Low'] = min(bar['Low'], row['Low'])
        bar['Close'] = row['Close']
        bar['Volume'] += row['Volume']
        
        imbalance_sum += trade_dollar_value if row['Close'] >= row['Open'] else -trade_dollar_value
        
        if abs(imbalance_sum) >= dollar_threshold:
            imbalance_bars.append(bar)
            bar = {'Open': None, 'High': -float('inf'), 'Low': float('inf'), 'Close': None, 'Volume': 0}
            dollar_sum = 0
            imbalance_sum = 0
    
    return pd.DataFrame(imbalance_bars)

# Step 1: Calculate the Dollar Volume
tick_data['DollarVolume'] = tick_data['Close'] * tick_data['Volume']

# Step 2: Calculate the Mean Dollar Volume
mean_dollar_volume = tick_data['DollarVolume'].mean()

# Step 3: Set the Threshold
# You can choose a multiple of the mean dollar volume
threshold_multiple = 2  # This can be adjusted based on your preference
dollar_threshold = threshold_multiple * mean_dollar_volume

print(f'Mean Dollar Volume: {mean_dollar_volume}')
print(f'Dollar Threshold: {dollar_threshold}')

dollar_bars = compute_dollar_bars(tick_data, dollar_threshold)
print(dollar_bars.head())

dollar_imbalance_bars = compute_dollar_imbalance_bars(tick_data, dollar_threshold)
print(dollar_imbalance_bars.head())

dollar_bars = compute_dollar_bars(tick_data, dollar_threshold)
dollar_imbalance_bars = compute_dollar_imbalance_bars(tick_data, dollar_threshold)

# Function to plot auto-correlation
def plot_auto_correlation(series, title):
    plot_acf(series, lags=50)
    plt.title(title)
    plt.xlabel('Lags')
    plt.ylabel('Auto-correlation')
    plt.show()

# Plot auto-correlation for Dollar Bars
plot_auto_correlation(dollar_bars['Close'], 'Auto-correlation of Dollar Bars')

# Plot auto-correlation for Dollar Imbalance Bars
plot_auto_correlation(dollar_imbalance_bars['Close'], 'Auto-correlation of Dollar Imbalance Bars')