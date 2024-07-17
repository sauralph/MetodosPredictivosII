import numpy as np
import pandas as pd
import yfinance as yf


# Step 1: Get Bitcoin historical data from Yahoo Finance
btc_data = yf.download("BTC-USD", start="2020-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
prices = btc_data['Close']

# Interpolate NA values
prices = prices.interpolate()

# Difference the series
prices = prices.diff().dropna()

prices

# Block bootstrapping function
def block_bootstrap(data, block_size, num_samples):
    n = len(data)
    bootstrapped_series = []
    
    for _ in range(num_samples):
        indices = np.arange(n)
        block_start = np.random.choice(indices[:-block_size])
        bootstrap_sample = []
        
        for _ in range(int(n/block_size)):
            block = data[block_start:block_start+block_size]
            bootstrap_sample.extend(block)
            block_start = np.random.choice(indices[:-block_size])
        
        bootstrapped_series.append(bootstrap_sample[:n])
    
    return np.array(bootstrapped_series)

# Parameters
block_size = 10
num_samples = 1000

# Perform block bootstrapping
bootstrapped_data = block_bootstrap(prices, block_size, num_samples)
# Example of fitting a model (ARIMA) to the bootstrapped series
from statsmodels.tsa.arima.model import ARIMA

model_params = []

for series in bootstrapped_data:
    model = ARIMA(series, order=(1, 0, 1))
    fitted_model = model.fit()
    model_params.append(fitted_model.params)

print(fitted_model.summary())

# Convert to DataFrame for analysis
model_params_df = pd.DataFrame(model_params, columns=['const', 'ar.L1', 'ma.L1','sigma2'])

# Calculate statistics (e.g., mean and confidence intervals)
param_means = model_params_df.mean()
param_ci = model_params_df.quantile([0.025, 0.975])

print("Parameter Means:")
print(param_means)
print("\n95% Confidence Intervals:")
print(param_ci)

