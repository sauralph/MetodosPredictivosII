import numpy as np
import pandas as pd
import yfinance as yf
import importlib


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

def apply_trading_logic(paths, mesh, T_max):
    N = paths.shape[0]
    results = []

    for pi, pi_bar in mesh:
        final_pnl = []
        
        for j in range(N):
            for t in range(T_max):
                pnl = paths[j, t] - paths[j, 0]
                if pnl <= pi or pnl >= pi_bar:
                    final_pnl.append(pnl)
                    break
                if t == T_max - 1:
                    final_pnl.append(pnl)
        
        sharpe_ratio = np.mean(final_pnl) / np.std(final_pnl)
        results.append([pi, pi_bar, sharpe_ratio])
    
    return pd.DataFrame(results, columns=['pi', 'pi_bar', 'sharpe_ratio'])

def determine_optimal_rule(results):
    return results.loc[results['sharpe_ratio'].idxmax()]


def construct_mesh(sigma):
    pi = np.linspace(-12 * sigma, -sigma, 10)
    pi_bar = np.linspace(sigma, 12 * sigma, 10)
    mesh = np.array(np.meshgrid(pi, pi_bar)).T.reshape(-1, 2)
    return mesh

sigma_hat = 1231.50
T_max = 100

results = apply_trading_logic(bootstrapped_data, construct_mesh(sigma_hat), T_max)
determine_optimal_rule(results)
sorted_results = results.sort_values(by=['pi', 'pi_bar'])
pivot_table = sorted_results.pivot_table('sharpe_ratio','pi', 'pi_bar')

plt.figure(figsize=(16, 9))
contour = plt.contourf(pivot_table.columns, pivot_table.index, pivot_table, cmap='viridis')
plt.title('Exploracion de espacio de parametros')
plt.xlabel('pi')
plt.ylabel('pi_bar')
plt.savefig('R_space_contour_bs.png')
plt.show()