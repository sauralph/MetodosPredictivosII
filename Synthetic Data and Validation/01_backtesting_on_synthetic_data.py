import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Step 1: Get Bitcoin historical data from Yahoo Finance
btc_data = yf.download("BTC-USD", start="2020-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
prices = btc_data['Close']

# Interpolate NA values
prices = prices.interpolate()

# Difference the series
prices = prices.diff().dropna()

# Step 1: Estimate input parameters {sigma, phi}
def estimate_parameters(prices, E0):
    T_max = len(prices)
    X = []
    Y = []
    
    for t in range(T_max - 1):
        X.append(prices[t] - E0)
        Y.append(prices[t + 1])
    
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y).reshape(-1, 1)
    
    model = LinearRegression().fit(X, Y)
    phi_hat = model.coef_[0][0]
    
    Z = np.full((T_max - 1, 1), E0)
    residuals = Y - Z - phi_hat * X
    sigma_hat = np.sqrt(np.var(residuals))
    
    return sigma_hat, phi_hat

# Step 2: Construct mesh of stop-loss and profit-taking pairs
def construct_mesh(sigma):
    pi = np.linspace(-12 * sigma, -sigma, 10)
    pi_bar = np.linspace(sigma, 12 * sigma, 10)
    mesh = np.array(np.meshgrid(pi, pi_bar)).T.reshape(-1, 2)
    return mesh

# Step 3: Generate paths
def generate_paths(N, T_max, sigma, phi, initial_price, E0):
    paths = np.zeros((N, T_max))
    paths[:, 0] = initial_price
    for i in range(N):
        for t in range(1, T_max):
            paths[i, t] = E0 + phi * (paths[i, t - 1] - E0) + np.random.normal(0, sigma)
    return paths

# Step 4: Apply stop-loss and profit-taking logic
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

# Step 5: Determine optimal trading rule
def determine_optimal_rule(results):
    return results.loc[results['sharpe_ratio'].idxmax()]

# Example usage
# Define input data
_E0 = prices.mean()

# Step 1: Estimate parameters
sigma_hat, phi_hat = estimate_parameters(prices.values, _E0)

# Step 2: Construct mesh
mesh = construct_mesh(sigma_hat)

# Step 3: Generate paths
N = 100000
T_max = 100
initial_price = prices.iloc[0]
paths = generate_paths(N, T_max, sigma_hat, phi_hat, initial_price, _E0)

# Step 4: Apply trading logic
results = apply_trading_logic(paths, mesh, T_max)

# Step 5: Determine optimal rule
optimal_rule = determine_optimal_rule(results)


print(optimal_rule)

# Plot the contour
sorted_results = results.sort_values(by=['pi', 'pi_bar'])
pivot_table = sorted_results.pivot_table('sharpe_ratio','pi', 'pi_bar')

plt.figure(figsize=(16, 9))
contour = plt.contourf(pivot_table.columns, pivot_table.index, pivot_table, cmap='viridis')
plt.title('Exploracion de espacio de parametros')
plt.xlabel('pi')
plt.ylabel('pi_bar')
plt.savefig('R_space_contour.png')
plt.show()

# Testing the strategy on Bitcoin data
def apply_strategy(prices, pi, pi_bar):
    initial_price = prices.iloc[0]
    pnl = 0
    for t in range(1, len(prices)):
        current_price = prices.iloc[t]
        pnl = current_price - initial_price
        
        if pnl <= pi or pnl >= pi_bar:
            break
    return pnl

# Define function to test strategy on historical data
def test_strategy(prices, pi, pi_bar):
    pnl_list = []
    return_list = []
    for start in range(len(prices) - 1):
        pnl = apply_strategy(prices[start:], pi, pi_bar)
        pnl_list.append(pnl)
        return_list.append(pnl/prices.iloc[start])

    sharpe_ratio = np.mean(pnl_list) / np.std(pnl_list)
    return pnl_list, sharpe_ratio, return_list

pi = optimal_rule['pi']
pi_bar = optimal_rule['pi_bar']

pnl_list, sharpe_ratio,return_list = test_strategy(prices, pi, pi_bar)

np.mean(pnl_list), np.std(pnl_list), sharpe_ratio
np.mean(return_list), np.std(return_list), sharpe_ratio
plt.figure(figsize=(16,9))
plt.hist(pnl_list, bins=50, alpha=0.75)
plt.title('Distribucion PnL')
plt.xlabel('PnL')
plt.ylabel('Frecuencia')
plt.savefig('PnL_distribution.png')
plt.show()
