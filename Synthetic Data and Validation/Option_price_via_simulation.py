# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set parameters
S0 = 100      # Initial stock price
K = 105       # Strike price
T = 1         # Time to maturity in years
r = 0.05      # Risk-free interest rate
sigma = 0.2   # Volatility of the stock
M = 100       # Number of time steps
N = 10000     # Number of simulation paths

# Simulate stock price paths
np.random.seed(123)
dt = T / M
S = np.zeros((N, M+1))
S[:, 0] = S0

for i in range(1, M+1):
    Z = np.random.randn(N)
    S[:, i] = S[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Calculate 95% confidence intervals
S_mean = np.mean(S, axis=0)
S_ci_upper = np.quantile(S, 0.975, axis=0)
S_ci_lower = np.quantile(S, 0.025, axis=0)

# Monte Carlo option pricing
payoffs = np.maximum(S[:, -1] - K, 0)
discounted_payoffs = np.exp(-r * T) * payoffs
mc_option_price = np.mean(discounted_payoffs)

# Black-Scholes option pricing
d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
bs_option_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Print option prices
print(f"Monte Carlo Option Price: {mc_option_price:.2f}")
print(f"Black-Scholes Option Price: {bs_option_price:.2f}")

# Plot all paths
plt.figure(figsize=(10, 6))
#plt.plot(S.T, color='gray', alpha=0.1)
plt.plot(S_ci_upper, color='red', linewidth=2, label='95% CI Upper')
plt.plot(S_ci_lower, color='red', linewidth=2, label='95% CI Lower')

# Shade area between highest and lowest path
S_max = np.max(S, axis=0)
S_min = np.min(S, axis=0)
plt.fill_between(range(M+1), S_max, S_min, color='gray', alpha=0.5, label='Min-Max Range')

# Add Black-Scholes option price as a horizontal line
#plt.axhline(y=bs_option_price, color='blue', linestyle='--', linewidth=2, label='Black-Scholes Price')

# Add labels and legend
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.title('Monte Carlo Simulation of Stock Prices')
plt.legend()
plt.show()
