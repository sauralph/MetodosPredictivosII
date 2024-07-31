import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt

stocks = [
    "AGRO.BA", "ALUA.BA", "AUSO.BA", "BBAR.BA", "BHIP.BA", "BMA.BA", "BPAT.BA",
    "BRIO.BA", "SUPV.BA", "BOLT.BA", "BYMA.BA", "CVH.BA", "CGPA2.BA", "CAPX.BA",
    "CADO.BA", "CELU.BA", "CECO2.BA", "CEPU.BA", "COME.BA", "INTR.BA", "CTIO.BA",
    "CRES.BA", "DOME.BA", "DYCA.BA", "EDN.BA", "FERR.BA", "FIPL.BA", "GARO.BA",
    "DGCU2.BA", "GBAN.BA", "GGAL.BA", "OEST.BA", "GRIM.BA", "VALO.BA", "HAVA.BA",
    "HARG.BA", "INAG.BA", "INVJ.BA", "IRSA.BA", "SEMI.BA", "LEDE.BA", "LOMA.BA",
    "LONG.BA", "METR.BA", "MIRG.BA", "MOLI.BA", "MORI.BA", "PAMP.BA", "PATA.BA",
    "POLL.BA", "RIGO.BA", "ROSE.BA", "SAMI.BA", "TECO2.BA", "TXAR.BA",
    "TRAN.BA", "TGNO4.BA", "YPFD.BA",
    "BTC-USD", "ETH-USD"
]

start_date = "2011-01-02"
end_date = datetime.now().strftime('%Y-%m-%d')

# Fetch historical stock data
data = yf.download(stocks, start=start_date, end=end_date)

data.interpolate(method='linear', inplace=True)

# Split the data into training (pre-2024) and testing (2024) sets
training_data = data[:'2023-12-31']
testing_data = data['2024-01-01':]

returns_train = training_data['Adj Close'].pct_change()
returns_test = testing_data['Adj Close'].pct_change()

nStocks = len(stocks)
R_train = returns_train.mean()
S_train = returns_train.cov()

R_test = returns_test.mean()
S_test = returns_test.cov()

def weights(w):
    w = np.clip(w, 0, 1)
    return w / sum(w)

def portfolio_return(w, R):
    return sum(w * R)

def portfolio_volatility(w, S):
    return np.dot(w.T, np.dot(S, w))

# Generate random portfolios
n_portfolios = 100000
random_portfolios = []

for _ in range(n_portfolios):
    w = np.random.rand(nStocks)
    w = weights(w)
    ret = portfolio_return(w, R_train)
    vol = portfolio_volatility(w, S_train)
    random_portfolios.append((w, ret, vol))

# Select the top 10 portfolios based on return/volatility ratio
top_10_portfolios = sorted(random_portfolios, key=lambda x: x[1]/x[2], reverse=True)[:10]

# Test the top 10 optimal portfolios with testing data
results = []
for w, ret, vol in top_10_portfolios:
    ret_test = portfolio_return(w, R_test)
    vol_test = portfolio_volatility(w, S_test)
    results.append((w, ret_test, vol_test))

# Function to print non-zero weights and their corresponding symbols
def print_non_zero_weights(results):
    for i, (w, ret, vol) in enumerate(results):
        print(f"Portfolio {i+1}:")
        non_zero_indices = np.where(w > 0)[0]
        for idx in non_zero_indices:
            print(f"  {stocks[idx]}: {w[idx]:.4f}")
        print(f"  Expected return: {ret}")
        print(f"  Expected volatility: {vol}")
        print()

# Print the results
print_non_zero_weights(results)

# Extract returns and volatilities for all portfolios
all_returns = [x[1] for x in random_portfolios]
all_volatilities = [x[2] for x in random_portfolios]

risk_free_rate = 0.02/365  # Daily risk-free rate

# Calculate the Sharpe ratio for the optimal portfolio
best_portfolio = top_10_portfolios[0]
best_return = best_portfolio[1]
best_volatility = best_portfolio[2]
sharpe_ratio = (best_return - risk_free_rate) / best_volatility

# Define the Capital Market Line (CML)
cml_x = np.linspace(min(all_volatilities), max(all_volatilities)/2, 100)
cml_y = risk_free_rate + sharpe_ratio * cml_x

# Plot all portfolios in a risk vs return scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(all_volatilities, all_returns, c='blue', marker='o', s=10, alpha=0.5, label='Portfolios Aleatorios')

# Highlight the top 10 portfolios
top_returns = [x[1] for x in top_10_portfolios]
top_volatilities = [x[2] for x in top_10_portfolios]
plt.scatter(top_volatilities, top_returns, c='red', marker='x', s=50, label='Mejores 10 Portfolios')
plt.plot(cml_x, cml_y, color='green', label='Linea de Mercado de Capitales (CML)')

plt.title('Riesgo vs Retorno de Portfolios')
plt.xlabel('Volatilidad (Riesgo)')
plt.ylabel('Retorno')
plt.legend()
plt.grid(True)
plt.show()
