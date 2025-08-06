import yfinance as yf
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# List of stocks
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
data = yf.download(stocks, start=start_date, end=end_date)['Close']
data.interpolate(method='linear', inplace=True)

# Normalize the prices
normalized_data = data.div(data.mean())

# Compute the DTW distance matrix
n = len(stocks)
distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i+1, n):
        # Cast the normalized data to arrays with enumerated indices
        asset_i_array = np.array(list(enumerate(normalized_data.iloc[:, i], start=1)))
        asset_j_array = np.array(list(enumerate(normalized_data.iloc[:, j], start=1)))
        # Perform DTW
        try:
            distance, _ = fastdtw(asset_i_array, asset_j_array, dist=euclidean)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
        except Exception as e:
            print(f"An error occurred: {e}")
            distance_matrix[i, j] = np.nan
            distance_matrix[j, i] = np.nan
        print("Processed pair ({}, {})".format(stocks[i], stocks[j]))

# Convert the distance matrix to a DataFrame
distance_df = pd.DataFrame(distance_matrix, index=stocks, columns=stocks)


# Output the distance matrix as a markdown table
markdown_table = distance_df.iloc[0:9,0:5].to_markdown()
print(markdown_table)


import seaborn as sns
import matplotlib.pyplot as plt

# Plot the heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(distance_df, cmap='viridis', annot=False, xticklabels=True, yticklabels=True)
plt.title('DTW Distance Matrix Heatmap')
plt.xlabel('Assets')
plt.ylabel('Assets')
plt.show()


