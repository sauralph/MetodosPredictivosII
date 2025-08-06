import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Fetch historical stock data for YPFD.BA
stock = "BTC-USD"
start_date = "2011-01-02"
end_date = "2024-07-31"

data = yf.download(stock, start=start_date, end=end_date)['Close']
data.interpolate(method='linear', inplace=True)

# Normalize the prices
normalized_data = data / data.mean()

# Perform Continuous Wavelet Transform (CWT)
wavelet = 'cmor'  # Complex Morlet wavelet
scales = np.arange(1, 128)
coeffs, frequencies = pywt.cwt(normalized_data, scales, wavelet)

# Compute the wavelet power spectrum
power_spectrum = (abs(coeffs)) ** 2

# Plot the wavelet power spectrum
plt.figure(figsize=(14, 7))
plt.contourf(normalized_data.index, scales, power_spectrum, extend='both', cmap='viridis')
plt.title('Wavelet Power Spectrum of YPFD.BA')
plt.xlabel('Date')
plt.ylabel('Scale')
plt.colorbar(label='Power')
plt.show()
