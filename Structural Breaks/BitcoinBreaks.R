library(quantmod)
library(strucchange)
library(urca)
library(changepoint)
library(WaveletComp)

# Obtain Bitcoin price data
getSymbols("BTC-USD", src = "yahoo", from = "2014-01-01", to = "2024-01-01")

# Extract adjusted closing prices and calculate daily returns
btc_prices <- Cl(`BTC-USD`)

png("fig11.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot.ts(btc_prices)
dev.off()

# CUSUM test
cusum_test <- efp(btc_prices ~ lag(btc_prices),type = "OLS-CUSUM")
png("fig12.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot(cusum_test, main = "CUSUM Test on Bitcoin Returns")
dev.off()

# Chow Test
breakpoint_index <- floor(length(btc_prices) / 2)
chow_test <- sctest(btc_prices ~ lag(btc_prices), type = "Chow", point = breakpoint_index)
chow_test_result <- chow_test$p.value
print(paste("Chow Test p-value:", chow_test_result))

# Bai-Perron
bp_test <- breakpoints(btc_prices ~ lag(btc_prices), breaks = 5)
summary(bp_test)
png("fig13.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot(bp_test, main = "Bai-Perron Test on Bitcoin Returns")
dev.off()
# Wavelet power
btc_df <- data.frame(date = index(btc_prices), returns = as.vector(btc_prices))
wavelet_analysis <- analyze.wavelet(btc_df, my.series = "returns", loess.span = 0)
png("fig14.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
wt.image(wavelet_analysis, 
         main = "Espectro de Potencia de las Retornos de Bitcoin",
         spec.time.axis = list(
           at = c(which(index(btc_prices)=="2016-07-09"),
                  which(index(btc_prices)=="2020-05-11"),
                  which(index(btc_prices)=="2023-04-23")
                  ), 
           labels = c("2016-07-09","2020-05-11","2023-04-23")))
dev.off()

mv_pelt <- cpt.meanvar(as.vector(btc_prices), method = "BinSeg",Q=5)
png("fig15.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
ts.plot(btc_prices, main = "Bitcoin con Puntos de Quiebre Detectados", ylab = "Precio (USD)", xlab = "Date")
abline(v = cpts(mv_pelt), col = "red", lty = 2)
abline(v=c(which(index(btc_prices)=="2016-07-09"),
           which(index(btc_prices)=="2020-05-11"),
           which(index(btc_prices)=="2023-04-23")),
       col="steelblue")
dev.off()

