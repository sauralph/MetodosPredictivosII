# Funciones del capitulo 2:
library(data.table)
library(lubridate)
library(readr)
library(dplyr)
library(arrow)
library(zoo)
library(ggplot2)

# Snippet 2.1 -------------------------------------------------------------


# The pcaWeights function computes the weights for a portfolio based on 
# Principal Component Analysis (PCA). 

pcaWeights <- function(cov, riskDist = NULL, riskTarget = 1) {
  # Following the riskAlloc distribution, match riskTarget
  eig <- eigen(cov)
  eVal <- eig$values
  eVec <- eig$vectors
  
  indices <- order(eVal, decreasing = TRUE)
  eVal <- eVal[indices]
  eVec <- eVec[, indices]
  
  if (is.null(riskDist)) {
    riskDist <- rep(0, nrow(cov))
    riskDist[length(riskDist)] <- 1
  }
  
  loads <- riskTarget * (riskDist / eVal)^0.5
  wghts <- eVec %*% matrix(loads, ncol = 1)
  
  # ctr <- (loads / riskTarget)^2 * eVal # verify riskDist
  return(wghts)
}


# Snippet 2.2 -------------------------------------------------------------

getRolledSeries <- function(pathIn, key) {
  series <- fread(pathIn)
  series[, Time := ymd_hms(Time)] # same as:series$Time <- ymd_hms(series$Time)
  setkey(series, Time) # for fast sorting
  
  gaps <- rollGaps(series)
  
  for (fld in c("Close", "VWAP")) {
    series[[fld]] <- series[[fld]] - gaps
  }
  
  return(series)
}

rollGaps <- function(series, dictio = list(Instrument = 'FUT_CUR_GEN_TICKER', Open = 'PX_OPEN', Close = 'PX_LAST'), matchEnd = TRUE) {
  # Compute gaps at each roll, between previous close and next open
  rollDates <- unique(series[[dictio$Instrument]])
  gaps <- rep(0, nrow(series))
  iloc <- which(series$Time %in% rollDates)
  iloc <- iloc - 1 # index of days prior to roll
  
  gaps[iloc[-1]] <- series[[dictio$Open]][iloc[-1]] - series[[dictio$Close]][iloc[-1]]
  gaps <- cumsum(gaps)
  
  if (matchEnd) {
    gaps <- gaps - tail(gaps, 1)
  }
  
  return(gaps)
}



# Snippet 2.3 -------------------------------------------------------------

# Step 1: Read CSV File
# filePath <- "your_file_path.csv"  # Update with your file path
# raw <- read_csv(filePath, col_types = cols())
raw <- read_parquet("../Capitulo2/spy_future.parquet")
colnames(raw)[6]<-"Time"
raw <- raw %>% mutate(Time = ymd_hms(Time))
raw <- raw %>% arrange(Time)
raw$Symbol <- "SPY"

gaps <- rollGaps(raw, dictio = list(Instrument = 'Symbol', Open = 'Open', Close = 'Close'))

# Step 3: Copy the DataFrame
rolled <- raw

# Step 4: Adjust Prices for Roll Gaps
for (fld in c("Open", "Close")) {
  rolled[[fld]] <- rolled[[fld]] - gaps
}

# Step 5: Calculate Returns
rolled <- rolled %>%
  mutate(Returns = (Close - lag(Close)) / lag(Close))

rolled$Returns[is.na(rolled$Returns)]<-0

# Step 6: Calculate Adjusted Prices
rolled <- rolled %>%
  mutate(rPrices = cumprod(1 + Returns))

plot(rolled$Time,rolled$rPrices,type="l")


# Snippet 2.4 -------------------------------------------------------------

getTEvents <- function(gRaw, h=0.01) {
  tEvents <- c()
  sPos <- 0
  sNeg <- 0
  diff <- diff(gRaw)
  
  for (i in 2:length(diff)) {
    sPos <- max(0, sPos + diff[i])
    sNeg <- min(0, sNeg + diff[i])
    if (sNeg < -h) {
      sNeg <- 0
      tEvents <- c(tEvents, index(gRaw)[i])
    } else if (sPos > h) {
      sPos <- 0
      tEvents <- c(tEvents, index(gRaw)[i])
    }
  }
  return(as.POSIXct(tEvents))
}


# Example -----------------------------------------------------------------

# Use getTEvents function
returns_zoo <- zoo(rolled$Returns, order.by = rolled$Time)
h <- 0.01 # Set your threshold value
tEvents <- getTEvents(returns_zoo, h)

# Visualize the adjusted prices
plot(rolled$Time, rolled$rPrices, type = "l", col = "blue", 
     main = "Precio Ajustado con eventos t", 
     xlab = "Fecha", ylab = "Precio Ajustado")
abline(v = tEvents, col = "red", lwd = 2, lty = 2)


# Excersises --------------------------------------------------------------


# 2.1 ---------------------------------------------------------------------


# Section a ---------------------------------------------------------------

# Function to create tick bars
create_tick_bars <- function(data, tick_size = 500) {
  data$bar_id <- rep(1:ceiling(nrow(data) / tick_size), each = tick_size)[1:nrow(data)]
  tick_bars <- data %>%
    group_by(bar_id) %>%
    summarise(Time = first(Time),
              Open = first(Open),
              Close = last(Close),
              Volume = sum(Volume),
              VWAP = sum(Close * Volume) / sum(Volume),
              .groups = 'drop')
  return(tick_bars)
}

# Function to create volume bars
create_volume_bars <- function(data, volume_size = 1000) {
  data$vol_cumsum <- cumsum(data$Volume)
  volume_bars <- data %>%
    group_by(bar_id = ceiling(vol_cumsum / volume_size)) %>%
    summarise(Time = first(Time),
              Open = first(Open),
              Close = last(Close),
              Volume = sum(Volume),
              VWAP = sum(Close * Volume) / sum(Volume),
              .groups = 'drop')
  return(volume_bars)
}

# Function to create dollar bars
create_dollar_bars <- function(data, dollar_size = 100000) {
  data$dollar_cumsum <- cumsum(data$Volume * data$VWAP)
  dollar_bars <- data %>%
    group_by(bar_id = ceiling(dollar_cumsum / dollar_size)) %>%
    summarise(Time = first(Time),
              Open = first(Open),
              Close = last(Close),
              Volume = sum(Volume),
              VWAP = sum(Close * Volume) / sum(Volume),
              .groups = 'drop')
  return(dollar_bars)
}

raw <- raw %>%
  mutate(VWAP = cumsum(Close * Volume) / cumsum(Volume))


# Example usage
tick_bars <- create_tick_bars(raw)
volume_bars <- create_volume_bars(raw)
dollar_bars <- create_dollar_bars(raw)

# Plotting functions
plot_bars <- function(bars, title) {
  plot(as.POSIXct(bars$Time), bars$Close, type = "l", col = "blue", main = title, xlab = "Time", ylab = "Close Price")
}


# Plot tick bars
plot_bars(tick_bars, "Tick Bars Close Prices")

# Plot volume bars
plot_bars(volume_bars, "Volume Bars Close Prices")

# Plot dollar bars
plot_bars(dollar_bars, "Dollar Bars Close Prices")


# Section b ---------------------------------------------------------------

# Function to count weekly bars
count_weekly_bars <- function(bars) {
  bars$week <- floor_date(as.POSIXct(bars$Time), "week")
  weekly_counts <- bars %>%
    group_by(week) %>%
    summarise(Count = n(), .groups = 'drop')
  return(weekly_counts)
}

# Count weekly bars
tick_weekly_counts <- count_weekly_bars(tick_bars)
volume_weekly_counts <- count_weekly_bars(volume_bars)
dollar_weekly_counts <- count_weekly_bars(dollar_bars)

# Plot weekly bar counts
plot_weekly_counts <- function(weekly_counts, title) {
  ggplot(weekly_counts, aes(x = week, y = Count)) +
    geom_line() +
    labs(title = title, x = "Time", y = "Weekly Bar Count") +
    theme_minimal()
}

plot_weekly_counts(tick_weekly_counts, "Tick Bars Weekly Count")
plot_weekly_counts(volume_weekly_counts, "Volume Bars Weekly Count")
plot_weekly_counts(dollar_weekly_counts, "Dollar Bars Weekly Count")


# section c ---------------------------------------------------------------

# Function to compute serial correlation
compute_serial_correlation <- function(bars) {
  returns <- diff(log(bars$Close))
  return(cor(returns[-1], returns[-length(returns)]))
}

tick_serial_corr <- compute_serial_correlation(tick_bars)
volume_serial_corr <- compute_serial_correlation(volume_bars)
dollar_serial_corr <- compute_serial_correlation(dollar_bars)

tick_serial_corr
volume_serial_corr
dollar_serial_corr

# section d ---------------------------------------------------------------

# Function to partition bars into monthly subsets and compute variance of returns
compute_variance_of_variances <- function(bars) {
  bars$Month <- floor_date(as.POSIXct(bars$Time), "month")
  monthly_variances <- bars %>%
    group_by(Month) %>%
    summarise(Variance = var(diff(log(Close))), .groups = 'drop')
  return(var(monthly_variances$Variance))
}

tick_var_of_vars <- compute_variance_of_variances(tick_bars)
volume_var_of_vars <- compute_variance_of_variances(volume_bars)
dollar_var_of_vars <- compute_variance_of_variances(dollar_bars)

tick_var_of_vars
volume_var_of_vars
dollar_var_of_vars


# section e ---------------------------------------------------------------

library(tseries)

# Function to apply Jarque-Bera test
apply_jarque_bera_test <- function(bars) {
  returns <- diff(log(bars$Close))
  jb_test <- jarque.bera.test(returns)
  print(jb_test)
  return(jb_test$statistic)
}

tick_jb_stat <- apply_jarque_bera_test(tick_bars)
volume_jb_stat <- apply_jarque_bera_test(volume_bars)
dollar_jb_stat <- apply_jarque_bera_test(dollar_bars)

tick_jb_stat
volume_jb_stat
dollar_jb_stat

returns <- diff(log(tick_bars$Close))
hist(returns)
qqnorm(returns)
qqline(returns)

write_parquet(tick_bars,"../Capitulo4/spy_future_tickbars.parquet")

# 2.2 ---------------------------------------------------------------------

# Function to determine BuySell using the tick rule
determine_buy_sell <- function(prices) {
  buy_sell <- numeric(length(prices))
  buy_sell[1] <- 1  # Assume first trade is a buy
  
  for (i in 2:length(prices)) {
    if (prices[i] > prices[i - 1]) {
      buy_sell[i] <- 1
    } else if (prices[i] < prices[i - 1]) {
      buy_sell[i] <- -1
    } else {
      buy_sell[i] <- buy_sell[i - 1]
    }
  }
  
  return(buy_sell)
}

# Function to compute tick dollar imbalances
compute_tick_dollar_imbalance <- function(data, window_size = 50) {
  data <- data %>%
    arrange(Time) %>%
    mutate(BuySell = determine_buy_sell(Close),
           BuyVolume = ifelse(BuySell == 1, Volume * Close, 0),
           SellVolume = ifelse(BuySell == -1, Volume * Close, 0))
  
  # Compute rolling sums
  data <- data %>%
    mutate(RollingBuyVolume = rollapply(BuyVolume, width = window_size, FUN = sum, align = "right", fill = NA),
           RollingSellVolume = rollapply(SellVolume, width = window_size, FUN = sum, align = "right", fill = NA))
  
  # Compute tick dollar imbalance
  data <- data %>%
    mutate(TickDollarImbalance = RollingBuyVolume - RollingSellVolume)
  
  return(data)
}

compute_serial_correlation(dollar_bars)
dollar_imbalance_bars<-compute_tick_dollar_imbalance(raw)
dollar_imbalance_bars_corr<-compute_serial_correlation(dollar_imbalance_bars)

barplot(
  c(tick_serial_corr, volume_serial_corr, dollar_serial_corr,dollar_imbalance_bars_corr),
  names.arg = c("Tick Bars", "Volume Bars", "Dollar Bars","Dollar Imbalance"),
  main = "Serial Correlation of Returns",
  ylab = "Serial Correlation",
  col = c("gray", "gray", "steelblue1","steelblue"),
  border = F
)


# Excersise 2.3 -----------------------------------------------------------


# Section (a) -------------------------------------------------------------

eurostoxx50<-read_parquet("../Capitulo2/eurostoxx50_future.parquet")
ex_rates<-read_parquet("../Capitulo2/eur_to_usd.parquet")
tail(ex_rates)
tail(eurostoxx50)

# Convert Datetime columns to POSIXct if not already in that format
eurostoxx50 <- eurostoxx50 %>%
  mutate(Time = as.POSIXct(Datetime, format = "%Y-%m-%d %H:%M:%S"))

eurostoxx50 <- eurostoxx50 %>%
  mutate(Date = as.Date(Datetime))

ex_rates <- ex_rates %>%
  mutate(Date = as.Date(as.POSIXct(Date, format = "%Y-%m-%d %H:%M:%S")))

# Merge the data frames on the datetime column
merged_data <- eurostoxx50 %>%
  left_join(ex_rates, by = "Date")

merged_data <- merged_data %>%
  filter(!is.na(EURUSD))

converted_data <- merged_data %>%
  mutate(Open_USD = Open * EURUSD,
         High_USD = High * EURUSD,
         Low_USD = Low * EURUSD,
         Close_USD = Close * EURUSD)

converted_data <- converted_data %>%
  select(Datetime, Open_USD, High_USD, Low_USD, Close_USD, Volume)

colnames(converted_data)[1:5]<-c("Time","Open","High","Low","Close")

converted_data<-converted_data %>%
  mutate(VWAP = cumsum(Close * Volume) / cumsum(Volume))

eurostoxx50_dollar_bars<-create_dollar_bars(converted_data)

# Function to adjust futures prices using ETF trick
compute_omega_t <- function(futures_sp500, futures_eurostoxx50, roll_dates) {
  omega_t <- numeric(length(roll_dates))
  for (i in seq_along(roll_dates)) {
    roll_date <- roll_dates[i]
    
    # Find the nearest dates in SP500 futures data
    sp500_before_roll <- futures_sp500 %>% filter(Time <= roll_date) %>% tail(1)
    sp500_after_roll <- futures_sp500 %>% filter(Time > roll_date) %>% head(1)
    
    # Find the nearest dates in Eurostoxx 50 futures data
    eurostoxx50_before_roll <- futures_eurostoxx50 %>% filter(Time <= roll_date) %>% tail(1)
    eurostoxx50_after_roll <- futures_eurostoxx50 %>% filter(Time > roll_date) %>% head(1)
    
    # Find the nearest dates in ETF data
    etf_before_roll <- eurostoxx50_before_roll
    etf_after_roll <- eurostoxx50_after_roll
    
    # Calculate the adjustment factor
    adj_factor <- (etf_after_roll$Close / etf_before_roll$Close) /
      (sp500_after_roll$Close / sp500_before_roll$Close)
    
    omega_t[i] <- adj_factor
  }
  return(omega_t)
}

euro_returns<-eurostoxx50 %>%
  mutate(Returns = (Close - lag(Close)) / lag(Close))


euro_returns_zoo <- zoo(euro_returns$Returns, order.by = euro_returns$Time)

roll_dates_for_eurostoxx50<-getTEvents(euro_returns_zoo,h = 0.005)
length(roll_dates_for_eurostoxx50)
# Compute \(\hat{\omega}_t\) vector
omega_t <- compute_omega_t(dollar_bars, eurostoxx50_dollar_bars, 
                           roll_dates_for_eurostoxx50)

# Print \(\hat{\omega}_t\) vector
print(omega_t)


# Section (b) -------------------------------------------------------------

# Align the time series by Datetime
aligned_data <- dollar_bars %>%
  inner_join(eurostoxx50_dollar_bars, by = "Time", suffix = c("_sp500", "_eurostoxx"))

aligned_data <- aligned_data %>%
  mutate(Spread = Close_sp500 - Close_eurostoxx)

aligned_data$Spread
plot(aligned_data$Time, aligned_data$Spread, type = "l", col = "blue",
     main = "S&P 500/Eurostoxx 50 Spread", xlab = "Datetime", ylab = "Spread (USD)")


# Section (c) -------------------------------------------------------------

adf_test_result <- adf.test(aligned_data$Spread, alternative = "stationary")
print(adf_test_result)


# Section 2.4 -------------------------------------------------------------

# Calculate rolling mean and Bollinger Bands
window_size <- 50  # Set the rolling window size

dollar_bars <- dollar_bars %>%
  arrange(Time) %>%
  mutate(
    rolling_mean = rollmean(Close, window_size, fill = NA, align = "right"),
    upper_band = rolling_mean * 1.05,
    lower_band = rolling_mean * 0.95
  )

# Count the crossings
dollar_bars <- dollar_bars %>%
  mutate(
    above_upper = ifelse(lag(Close) <= lag(upper_band) & Close > upper_band, 1, 0),
    below_lower = ifelse(lag(Close) >= lag(lower_band) & Close < lower_band, 1, 0)
  )

crossings <- sum(dollar_bars$above_upper, na.rm = TRUE) + sum(dollar_bars$below_lower, na.rm = TRUE)

# Print the number of crossings
print(paste("Number of crossings: ", crossings))

# Plot the series and Bollinger Bands
plot(dollar_bars$Time, dollar_bars$Close, type = "l", col = "gray", 
     main = "Close Prices with Bollinger Bands", xlab = "Time", ylab = "Price",
     ylim = range(c(dollar_bars$upper_band,dollar_bars$lower_band),na.rm = T),
     frame.plot = F)
lines(dollar_bars$Time, dollar_bars$rolling_mean, col = "aquamarine4")
lines(dollar_bars$Time, dollar_bars$upper_band, col = "indianred", lty="dashed")
lines(dollar_bars$Time, dollar_bars$lower_band, col = "indianred",lty="dashed")
legend("topright", legend = c("Close", "Rolling Mean", "Upper Band", "Lower Band"),
       col = c("gray", "aquamarine4", "indianred", "indianred"),
       border = NA,lty=c("solid","solid","dashed","dashed"))


# Section (b) -------------------------------------------------------------

cusum_events<-getTEvents(returns_zoo,h = .05)
abline(v=cusum_events,col="indianred",lty="dashed")


