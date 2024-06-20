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

getTEvents <- function(gRaw, h) {
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


