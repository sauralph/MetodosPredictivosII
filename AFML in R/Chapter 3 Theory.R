library(dplyr)
library(arrow)

spy<-read_parquet("../Capitulo2/spy_future.parquet")

# Define threshold
epsilon <- 0.001

spy$Close
lead(spy$Close,10)

# Calculate returns
spy <- spy %>%
  mutate(return = (lead(Close, 10) / Close) - 1)

# Label based on fixed threshold
spy <- spy %>%
  mutate(label = case_when(
    return < -epsilon ~ -1,
    abs(return) <= epsilon ~ 0,
    return > epsilon ~ 1
  ))

head(spy)

table(spy$label)


# Seccion 2.3 -------------------------------------------------------------

library(dplyr)
library(tibbletime)

# Function to calculate daily volatility
getDailyVol <- function(close, span0=100) {
  daily_returns <- close / lag(close, 1) - 1
  vol <- rollapply(daily_returns, width = span0, FUN = sd, 
                   fill = NA, align = "right")
  return(vol)
}

# Create Daily Bars:
spy <- spy %>%
  mutate(date = as.Date(date))

# Group by date and summarize
daily_summary <- spy %>%
  group_by(date) %>%
  summarize(
    daily_open = first(Open),
    daily_high = max(High),
    daily_low = min(Low),
    daily_close = last(Close),
    total_volume = sum(Volume)  
)


# Assuming `data` is a data frame with date and close columns
daily_summary <- daily_summary %>%
  mutate(volatility = getDailyVol(daily_close,span0=2))

head(daily_summary)


# Triple Barrier Method ---------------------------------------------------

library(dplyr)

applyPtSlOnT1 <- function(close, events, ptSl) {
  events <- events %>% mutate(t1 = ifelse(is.na(t1), max(close$date), t1))
  result <- events
  
  for (i in 1:nrow(events)) {
    loc <- events$date[i]
    t1 <- events$t1[i]
    trgt <- events$trgt[i]
    path <- close %>% filter(date >= loc & date <= t1)
    path_returns <- (path$close / close$close[which(close$date == loc)] - 1) * events$side[i]
    
    sl <- -ptSl[2] * trgt
    pt <- ptSl[1] * trgt
    
    result$sl[i] <- min(path$date[path_returns < sl], na.rm = TRUE)
    result$pt[i] <- min(path$date[path_returns > pt], na.rm = TRUE)
  }
  return(result)
}

# Create a sample dataset
set.seed(37)
sample_dates <- seq.POSIXt(from = as.POSIXct("2024-06-01 09:30"), to = as.POSIXct("2024-06-10 16:00"), by = "min")
n <- length(sample_dates)

sample_data <- tibble(
  date = sample_dates,
  close = cumsum(rnorm(n, 0, 0.5)) + 100,
  Open = cumsum(rnorm(n, 0, 0.5)) + 100,
  High = cumsum(rnorm(n, 0, 0.5)) + 100,
  Low = cumsum(rnorm(n, 0, 0.5)) + 100,
  Volume = sample(1:1000, n, replace = TRUE)
)

# Sample events data
sample_events <- tibble(
  date = sample(sample_dates, 2),
  t1 = sample(sample_dates, 2),
  trgt = runif(2, 0.1, 0.5),
  side = sample(c(-1, 1), 2, replace = TRUE)
)

# Print sample data
print(sample_data)
print(sample_events)

result <- applyPtSlOnT1(sample_data, sample_events, c(1, 1))

# Plot using base plot
plot(sample_data$date, sample_data$close, type = "l", 
     col = "steelblue", xlab = "Date", ylab = "Close Price", 
     main = "Close Prices with Event Points",
     frame.plot = F)

# Adding event points
points(result$date[1], 
       sample_data$close[match(result$date, sample_data$date)][1], 
       col = "indianred", pch = 16)

target=.1
segments(
  x0=result$date,
  x1=result$date,
  y0=sample_data$close[match(result$date[1],sample_data$date)] * (1 + target),
  y1=sample_data$close[match(result$date[1],sample_data$date)] * (1 - target),
  col="indianred",
  lwd=2,
  lty=c("solid","dashed")
)

tp = sample_data$close[match(result$date[1],sample_data$date)] * (1 + target)
sl = sample_data$close[match(result$date[1],sample_data$date)] * (1 - target)
bl = sample_data$close[match(result$date[1],sample_data$date)]
start_date = result$date[1]
end_date = result$date[2]

segments(
  x0=rep(start_date,3),
  x1=rep(end_date,3),
  y0=c(tp,sl,bl),
  y1=c(tp,sl,bl),
  col="indianred",
  lwd=2,
  lty=c("dashed")
         )

abline(v=result$date,col="indianred",lty="dashed")


points(result$sl, sample_data$close[match(result$sl, sample_data$date)], col = "orange", pch = 4)
points(result$pt, sample_data$close[match(result$pt, sample_data$date)], col = "green", pch = 3)

legend("topright", legend = c("Event", "Stop-Loss", "Profit-Taking"), col = c("red", "orange", "green"), pch = c(16, 4, 3))

#events <- applyPtSlOnT1(close, events, c(1, 1))

#head(events)


# Learning Side and Size --------------------------------------------------

library(dplyr)

getEvents <- function(close, tEvents, ptSl, trgt, minRet) {
  trgt <- trgt[trgt > minRet]
  side <- rep(1, length(trgt))
  events <- data.frame(t1 = t1, trgt = trgt, side = side)
  
  events <- applyPtSlOnT1(close, events, c(ptSl, ptSl))
  events <- events %>% mutate(t1 = pmin(sl, pt, na.rm = TRUE))
  
  return(events)
}

# Example usage
# Assuming close, tEvents, trgt, and minRet are defined
events <- getEvents(close, tEvents, 1, trgt, 0.01)

head(events)


# Meta-labeling -----------------------------------------------------------

library(dplyr)

getBins <- function(events, close) {
  events <- events %>% drop_na(t1)
  px <- close %>% filter(date %in% c(events$date, events$t1))
  px <- px %>% arrange(date)
  
  events <- events %>%
    rowwise() %>%
    mutate(
      ret = close[date == t1] / close[date == date] - 1,
      bin = sign(ret)
    ) %>%
    ungroup()
  
  return(events)
}

# Example usage
# Assuming events and close are defined
bins <- getBins(events, close)

head(bins)


# Droping Labels  ---------------------------------------------------------

library(dplyr)

dropLabels <- function(events, minPct = 0.05) {
  while(TRUE) {
    label_counts <- events %>%
      count(bin) %>%
      mutate(pct = n / sum(n))
    
    if (min(label_counts$pct) > minPct | nrow(label_counts) < 3) break
    
    min_label <- label_counts %>% filter(pct == min(pct)) %>% pull(bin)
    events <- events %>% filter(bin != min_label)
  }
  
  return(events)
}

# Example usage
# Assuming events is defined
filtered_events <- dropLabels(events, 0.05)

head(filtered_events)


