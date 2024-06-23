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

# Example usage
# Assuming close is a data frame with date and close columns, and events is a data frame with date, t1, trgt, and side columns
events <- applyPtSlOnT1(close, events, c(1, 1))

head(events)


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


