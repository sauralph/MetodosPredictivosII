mpNumCoEvents <- function(closeIdx, t1, molecule) {
  # Compute the number of concurrent events per bar.
  # +molecule[1] is the date of the first event on which the weight will be computed 
  # +molecule[length(molecule)] is the date of the last event on which the weight will be computed
  # Any event that starts before max(t1[molecule]) impacts the count.
  
  # 1) Find events that span the period [molecule[1], molecule[length(molecule)]]
  t1[is.na(t1)] <- tail(closeIdx, n = 1)  # unclosed events still must impact other weights
  t1 <- t1[t1 >= molecule[1]]  # events that end at or after molecule[1]
  t1 <- t1[1:max(t1[molecule])]  # events that start at or before max(t1[molecule])
  
  # 2) Count events spanning a bar
  iloc <- findInterval(c(t1[1], max(t1)), closeIdx)
  count <- rep(0, length(closeIdx[iloc[1]:(iloc[2] + 1)]))
  names(count) <- closeIdx[iloc[1]:(iloc[2] + 1)]
  
  for (i in seq_along(t1)) {
    tIn <- as.character(index(t1)[i])
    tOut <- as.character(t1[i])
    count[tIn:tOut] <- count[tIn:tOut] + 1
  }
  
  return(count[as.character(molecule[1]):as.character(max(t1[molecule]))])
}
