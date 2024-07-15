---
marp: true
theme: gaia
math: katex
class: lead
---

# Understanding Overlapping Outcomes in Financial ML

## Key Concepts:
- **Labeling Financial Data**
  - Labels ($y$) are assigned to observed features ($X$) based on price bars over intervals $[t_{i,0}, t_{i,1}]$.
- **Overlapping Intervals**
  - Labels $y_i$ and $y_j$ depend on common returns if intervals overlap: $t_{i,1} > t_{j,0}$.

---

# Challenges and Solutions

## Challenges:
- **Non-IID Labels**
  - Overlapping intervals cause labels to be non-independent and identically distributed (non-IID).
- **Restricting Overlaps**
  - Eliminating overlaps by ensuring $t_{i,1} \leq t_{i+1,0}$ results in coarse models with limited sampling frequency.

---
 
## Solutions:
- **Correcting for Overlaps**
  - Design sampling and weighting schemes to adjust for the influence of overlapping outcomes.

---

- **Concurrent Labels:**
  - Two labels, $y_i$ and $y_j$, are concurrent at time $t$ if influenced by the same return, $r_{t-1,t} = p_t - p_{t-1}$.
  - Overlap does not need to be perfect.

---

# Binary Array and Counting Concurrent Labels

## Binary Array for Overlap:
- For each time point $t = 1, \ldots, T$, create a binary array $1_{t,i}$ for $i = 1, \ldots, I$.
- $1_{t,i} = 1$ if $[t_{i,0}, t_{i,1}]$ overlaps with $[t-1, t]$, otherwise $1_{t,i} = 0$.

## Counting Concurrent Labels:
- Sum the binary array: $c_t = \sum_{i=1}^{I} 1_{t,i}$.
- $c_t$ represents the number of labels concurrent at time $t$.


