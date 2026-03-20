# tesi-fkm-code

R code for the master's thesis:
**"Fuzzy K-Means Clustering and Deep Neural Networks for Tactical Asset Allocation"**
Lorenzo Berti — Università Cattolica del Sacro Cuore, 2025

---

## Repository structure

| File | Description |
|------|-------------|
| `UNSUPERVISED.R` | Unsupervised pipeline: data panel construction, momentum features, monthly adaptive PCA, FKM grid search (70 configurations), threshold-based trading strategy |
| `SUPERVISED_DNN_2.R` | DNN engine for a single FKM configuration: cluster-specific 3×64 architecture, 5-model ensemble, v8 trading rule |
| `SUPERVISED_DNN_3.R` | Batch runner: sequentially calls `SUPERVISED_DNN_2.R` over the 8 selected FKM configurations |
| `TC_ANALYSIS_FKM.R` | Transaction cost analysis: net return estimation at 0, 10, and 20 basis points (turnover-based) |

## Execution order

```
1. UNSUPERVISED.R        →  produces final_df.rds, refit_10configs_RS20.rds
2. SUPERVISED_DNN_2.R    →  (called by SUPERVISED_DNN_3.R)
3. SUPERVISED_DNN_3.R    →  runs DNN over the 8 selected configurations
4. TC_ANALYSIS_FKM.R     →  reads results, applies transaction cost scenarios
```

## Main dependencies

```r
install.packages(c(
  "fclust",     # Fuzzy K-Means (Palumbo, Iodice D'Enza, Vichi)
  "keras",      # Deep neural networks
  "tensorflow",
  "tidyverse",
  "PerformanceAnalytics",
  "xts",
  "lubridate"
))
```

## Data

Input data (price series and fundamentals) are not included in this repository.
The scripts expect the data to be available in the working directory as specified in each file's header.
