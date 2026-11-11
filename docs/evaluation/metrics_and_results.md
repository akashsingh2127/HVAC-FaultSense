# Comprehensive Model Comparison & Experimental Results

## 1. Consolidated Benchmark Table

All models were evaluated on the strictly held-out test partition ($60,000$ test records, $179,925$ training records) derived from the official LBNL RTU dataset with zero target leakage.

| Model | Analytical Task | MAE (W) | RMSE (W) | $R^2$ | Accuracy | Precision | Recall | Macro-F1 | Silhouette | Davies-Bouldin | Calinski-Harabasz | Training Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Mean Baseline** | Regression | 794.2322 | 944.5896 | -0.0056 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.000 |
| **OLS Regression (Scratch)** | Regression | 92.2755 | 119.3139 | 0.9840 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 10.293 |
| **Ridge Regression (Scratch)** | Regression | 156.4354 | 187.7189 | 0.9603 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 3.200 |
| **Linear Regression (Reference)** | Regression | **72.6888** | **102.2924** | **0.9882** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.150 |
| **K-Means (Scratch)** | Clustering | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **0.4966** | **0.7271** | **4967.2** | 3.206 |
| **GMM / EM (Scratch)** | Clustering | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.4644 | 0.7380 | 3764.5 | 6.217 |
| **Majority Class Baseline** | Classification | N/A | N/A | N/A | 0.2000 | 0.0286 | 0.1429 | 0.0476 | N/A | N/A | N/A | 0.107 |
| **CART Decision Tree (Scratch)** | Classification | N/A | N/A | N/A | **0.7186** | **0.6766** | **0.5953** | **0.5830** | N/A | N/A | N/A | 7.771 |
| **AdaBoost SAMME (Scratch)** | Classification | N/A | N/A | N/A | 0.6436 | 0.4990 | 0.5167 | 0.4908 | N/A | N/A | N/A | 50.189 |
| **CART Decision Tree (Reference)** | Classification | N/A | N/A | N/A | 0.6725 | 0.7279 | 0.5695 | 0.5346 | N/A | N/A | N/A | 3.550 |
| **AdaBoost (Reference)** | Classification | N/A | N/A | N/A | 0.6196 | 0.4708 | 0.4953 | 0.4745 | N/A | N/A | N/A | 18.660 |
| **CART (Unseen Severity Holdout)** | Classification | N/A | N/A | N/A | 0.5777 | 0.6751 | 0.4622 | 0.4954 | N/A | N/A | N/A | 0.000 |

## 2. Best-Performing Model per Analytical Task

### Task 1: Continuous Power Baseline Regression
- **Winner:** **Linear Regression (Reference)** (RMSE = $102.29$ W, $R^2 = 0.9882$) closely followed by **OLS Regression (Scratch)** (RMSE = $119.31$ W, $R^2 = 0.9840$).
- **Why it won:** Predicted strictly from driving boundary conditions (weather, zone load, staging status) without tautological component power leakage. Captures physical cooling power demand with $\sim 3.5\%$ relative error on a machine drawing $2,000 - 5,000$ W.

### Task 2: Unsupervised Operating Regime Discovery
- **Winner:** **K-Means (Scratch)** (Silhouette = **0.4966**, Davies-Bouldin = **0.7271**, Calinski-Harabasz = **4967.2**).
- **Why it won:** Effectively partitions continuous operational telemetry into cohesive clusters corresponding to cooling capacity operating modes.

### Task 3: Supervised Multi-Class Fault Diagnosis
- **Winner:** **CART Decision Tree (Scratch)** (Accuracy = **71.86%**, Macro-F1 = **0.5830**).
- **Why it won:** CART identified hierarchical threshold rules directly aligned with physical thermodynamic limits (e.g. condensing pressure $> 2100$ kPa indicates condenser fouling, while low suction pressure indicates refrigerant undercharge). It significantly surpassed the majority class baseline (Macro-F1 $0.0476$) and outperformed the scikit-learn reference CART (Macro-F1 $0.5346$) and AdaBoost ($0.4908$). On completely unseen fault severities (Protocol B), CART maintained a solid **0.4954** Macro-F1.
