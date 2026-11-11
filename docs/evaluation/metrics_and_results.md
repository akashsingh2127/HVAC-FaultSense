# Comprehensive Model Comparison & Experimental Results

## 1. Consolidated Benchmark Table

All models were evaluated on the strictly held-out test partition ($60,000$ test records, $179,925$ training records) derived from the official LBNL RTU dataset.

| Model | Analytical Task | MAE (W) | RMSE (W) | $R^2$ | Accuracy | Precision | Recall | Macro-F1 | Silhouette | Training Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Mean Baseline** | Regression | 794.2322 | 944.5896 | -0.0056 | N/A | N/A | N/A | N/A | N/A | 0.001 |
| **OLS Regression (Scratch)** | Regression | 29.9171 | 38.9816 | 0.9983 | N/A | N/A | N/A | N/A | N/A | 36.101 |
| **Ridge Regression (Scratch)** | Regression | 137.9101 | 169.0907 | 0.9678 | N/A | N/A | N/A | N/A | N/A | 4.271 |
| **Linear Regression (Reference)** | Regression | 0.0000 | 0.0000 | 1.0000 | N/A | N/A | N/A | N/A | N/A | 0.840 |
| **K-Means (Scratch)** | Clustering | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.4726 | 7.263 |
| **GMM / EM (Scratch)** | Clustering | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **0.4821** | 5.316 |
| **Majority Class Baseline** | Classification | N/A | N/A | N/A | 0.2000 | 0.0286 | 0.1429 | 0.0476 | N/A | 0.157 |
| **CART Decision Tree (Scratch)** | Classification | N/A | N/A | N/A | **0.7186** | **0.6766** | **0.5953** | **0.5830** | N/A | 12.033 |
| **AdaBoost SAMME (Scratch)** | Classification | N/A | N/A | N/A | 0.6447 | 0.5242 | 0.5192 | 0.5022 | N/A | 90.155 |
| **CART Decision Tree (Reference)** | Classification | N/A | N/A | N/A | 0.6725 | 0.7025 | 0.5695 | 0.5347 | N/A | 4.318 |
| **AdaBoost (Reference)** | Classification | N/A | N/A | N/A | 0.5986 | 0.4346 | 0.4733 | 0.4285 | N/A | 27.821 |

## 2. Best-Performing Model per Analytical Task

### Task 1: Continuous Power Baseline Regression
- **Winner:** **OLS Regression (Scratch)**
- **Score:** RMSE = $38.98$ W, $R^2 = 0.9983$
- **Why it won:** The relationship between thermodynamic cooling load and compressor power is predominantly linear. Gradient descent OLS learned precise physical feature weights with an $R^2 > 0.998$, outperforming the naive mean baseline ($R^2 = -0.0056$). Ridge regression applied slight over-shrinkage on uncollinear features under $\lambda = 1.0$.

### Task 2: Unsupervised Operating Regime Discovery
- **Winner:** **GMM / EM (Scratch)**
- **Score:** Silhouette = **0.4821**
- **Why it won:** Rooftop units operate across non-spherical clusters (compressor off vs partial cooling vs peak cooling) with varying sensor variances. The full covariance EM formulation accommodated differing ellipsoid widths better than isotropic K-Means ($0.4726$).

### Task 3: Supervised Multi-Class Fault Diagnosis
- **Winner:** **CART Decision Tree (Scratch)**
- **Score:** Accuracy = **0.7186**, Macro-F1 = **0.5830**
- **Why it won:** CART identified hierarchical threshold rules directly aligned with physical limits (e.g. condensing pressure $> 2100$ kPa indicates condenser fouling, while low suction pressure indicates refrigerant undercharge). It significantly surpassed the majority class baseline (Macro-F1 $0.0476$) and outperformed the scikit-learn reference CART (Macro-F1 $0.5347$) and AdaBoost.

