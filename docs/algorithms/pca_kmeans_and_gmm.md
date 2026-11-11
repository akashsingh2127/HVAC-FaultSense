# Unit 3: PCA, K-Means, and GMM/EM from Scratch

## 1. What the Algorithms Do

Unsupervised techniques discover operating regimes and reduce high-dimensional telemetry without relying on fault labels:
- **PCA:** Projects collinear sensor measurements onto orthogonal axes of maximum variance.
- **K-Means:** Partitions continuous sensor observations into $K$ distinct operational clusters.
- **GMM/EM:** Fits a probabilistic mixture of Gaussians, providing soft assignment probabilities (responsibilities).

## 2. Mathematical Formulations

### A. Principal Component Analysis (PCA)
1. Center features: $\mathbf{X}_c = \mathbf{X} - \bar{\mathbf{X}}$.
2. Empirical covariance matrix:
   $$\mathbf{\Sigma} = \frac{1}{n-1} \mathbf{X}_c^T \mathbf{X}_c$$
3. Spectral Eigendecomposition:
   $$\mathbf{\Sigma} \mathbf{v}_j = \lambda_j \mathbf{v}_j$$
4. Sort eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p \ge 0$.
5. Projection into $k$ dimensions: $\mathbf{Z} = \mathbf{X}_c \mathbf{W}$, where $\mathbf{W} = [\mathbf{v}_1, \dots, \mathbf{v}_k]$.
6. Explained variance ratio: $\frac{\lambda_j}{\sum_{i=1}^p \lambda_i}$.

### B. K-Means Clustering
Objective (Inertia):
$$J_{\text{KMeans}} = \sum_{i=1}^{n} \min_{k \in \{1,\dots,K\}} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$$

- **Initialization:** k-means++ assigns the first centroid uniformly at random and subsequent centroids with probability proportional to squared distance $D(\mathbf{x})^2$.
- **Assignment Step:** $c_i = \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$.
- **Update Step:** $\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i$.

### C. Gaussian Mixture Model (GMM) with Expectation-Maximization (EM)
Log-Likelihood objective:
$$\ln p(\mathbf{X} | \boldsymbol{\pi}, \boldsymbol{\mu}, \mathbf{\Sigma}) = \sum_{i=1}^{n} \ln \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \mathbf{\Sigma}_k) \right)$$

- **E-Step:** Compute posterior responsibilities using log-sum-exp stabilization:
  $$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \mathbf{\Sigma}_j)}$$
- **M-Step:** Parameter updates with covariance regularizer $\epsilon \mathbf{I}$:
  $$N_k = \sum_{i=1}^n \gamma_{ik}, \quad \pi_k = \frac{N_k}{n}, \quad \boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} \mathbf{x}_i$$
  $$\mathbf{\Sigma}_k = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T + \epsilon \mathbf{I}$$

## 3. Measured Results on Held-Out Test Data

- **PCA Explained Variance:** PC1 captures **54.25%**, PC2 captures **16.66%** (Cumulative 2D representation: **70.91%**).
- **K-Means Silhouette Score:** $0.4726$
- **GMM / EM Silhouette Score:** **0.4821** (Best clustering configuration)

## 4. Engineering Interpretation Rule

Unsupervised clusters represent **operating regimes** (e.g. stage off, partial cooling, maximum cooling demand), **not health states**. Cluster 0 does not automatically mean "healthy", and Cluster 1 does not mean "faulty".

