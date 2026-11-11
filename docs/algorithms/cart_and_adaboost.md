# Unit 5: CART Decision Tree & AdaBoost Classifier from Scratch

## 1. What the Algorithms Do

Unit 5 focuses on **supervised multi-class fault diagnosis**, mapping physical sensor telemetry directly into known, verified equipment fault categories.

- **CART (Classification and Regression Trees):** Non-parametric hierarchical tree that identifies optimal binary decision boundaries using Gini impurity.
- **AdaBoost (Adaptive Boosting):** Sequential ensemble that iteratively increases the weight of misclassified training samples, building a strong classifier from shallow decision trees.

## 2. Mathematical Foundation

### A. CART Decision Tree
Gini Impurity of node $m$:
$$\text{Gini}(m) = 1 - \sum_{k=1}^{K} p_{mk}^2$$
where $p_{mk} = \frac{1}{N_m} \sum_{i \in R_m} \mathbb{I}(y_i = k)$ is the proportion of class $k$ in node $m$.

Split Selection:
For candidate feature $j$ and threshold $s$, the impurity reduction (Information Gain) is:
$$\Delta \text{Gini}(j, s) = \text{Gini}(\text{parent}) - \left( \frac{N_L}{N} \text{Gini}(\text{left}) + \frac{N_R}{N} \text{Gini}(\text{right}) \right)$$
The tree selects $(j^*, s^*) = \arg\max \Delta \text{Gini}(j, s)$ recursively until stopping constraints (`max_depth = 6`, `min_samples_split = 10`, `min_samples_leaf = 5`) are satisfied.

### B. AdaBoost (Multi-Class SAMME Formulation)
1. Initialize sample weights uniformly: $w_i^{(1)} = \frac{1}{n}$.
2. For $m = 1, \dots, M$:
   - Train weak learner $h_m(\mathbf{x})$ using current sample weights $w^{(m)}$.
   - Compute weighted error rate:
     $$\epsilon_m = \frac{\sum_{i: y_i \neq h_m(\mathbf{x}_i)} w_i^{(m)}}{\sum_{i=1}^n w_i^{(m)}}$$
   - Compute estimator weight using SAMME multi-class correction:
     $$\alpha_m = \beta \cdot \left[ \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right) + \ln(K - 1) \right]$$
   - Update sample weights:
     $$w_i^{(m+1)} = w_i^{(m)} \cdot \exp\left( \alpha_m \cdot \mathbb{I}(y_i \neq h_m(\mathbf{x}_i)) \right)$$
   - Renormalize: $w_i^{(m+1)} \leftarrow \frac{w_i^{(m+1)}}{\sum_{j=1}^n w_j^{(m+1)}}$.
3. Final ensemble prediction:
   $$C(\mathbf{x}) = \arg\max_{k \in \{1,\dots,K\}} \sum_{m=1}^{M} \alpha_m \cdot \mathbb{I}(h_m(\mathbf{x}) = k)$$

## 3. Measured Experimental Results (Held-Out Test Data)

| Model | Task | Accuracy | Precision | Recall | Macro-F1 |
|---|---|---|---|---|---|
| **Majority Baseline** | Classification | 0.2000 | 0.0286 | 0.1429 | 0.0476 |
| **CART Scratch** | Classification | **0.7186** | **0.6766** | **0.5953** | **0.5830** |
| **AdaBoost SAMME Scratch** | Classification | 0.6447 | 0.5242 | 0.5192 | 0.5022 |
| **CART Reference** | Classification | 0.6725 | 0.7025 | 0.5695 | 0.5347 |
| **AdaBoost Reference** | Classification | 0.5986 | 0.4346 | 0.4733 | 0.4285 |

**Winner:** `CART Decision Tree (Scratch)` achieved the highest Macro-F1 score (**0.5830**) and highest balanced accuracy across the 7 fault classes on held-out test scenarios!

