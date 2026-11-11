# Evaluation Protocol & Leakage Prevention Safeguards

## 1. Time-Aware and Scenario-Aware Splitting

Random row-shuffling is strictly prohibited in time-series HVAC telemetry because adjacent 1-minute observations share nearly identical physical states, causing inflated, misleading accuracy scores.

The HVAC FaultSense evaluation protocol enforces:
1. **Scenario-Wise Chronological Partition:** For each scenario series $s \in S$, the initial 75% of timestamps form the training split; the final 25% of timestamps form the held-out test split.
2. **Zero Information Bleed:** The test set remains completely unseen during all parameter estimation, feature scaling, and model tuning phases.
3. **Training-Only Transformers:** Imputers and scalers (`StandardScalerScratch`) are fitted strictly on training data:
   $$\mu_{\text{train}} = \frac{1}{n_{\text{train}}} \sum_{i \in \text{train}} \mathbf{x}_i, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{n_{\text{train}}} \sum_{i \in \text{train}} (\mathbf{x}_i - \mu_{\text{train}})^2}$$

## 2. Evaluation Metrics Protocol

Different analytical tasks require distinct, task-appropriate metrics. Under no circumstances are regression and classification metrics blended into a single misleading index.

- **Continuous Regression:**
  - Mean Absolute Error (MAE): $\frac{1}{n} \sum |y - \hat{y}|$
  - Root Mean Squared Error (RMSE): $\sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$
  - Coefficient of Determination ($R^2$): $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$
- **Unsupervised Clustering:**
  - Silhouette Score: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$
- **Supervised Classification:**
  - Accuracy: $\frac{\sum \mathbb{I}(y = \hat{y})}{n}$
  - Macro-F1: Unweighted mean of per-class F1-scores, penalizing models that ignore minority fault classes.
  - Balanced Accuracy: Mean of per-class recall rates.
  - Confusion Matrix: Explicit mapping of true vs predicted fault categories.

