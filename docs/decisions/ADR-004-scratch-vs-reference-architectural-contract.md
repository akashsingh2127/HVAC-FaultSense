# ADR-004: Scratch Implementation vs. Scikit-learn Baseline Contract

## Status
Accepted

## Context
HVAC FaultSense is designed as a rigorous academic and applied Machine Learning research project covering five core syllabus units:
1. Data understanding and preprocessing.
2. OLS and Ridge regression.
3. PCA, K-Means, and GMM/EM.
4. Discrete Hidden Markov Models with log-space Viterbi decoding.
5. CART decision trees and AdaBoost (SAMME) multi-class boosting.

A central requirement is that all primary learning algorithms must be developed from first mathematical principles in pure Python/NumPy, without delegating model fitting or inference to high-level libraries. At the same time, objective scientific validation requires comparing these custom implementations against battle-tested reference implementations.

## Decision
1. **Primary Implementations:**
   - All core models (`LinearRegressionScratch`, `RidgeRegressionScratch`, `PCAScratch`, `KMeansScratch`, `GaussianMixtureScratch`, `DiscreteHMMViterbiScratch`, `DecisionTreeClassifierScratch`, `AdaBoostClassifierScratch`) are written purely in NumPy.
   - Numerical safeguards (e.g., log-sum-exp stabilization, log-space trellis arithmetic in Viterbi, covariance shrinkage in GMM, and epsilon jitter on diagonal matrices) are implemented directly in scratch code.
2. **Reference Implementations:**
   - Scikit-learn models (`LinearRegression`, `DecisionTreeClassifier`, `AdaBoostClassifier`) and baseline heuristics (`MeanBaselineRegressor`, `MajorityBaselineClassifier`) are executed alongside custom algorithms as independent reference baselines.
   - Reference implementations run on identical datasets, identical feature matrices, and identical evaluation splits.
   - Scikit-learn is never used as a silent fallback or hidden component inside scratch model logic.
3. **Registry and Dashboard Presentation:**
   - The `ModelComparisonRegistry` and Streamlit dashboard present custom vs. reference results side-by-side with transparent execution timings, metric deltas, and residual analyses.

## Consequences
- **Positive:**
  - Demonstrates mathematical depth and full control over numerical stability, decision boundaries, and loss surfaces.
  - Transparent comparison proves that custom implementations achieve competitive performance (e.g., custom CART 71.86% vs reference CART 67.25% accuracy) while maintaining pedagogical clarity.
- **Negative / Trade-offs:**
  - Pure Python/NumPy implementations exhibit longer training times for iterative ensembles (e.g., AdaBoost SAMME 50.19s vs scikit-learn Cythonized 18.66s).

