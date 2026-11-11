# Data Quality, Statistical Profiling & Preprocessing

## 1. Data Quality Audit Findings

Execution of `src/hvac_faultsense/data/validation.py` across the simulated RTU dataset confirmed:

1. **Completeness:** 0 missing values across all 25 simulated CSV files ($0.0\%$ missingness).
2. **Duplicate Checks:** 0 duplicate rows. Duplicate timestamps only exist across distinct scenario files representing parallel experimental injections.
3. **Sampling Continuity:** Exactly 1-minute intervals across all 143,941 timestamps per scenario file.
4. **Outlier Boundaries:** Outlier counts computed using the standard Interquartile Range rule:
   $$\text{IQR} = Q_3 - Q_1$$
   $$\text{Lower} = Q_1 - 1.5 \cdot \text{IQR}, \quad \text{Upper} = Q_3 + 1.5 \cdot \text{IQR}$$
   Observations outside these bounds correspond to extreme hot days or severe fault stages (e.g., $50\%$ condenser fouling causing high discharge pressure), rather than corrupted sensor transmissions.

## 2. Leakage-Safe Preprocessing Pipeline

To ensure rigorous scientific validity and prevent data leakage:

### A. Chronological Split within Scenarios
For every distinct operational scenario, observations are sorted chronologically. The first $75\%$ of the timeline is allocated to training; the final $25\%$ of the timeline is held out untouched for test evaluation.

$$\text{Train Window:} \quad t_1 \le t \le t_{\lfloor 0.75 \cdot T \rfloor}$$
$$\text{Test Window:} \quad t_{\lfloor 0.75 \cdot T \rfloor + 1} \le t \le t_T$$

Under no circumstances are rows shuffled randomly across time.

### B. Standardization Fitted Exclusively on Training Data
Feature transformation uses `StandardScalerScratch`:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

The mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ are calculated exclusively from the training set and applied to the test set without re-estimation. Constant features with $\sigma < 10^{-8}$ have their standard deviation set to $1.0$ to guarantee numerical stability.

### C. Feature Separation
Target variables (`RTU_TOT_WATT`), fault categories (`fault_type`), fault severities, and timestamps are stripped from feature matrices prior to model ingestion.

