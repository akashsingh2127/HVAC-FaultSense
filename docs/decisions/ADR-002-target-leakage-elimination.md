# ADR-002: Elimination of Target Leakage & Boundary-Driven Load Modeling

## Status
Accepted

## Context
During initial repository inspection, reference linear regression reported $R^2 = 1.0000$ and $\text{RMSE} = 0.0000$ W when predicting total electrical power (`RTU_TOT_WATT`). Technical audit revealed two critical leakage mechanisms:
1. Direct inclusion of `RTU_TOT_WATT` in the denominator of engineered feature columns (`feat_comp_power_fraction` and `feat_fan_power_fraction`).
2. Inclusion of electrical sub-component powers (`RTU_COMP_WATT` and `RTU_SA_FAN_WATT`) in the predictor matrix $X$. Because $\text{RTU\_TOT\_WATT} \approx \text{RTU\_COMP\_WATT} + \text{RTU\_SA\_FAN\_WATT}$, any linear regressor trivially solves this physical identity with weights $[1.0, 1.0, 0, \dots]$, yielding an artificial $R^2 = 1.0$.

## Decision
1. **Remove Target-Derived Features:** Eliminate all engineered features that depend directly or indirectly on `RTU_TOT_WATT`.
2. **Exclude Sub-Component Powers:** Strictly exclude `RTU_COMP_WATT` and `RTU_SA_FAN_WATT` from the regression feature matrix $X$.
3. **ASHRAE Guideline 14 & IPMVP Compliance:** Restructure the baseline energy model to predict electrical demand exclusively from independent driving boundary conditions:
   - Ambient weather forcing: `RTU_OA_TEMP`, `RTU_OA_HUM`, `RTU_OA_FLOW`.
   - Indoor zone thermal state: `ZA_TEMP`, `ZA_HUM`, `ZA_TEMP_SPT`.
   - Control signals: `RTU_STG_STA` (compressor staging status).
   - Airflow state: `RTU_RA_TEMP`, `RTU_RA_HUM`, `RTU_RA_FLOW`, `RTU_SA_FLOW`.
4. **Enforce with Automated Tests:** Add assertions in `tests/test_no_data_leakage.py` guaranteeing that target and component power columns cannot enter the regression feature matrix.

## Consequences
- **Positive:**
  - Eliminates the artificial $R^2 = 1.0$ tautology, producing true physical baseline performance ($R^2 = 0.9840$, $\text{RMSE} = 119.31$ W for custom OLS).
  - Regression residuals $e_t = y_t - \hat{y}_t$ gain genuine diagnostic utility: positive residuals signify excess energy draw indicative of condenser fouling or thermodynamic inefficiencies.
- **Negative / Trade-offs:**
  - Requires clear documentation in the Streamlit frontend and technical reports explaining why earlier $R^2 = 1.0$ numbers were invalid.

