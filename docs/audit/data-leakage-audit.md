# HVAC FaultSense — Data Leakage and Evaluation Audit

> **Document Status:** Official Audit & Remediation Plan  
> **Auditor:** Senior Machine Learning Engineer & Research Reviewer  
> **Date:** 2026-11-11  

---

## 1. Investigation of the "Perfect" Regression Result

In the initial pipeline run, the reference linear regression model reported:
- **Mean Absolute Error (MAE):** $0.0000$ W
- **Root Mean Squared Error (RMSE):** $0.0000$ W
- **$R^2$ Score:** $1.0000$
- **From-Scratch OLS Regression:** $\text{RMSE} = 38.98$ W, $R^2 = 0.9983$

A perfect score ($R^2 = 1.0$) on a complex thermodynamic system is an immediate red flag indicating data leakage.

### Root Cause Analysis

Our audit traced the phenomenon to two distinct leakage mechanisms in `features/engineering.py` and `pipeline.py`:

#### 1. Direct Target Leakage in Feature Engineering
In `src/hvac_faultsense/features/engineering.py`, the following lines were present:
```python
if "RTU_COMP_WATT" in out.columns and "RTU_TOT_WATT" in out.columns:
    denom = out["RTU_TOT_WATT"].replace(0, np.nan)
    out["feat_comp_power_fraction"] = (out["RTU_COMP_WATT"] / denom).fillna(0.0)

if "RTU_SA_FAN_WATT" in out.columns and "RTU_TOT_WATT" in out.columns:
    denom = out["RTU_TOT_WATT"].replace(0, np.nan)
    out["feat_fan_power_fraction"] = (out["RTU_SA_FAN_WATT"] / denom).fillna(0.0)
```
The target variable `RTU_TOT_WATT` was directly used in the denominator of these engineered features. Because these features were included in the predictor matrix $\mathbf{X}$, the models had direct mathematical access to the target variable.

#### 2. Tautological Component-to-Total Power Identity
In physical HVAC engineering, the total electrical power consumed by a packaged rooftop unit is the direct physical sum of its major electrical subsystems:
$$\text{RTU\_TOT\_WATT} = \text{RTU\_COMP\_WATT} + \text{RTU\_SA\_FAN\_WATT} + \text{RTU\_COND\_FAN\_WATT} + P_{\text{controls}}$$

Because `RTU_COMP_WATT` and `RTU_SA_FAN_WATT` were included in the predictor set $\mathbf{X}$, any linear estimator can trivially fit:
$$\mathbf{w}_{\text{COMP}} \approx 1.0, \quad \mathbf{w}_{\text{FAN}} \approx 1.0, \quad \mathbf{w}_{\text{other}} \approx 0.0$$
This is not an energy baseline model; it is a trivial measurement identity.

---

## 2. Remediation: True Boundary-Driven HVAC Baseline

In accordance with **ASHRAE Guideline 14** and the **International Performance Measurement and Verification Protocol (IPMVP)**, a true baseline model must predict total electrical power consumption strictly from **independent driving variables**:

1. **Weather and Boundary Forcing:**
   - `RTU_OA_TEMP` (Outdoor air drybulb temperature)
   - `RTU_OA_HUM` (Outdoor air relative humidity)
   - `RTU_OA_FLOW` (Outdoor ventilation flow rate)
2. **Indoor Zone Thermal Conditions & Controls:**
   - `ZA_TEMP` (Zone temperature)
   - `ZA_HUM` (Zone relative humidity)
   - `ZA_TEMP_SPT` (Thermostat temperature setpoint)
   - `RTU_STG_STA` (Compressor staging state)
3. **Return Air States:**
   - `RTU_RA_TEMP`, `RTU_RA_HUM`, `RTU_RA_FLOW`
4. **Supply Air Airflow:**
   - `RTU_SA_FLOW`

### Strict Exclusion Rule
The following columns are **strictly excluded** from the regression feature matrix $\mathbf{X}$:
- `RTU_TOT_WATT` (Target)
- `RTU_COMP_WATT` (Sub-component power)
- `RTU_SA_FAN_WATT` (Sub-component power)
- `feat_comp_power_fraction` (Target leakage)
- `feat_fan_power_fraction` (Target leakage)

When predicting total electrical power from driving conditions alone, regression residuals $e_t = y_t - \hat{y}_t$ become genuine thermodynamic indicators of excess energy consumption or operating inefficiencies.

---

## 3. Evaluation Split Protocol Audit

### Previous Methodology: In-Scenario Chronological Split Only
The previous split held out the final 25% of each scenario file chronologically. While this properly prevented future-to-past temporal leakage, it meant that every fault scenario and every fault severity level was present in the training set.

### Improved Dual Evaluation Protocol
To provide a complete scientific evaluation:
1. **Protocol A (Temporal Generalization on Known Scenarios):**
   Chronological 75% train / 25% test within scenarios. Evaluates how well models predict future observations under known operating conditions.
2. **Protocol B (Scenario Holdout Generalization):**
   Holding out selected fault severity levels entirely from the training partition:
   - Held-out severity scenarios: `RTU_sim_condfouling30.csv`, `RTU_sim_evapfouling30.csv`, `RTU_sim_undercharge15.csv`, `RTU_sim_overcharge15.csv`.
   - Evaluates whether the fault classifiers can generalize to intermediate or novel fault severities never encountered during training.

All previous invalid regression metrics are formally discarded and will be regenerated using this sanitized protocol.

