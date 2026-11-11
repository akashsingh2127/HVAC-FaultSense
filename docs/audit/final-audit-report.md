# HVAC FaultSense — Final Technical Audit & Verification Report

> **Audit Date:** 2026-11-11  
> **Auditor Role:** Senior Machine Learning Engineer & Research Reviewer  
> **Audit Status:** Complete & Verified  

---

## 1. Executive Summary & Verification State

This final technical audit report synthesizes the end-to-end audit, remediation, and verification performed on **HVAC FaultSense: Explainable HVAC Fault Detection, Diagnosis & Early-Warning System**.

All critical flaws identified during the initial audit—most notably **direct target leakage** and **tautological component power inclusion** in the energy baseline regression pipeline—have been resolved. The feature engineering pipeline has been restructured to adhere strictly to ASHRAE Guideline 14 and IPMVP standards, predicting total electrical power exclusively from driving ambient, thermal, and control variables.

A dual evaluation protocol was implemented and executed:
1. **Protocol A (In-Scenario Chronological Holdout):** Evaluates temporal generalization on unseen future time horizons across known equipment operating scenarios ($179,925$ train records, $60,000$ test records).
2. **Protocol B (Unseen Scenario Severity Holdout):** Evaluates out-of-distribution generalization by withholding entire fault severity levels ($201,537$ train records, $38,388$ test records across unseen 30% fouling, 15% undercharge/overcharge cases).

All reported metrics are verified, reproducible, and produced by actual runs on the official Lawrence Berkeley National Laboratory (LBNL) Rooftop Unit (RTU) dataset.

---

## 2. Issues Discovered and Remediations Implemented

| Issue ID | Severity | Discovered Weakness | Engineering Remediation | Verification Status |
|---|---|---|---|---|
| **AUD-001** | **CRITICAL** | Target variable `RTU_TOT_WATT` was used in the denominator to create `feat_comp_power_fraction` and `feat_fan_power_fraction`, directly leaking target values into $\mathbf{X}$. | Removed all target-derived ratios from `features/engineering.py`. | **Verified & Resolved** |
| **AUD-002** | **CRITICAL** | Subsystem powers (`RTU_COMP_WATT`, `RTU_SA_FAN_WATT`) were included in the predictor matrix, allowing linear regression to trivially learn the physical identity $\text{Total} \approx \text{Comp} + \text{Fan}$ ($R^2 = 1.0000$, $\text{RMSE} = 0.0$ W). | Enforced strict feature separation in `get_feature_column_names(task='regression')`: only driving boundary variables (weather, zone load, staging status) are used. Resulting genuine $R^2 = 0.9840$, $\text{RMSE} = 119.31$ W. | **Verified & Resolved** |
| **AUD-003** | **HIGH** | Evaluation was previously limited to in-scenario chronological partitions, meaning models had seen every fault class during training. | Introduced dual evaluation protocol including Protocol B (unseen scenario holdout), testing fault classification on unseen severity levels. | **Verified & Resolved** |
| **AUD-004** | **HIGH** | 15-minute temporal downsampling (`stride=15`) was applied without documenting its impact on rapid transient dynamics vs building thermal time constants. | Documented downsampling rationale in `docs/dataset/dataset_overview.md` and exposed configuration in `configs/data.yaml`. | **Verified & Resolved** |
| **AUD-005** | **MEDIUM** | PCA lacked Frobenius reconstruction error; clustering evaluated only Silhouette score without Davies-Bouldin or Calinski-Harabasz metrics. | Added `compute_reconstruction_error()` to `PCAScratch` (MSE = $0.2965$) and integrated Davies-Bouldin ($0.7271$) and Calinski-Harabasz ($4,967.2$) into benchmark tables. | **Verified & Resolved** |
| **AUD-006** | **MEDIUM** | Streamlit dashboard lacked HVAC physical interpretability connecting regression residuals to physical faults. | Enriched all 8 dashboard tabs with thermodynamic failure signatures and physical explanations. | **Verified & Resolved** |
| **AUD-007** | **LOW** | Pytest test suite did not explicitly assert omission of sub-component powers from regression features. | Added dedicated test `test_no_target_or_component_power_leakage_in_regression` to `tests/test_no_data_leakage.py`. | **Verified & Resolved (28/28 tests passing)** |

---

## 3. Dataset Integrity & Downsampling Findings

- **Source Provenance:** Official LBNL RTU dataset portal (<https://faultdetection.lbl.gov/dataset/rtu/>).
- **Physical Dataset Inventory in `data/raw/` (85 Total Files on Disk):**
  - **81 Telemetry Scenario CSV Files** (Cataloged in `data/interim/dataset_manifest.csv`):
    - 25 simulated scenario CSV files (`data/raw/simulated/`, 3,598,525 raw rows total)
    - 4 field operational CSV files (`data/raw/field/`)
    - 52 laboratory environmental chamber CSV files (`data/raw/laboratory/`)
  - **2 Brick Schema Semantic Ontology Files** (`data/raw/metadata/`):
    - `LBNL_FDD_Data_Sets_RTU_Sim.ttl` (Simulated RTU ontology)
    - `LBNL_FDD_Data_Sets_RTU_Exp.ttl` (Experimental RTU ontology)
  - **2 Documentation & Source Archive Files** (`data/raw/`):
    - `LBNL_FDD_Data_Sets_RTU.pdf` (Official technical report, 1.6 MB)
    - `LBNL_FDD_Data_Sets_RTU.zip` (Original downloaded distribution archive, 291.4 MB)
  - *Mathematical Verification:*
    $$\begin{aligned}
    \text{Scenario CSV Files} &= 25 \text{ (sim)} + 4 \text{ (field)} + 52 \text{ (lab)} = 81 \\
    \text{Data \& Ontology Files} &= 81 \text{ (CSVs)} + 2 \text{ (TTLs)} = 83 \\
    \text{Total Raw Files on Disk} &= 83 + 1 \text{ (PDF)} + 1 \text{ (ZIP)} = 85
    \end{aligned}$$
- **Physical Sensor Schema:** 25 sensors verified against Brick Schema RDF ontology (`LBNL_FDD_Data_Sets_RTU_Sim.ttl`). Zero fabricated columns or synthetic sensor points.
- **Data Quality:** Zero missing values ($0.0\%$), zero duplicate records, contiguous 1-minute sampling intervals across all 25 simulated scenarios ($3,598,525$ raw rows).
- **Temporal Downsampling:** Downsampled via `stride=15` to $239,925$ rows ($9,597$ rows per scenario). This preserves diurnal and seasonal thermal dynamics (building thermal time constants typically range from 30 to 120 minutes) while allowing rapid, memory-safe execution.

---

## 4. Final Model Comparison & Verified Benchmark Metrics

The table below presents the verified benchmark results obtained on held-out test data after complete leakage elimination:

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

### Winning Models Summary
1. **Regression Winner:** `Linear Regression (Reference)` (RMSE = $102.29$ W, $R^2 = 0.9882$) closely followed by from-scratch `OLS Regression (Scratch)` (RMSE = $119.31$ W, $R^2 = 0.9840$). Both demonstrate that boundary-driven weather and zone temperature models accurately predict total cooling load without tautological component power leakage.
2. **Clustering Winner:** `K-Means (Scratch)` (Silhouette = **0.4966**, Davies-Bouldin = **0.7271**, Calinski-Harabasz = **4967.2**), effectively separating steady-state cooling capacity regimes.
3. **Classification Winner:** `CART Decision Tree (Scratch)` (Accuracy = **71.86%**, Macro-F1 = **0.5830**). CART split boundaries naturally aligned with physical thermodynamic thresholds, outperforming both the scikit-learn reference tree ($0.5346$) and AdaBoost ($0.4908$). On completely unseen fault severities (Protocol B), CART maintained a solid **0.4954** Macro-F1.

---

## 5. Test Execution Results

All unit and integration tests were executed via `pytest`:
```bash
python -m pytest tests -v
```
**Output:**
```text
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.1.1, pluggy-1.6.0
collected 28 items

tests/test_classifiers.py::test_cart_classifier_separable_data PASSED    [  3%]
tests/test_classifiers.py::test_adaboost_classifier_boosting PASSED      [  7%]
tests/test_classifiers.py::test_classification_metrics PASSED            [ 10%]
tests/test_classifiers.py::test_majority_baseline_classifier PASSED      [ 14%]
tests/test_dashboard_contracts.py::test_metadata_artifacts_exist PASSED  [ 17%]
tests/test_dashboard_contracts.py::test_reports_and_benchmark_results PASSED [ 21%]
tests/test_hmm_viterbi.py::test_viterbi_deterministic_example PASSED     [ 25%]
tests/test_hmm_viterbi.py::test_viterbi_numerical_stability PASSED       [ 28%]
tests/test_ingestion.py::test_create_synthetic_fixture PASSED            [ 32%]
tests/test_ingestion.py::test_data_dictionary_schema PASSED              [ 35%]
tests/test_ingestion.py::test_load_nonexistent_file_raises_error PASSED  [ 39%]
tests/test_ingestion.py::test_load_single_csv_with_stride PASSED         [ 42%]
tests/test_no_data_leakage.py::test_chronological_ordering_guarantee PASSED [ 46%]
tests/test_no_data_leakage.py::test_scaler_zero_leakage PASSED           [ 50%]
tests/test_no_data_leakage.py::test_no_target_or_component_power_leakage_in_regression PASSED [ 53%]
tests/test_no_data_leakage.py::test_scenario_holdout_split PASSED        [ 57%]
tests/test_pca_kmeans.py::test_pca_scratch PASSED                        [ 60%]
tests/test_pca_kmeans.py::test_kmeans_scratch_clusters PASSED            [ 64%]
tests/test_pca_kmeans.py::test_gmm_scratch_responsibilities PASSED       [ 67%]
tests/test_preprocessing.py::test_standard_scaler_scratch PASSED         [ 71%]
tests/test_preprocessing.py::test_standard_scaler_constant_column PASSED [ 75%]
tests/test_preprocessing.py::test_compute_domain_features PASSED         [ 78%]
tests/test_preprocessing.py::test_split_time_and_scenario_aware PASSED   [ 82%]
tests/test_regression.py::test_linear_regression_scratch_synthetic PASSED [ 85%]
tests/test_regression.py::test_linear_regression_closed_form PASSED      [ 89%]
tests/test_regression.py::test_ridge_regression_scratch_shrinkage PASSED [ 92%]
tests/test_regression.py::test_regression_metrics_calculations PASSED    [ 96%]
tests/test_regression.py::test_mean_baseline_regressor PASSED            [100%]

============================= 28 passed in 2.33s ==============================
```

---

## 6. Streamlit Dashboard Verification

The interactive Streamlit dashboard (`app/streamlit_app.py`) was verified across all 8 modular pages:
1. **A. Overview:** Verified dynamic loading of cataloged equipment scenario files ($81$ CSVs in manifest), simulated scenarios ($25$), sensor channels ($25$), and top model metric cards.
2. **B. Dataset Explorer:** Verified interactive subsystem filtering on the Brick Schema Data Dictionary, scenario manifest filtering, and missing value audit chart.
3. **C. Sensor Analysis:** Verified time-series telemetry charts, distribution histograms, and box plots grouped by fault class.
4. **D. Regression Analysis:** Verified actual vs predicted power scatter plots, residual histograms, and $\pm 3\sigma$ threshold anomaly detection over time with thermodynamic physical interpretation.
5. **E. Anomaly and Clustering:** Verified 2D PCA projection scatter plots colored by K-Means operating regimes vs true scenario metadata, alongside Silhouette ($0.4966$), Davies-Bouldin ($0.7271$), and Calinski-Harabasz ($4,967.2$) metric cards.
6. **F. Sequential Analysis:** Verified temporal comparison of discrete K-Means cluster observations against the smoothed Viterbi latent path.
7. **G. Fault Diagnosis:** Verified normalized confusion matrices for CART (Scratch), AdaBoost (Scratch), and CART (Unseen Severity Holdout), per-class precision/recall tables, and physical fault explanations.
8. **H. Model Comparison:** Verified consolidated benchmark tables and Plotly comparison bar charts.

No metrics or charts are hardcoded; all visual elements load directly from `reports/results/` and `data/processed/`.

---

## 7. Remaining Limitations

1. **Simulated vs Real-World Equipment Dynamics:** The primary baseline models are trained on physics-based EnergyPlus/Modelica simulation datasets. Real-world commercial RTUs exhibit sensor calibration drift, airflow damper linkage binding, and dirty coils with non-uniform fouling profiles that require site-specific re-calibration.
2. **Advisory Decision Support:** The system is strictly an advisory decision-support prototype. It must not execute automated control shutdowns or supersede qualified HVAC technicians.
3. **No Remaining Useful Life (RUL) Claim:** The dataset documents step-injected faults rather than gradual wear-and-tear run-to-failure fatigue curves; estimating component remaining useful life is scientifically unsupported without run-to-failure degradation data.

---

## 8. Exact Setup and Execution Commands

```powershell
# 1. Navigate to workspace
cd "c:\Users\Akash Singh\Desktop\HVAC FaultSense"

# 2. Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps

# 3. Execute the full automated pipeline (runs all 5 ML units & exports results)
python src/hvac_faultsense/pipeline.py

# 4. Run the full test suite
python -m pytest tests -v

# 5. Export standalone interactive HTML figures
python scripts/export_figures.py

# 6. Launch the Streamlit diagnostic dashboard
streamlit run app/streamlit_app.py
```

