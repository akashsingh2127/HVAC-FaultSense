# HVAC FaultSense — Initial Technical Audit Report

> **Audit Date:** 2026-11-11  
> **Auditor Role:** Senior Machine Learning Engineer & Research Reviewer  
> **Scope:** Full repository audit (data ingestion, preprocessing, leakage, custom algorithms, evaluation, Streamlit dashboard)

---

## 1. Executive Summary

An in-depth technical audit of the **HVAC FaultSense** codebase was conducted to verify scientific validity, mathematical correctness, preprocessing rigor, and evaluation reliability.

While the project has successfully downloaded and organized the official Lawrence Berkeley National Laboratory (LBNL) RTU dataset and implemented all required algorithms from scratch in NumPy, the audit identified **critical target leakage** in the regression feature pipeline that explains the artificially perfect regression scores ($R^2 = 1.0000$, $\text{RMSE} = 0.0000$ W). Furthermore, the evaluation protocol requires both chronological and scenario-holdout validation to rigorously test model generalization.

---

## 2. Prioritized Audit Findings

| Issue ID | Severity | Component | Finding Description | Remediation Plan |
|---|---|---|---|---|
| **AUD-001** | **CRITICAL** | `features/engineering.py` | **Direct Target Leakage in Feature Engineering:** `feat_comp_power_fraction` and `feat_fan_power_fraction` were calculated by dividing by the target variable `RTU_TOT_WATT`. | Remove all engineered features that depend on `RTU_TOT_WATT`. |
| **AUD-002** | **CRITICAL** | `features/engineering.py` & `pipeline.py` | **Tautological Predictor Leakage in Baseline Regression:** Predictor matrix $X$ included component electrical powers (`RTU_COMP_WATT`, `RTU_SA_FAN_WATT`). Because $\text{Total Power} \approx \text{Compressor Power} + \text{Fan Power}$, linear models learn an identity mapping resulting in artificial $R^2 = 1.0$. | Strictly exclude all component power measurements from $X$ when predicting total electrical power; predict power strictly from driving ambient/thermal variables. |
| **AUD-003** | **HIGH** | `pipeline.py` & `data/preprocessing.py` | **Evaluation Strategy Limited to In-Scenario Time Splits:** Train/test split held out the last 25% of each scenario. While chronologically sound, every fault class was seen during training. | Introduce dual evaluation: (1) Chronological holdout on known scenarios, and (2) Unseen-scenario holdout (generalization to unseen fault severities). |
| **AUD-004** | **HIGH** | `data/ingestion.py` | **Undocumented Temporal Downsampling:** Stride parameter (`stride=15`) reduced 3,598,525 1-minute records to 239,925 15-minute records without explicit documentation of its impact on transient physical dynamics. | Document downsampling rationale (thermal time constants vs fast valve transients) and expose configuration in `configs/data.yaml`. |
| **AUD-005** | **MEDIUM** | `evaluation/` & `models/pca_scratch.py` | **Incomplete Unsupervised Metrics:** PCA lacked explicit Frobenius reconstruction error calculation; clustering evaluated only Silhouette score without Davies-Bouldin or Calinski-Harabasz metrics. | Add reconstruction error to PCA and compute Davies-Bouldin and Calinski-Harabasz scores for clustering. |
| **AUD-006** | **MEDIUM** | `app/streamlit_app.py` | **HVAC Explainability Depth:** Dashboard lacked interactive interpretation connecting regression residuals to physical thermodynamic faults (e.g. positive residual indicating condenser fouling vs negative residual). | Enhance diagnostic explainability across dashboard tabs, linking sensor patterns to physical failure modes. |
| **AUD-007** | **LOW** | `tests/` | **Test Suite Coverage Gaps:** Existing tests did not explicitly verify that component powers are omitted from the regression predictor matrix. | Add dedicated anti-leakage tests asserting feature matrix column exclusions. |

---

## 3. Dataset Audit & Downsampling Verification

1. **Official LBNL RTU Dataset Structure:**
   - **81 Equipment Scenario CSV Files** (Cataloged in `dataset_manifest.csv`):
     - 25 simulated scenario CSV files in `data/raw/simulated/`
     - 4 field operational CSV files in `data/raw/field/`
     - 52 laboratory environmental chamber CSV files in `data/raw/laboratory/`
   - **2 Brick Schema RDF Ontology Files** in `data/raw/metadata/` (`LBNL_FDD_Data_Sets_RTU_Sim.ttl`, `LBNL_FDD_Data_Sets_RTU_Exp.ttl`)
   - **2 Documentation & Source Archive Files** in `data/raw/`:
     - 1 official technical report: `LBNL_FDD_Data_Sets_RTU.pdf` (1.6 MB)
     - 1 original archive package: `LBNL_FDD_Data_Sets_RTU.zip` (291.4 MB)
   - **Total Raw Files on Disk:** **85 files** in `data/raw/` (81 Scenario CSVs + 2 Metadata TTLs + 1 PDF report + 1 ZIP archive = 85 files; Subtotal of data and ontology files = 83 files).

2. **Record Count Reconciliation:**
   - Raw records per file: $143,941$ rows
   - Total raw simulated records: $25 \times 143,941 = 3,598,525$ rows
   - Stride factor: $15$ (subsampling every 15th observation)
   - Processed records per file: $\lceil 143,941 / 15 \rceil = 9,597$ rows
   - Total processed records: $25 \times 9,597 = 239,925$ rows

3. **Temporal Impact of 15-Minute Downsampling:**
   - Commercial building thermal zones have time constants ranging from 30 minutes to several hours; 15-minute intervals adequately capture zone temperature and cooling demand shifts.
   - However, fast control actions (e.g., compressor stage hunting, instantaneous TXV hunting) occur on 1- to 3-minute time scales.
   - The downsampling is scientifically valid for baseline energy modeling and macro fault diagnosis, provided this design decision is transparently documented.

