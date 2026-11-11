# HVAC FaultSense
## Explainable HVAC Fault Detection, Diagnosis & Early-Warning System

> **Project Status:** Implemented, Audited, Evaluated & Validated  
> **Domain:** Commercial Rooftop Air Conditioning Units (RTU) Fault Detection & Diagnostics (FDD)  
> **Primary Dataset:** Lawrence Berkeley National Laboratory (LBNL) RTU Dataset (25 Simulated Scenarios + Field & Lab Repositories)  
> **Core Implementations:** From-scratch NumPy implementations for all major curriculum algorithms  

HVAC FaultSense is an explainable machine learning system for analyzing HVAC sensor telemetry, identifying abnormal energy deviations, discovering unlabelled operational regimes, decoding latent temporal states, and diagnosing verified mechanical equipment faults.

---

## 1. Five Curriculum Units & Implementation Scope

| Unit | Syllabus Focus | Custom Algorithm (NumPy from Scratch) | Evaluation Role | Best Metric |
|---|---|---|---|---|
| **Unit 1** | Dataset Profiling & Preprocessing | Statistical moments, IQR outlier filtering, split-safe z-scaling | Data quality audit & zero leakage | 0% missing values across 3.6M points |
| **Unit 2** | Power Baseline Regression | `LinearRegressionScratch` & `RidgeRegressionScratch` (Gradient Descent) | Predict total power from boundary conditions | **OLS $R^2 = 0.9840$, RMSE = 119.31 W** |
| **Unit 3** | Operating Regime Discovery | `PCAScratch`, `KMeansScratch`, `GaussianMixtureScratch` (EM) | Unsupervised operating mode partitioning | **K-Means Silhouette = 0.4966** (PC1+PC2: 71.09% var) |
| **Unit 4** | Probabilistic Sequence Tracking | `DiscreteHMMViterbiScratch` (Log-space Trellis Dynamic Programming) | Temporal smoothing & chatter reduction | Optimal latent sequence ($\ln P^* = -2761.80$) |
| **Unit 5** | Supervised Fault Classification | `DecisionTreeClassifierScratch` (CART Gini) & `AdaBoostClassifierScratch` (SAMME) | Multi-class fault diagnosis across 7 verified categories | **CART Accuracy = 71.86%, Macro-F1 = 0.5830** |

---

## 2. Official Dataset Provenance

HVAC FaultSense utilizes official files from the **Lawrence Berkeley National Laboratory (LBNL)** Fault Detection and Diagnostics archive:
- **Official Portal:** <https://faultdetection.lbl.gov/dataset/rtu/>
- **Data Catalog:** <https://catalog.data.gov/dataset/lbnl-fault-detection-and-diagnostics-datasets>
- **Dataset Publication:** Granderson et al. (2023), *A labeled dataset for building HVAC systems operating in faulted and fault-free states*, Nature Scientific Data. DOI: [10.1038/s41597-023-02197-w](https://doi.org/10.25984/1881324)

### Data Integrity Summary:
- **25 Simulated RTU Scenarios:** 143,941 rows per scenario at 1-minute intervals (Total: 3,598,525 rows).
- **Physical Sensors:** 25 verified channels mapped directly to the **Brick Schema RDF ontology** (`LBNL_FDD_Data_Sets_RTU_Sim.ttl`).
- **Separated Datasets:** Simulated (`data/raw/simulated/`), Field (`data/raw/field/`), and Laboratory (`data/raw/laboratory/`) data are cataloged independently.
- **No Fabricated Columns:** No mock labels or artificial column names were used.
- **Temporal Downsampling:** Downsampled via `stride=15` to $239,925$ rows ($9,597$ rows per scenario). This preserves diurnal and seasonal thermal dynamics (building thermal time constants typically range from 30 to 120 minutes) while allowing rapid, memory-safe execution.

---

## 3. Comprehensive Model Evaluation & Benchmarks

All models were evaluated on the strictly held-out test split ($60,000$ test records, $179,925$ training records) using time-aware and scenario-aware partitioning to prevent data leakage.

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

---

## 4. Repository Structure

```text
HVAC FaultSense/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── data.yaml
│   └── experiments.yaml
│
├── data/
│   ├── raw/
│   │   ├── LBNL_FDD_Data_Sets_RTU.pdf
│   │   ├── LBNL_FDD_Data_Sets_RTU.zip
│   │   ├── metadata/
│   │   ├── simulated/
│   │   ├── field/
│   │   └── laboratory/
│   ├── interim/
│   │   ├── dataset_manifest.csv
│   │   ├── data_dictionary.csv
│   ├── processed/
│   │   ├── test_evaluation_sample.parquet
│   │   └── test_evaluation_sample.csv
│   └── README.md
│
├── docs/
│   ├── architecture.md
│   ├── audit/
│   │   ├── initial-audit.md
│   │   ├── data-leakage-audit.md
│   │   └── final-audit-report.md
│   ├── dataset/
│   │   ├── dataset_overview.md
│   │   ├── data_dictionary.md
│   │   ├── data_quality_and_preprocessing.md
│   │   └── label_mapping.md
│   ├── algorithms/
│   │   ├── regression_and_ridge.md
│   │   ├── pca_kmeans_and_gmm.md
│   │   ├── hmm_and_viterbi.md
│   │   └── cart_and_adaboost.md
│   └── evaluation/
│       ├── evaluation_protocol.md
│       └── metrics_and_results.md
│
├── src/
│   └── hvac_faultsense/
│       ├── data/
│       │   ├── download.py
│       │   ├── ingestion.py
│       │   ├── metadata.py
│       │   ├── validation.py
│       │   └── preprocessing.py
│       ├── features/
│       │   └── engineering.py
│       ├── models/
│       │   ├── regression_scratch.py
│       │   ├── ridge_scratch.py
│       │   ├── pca_scratch.py
│       │   ├── kmeans_scratch.py
│       │   ├── gmm_em_scratch.py
│       │   ├── hmm_viterbi_scratch.py
│       │   ├── cart_classifier.py
│       │   └── adaboost_classifier.py
│       ├── evaluation/
│       │   ├── regression_metrics.py
│       │   ├── classification_metrics.py
│       │   └── validation.py
│       ├── visualization/
│       │   └── plots.py
│       └── pipeline.py
│
├── app/
│   └── streamlit_app.py
│
├── notebooks/
│   ├── 01_dataset_profiling.ipynb
│   └── 02_experiment_analysis.ipynb
│
├── experiments/
│   ├── trained_pipeline.pkl
│   └── README.md
│
├── reports/
│   ├── figures/
│   └── results/
│       ├── benchmark_summary.csv
│       ├── experiment_results.json
│       ├── data_profile.md
│       └── data_profile.json
│
└── tests/
    ├── test_ingestion.py
    ├── test_preprocessing.py
    ├── test_regression.py
    ├── test_pca_kmeans.py
    ├── test_hmm_viterbi.py
    ├── test_classifiers.py
    ├── test_no_data_leakage.py
    └── test_dashboard_contracts.py
```

---

## 5. Quickstart & Installation

### Step 1: Environment Setup
```powershell
# Navigate to repository
cd "c:\Users\Akash Singh\Desktop\HVAC FaultSense"

# Install dependencies
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```

### Step 2: Download Dataset & Run Sanitized Pipeline
```powershell
# Acquire official LBNL RTU dataset
python src/hvac_faultsense/data/download.py

# Execute end-to-end training and evaluation pipeline
python src/hvac_faultsense/pipeline.py
```

### Step 3: Run Automated Test Suite
```powershell
python -m pytest tests -v
```

### Step 4: Launch Interactive Streamlit Diagnostic Dashboard
```powershell
streamlit run app/streamlit_app.py
```

---

## 6. Limitations and Responsible Use

- **Decision-Support Only:** HVAC FaultSense is an analytical decision-support prototype. It must not directly control HVAC building automation systems or automatically shut down equipment.
- **Latent State Semantics:** Hidden Markov Model states reflect statistical persistence regimes, not confirmed mechanical degradation or physical health levels.
- **RUL & Future Warnings:** The system does not claim to predict remaining useful life or exact future breakdown dates without historical run-to-failure records.
