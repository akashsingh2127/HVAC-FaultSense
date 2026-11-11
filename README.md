# HVAC FaultSense
## Explainable HVAC Fault Detection, Diagnosis & Early-Warning System

> **Project Status:** Implemented, Evaluated & Validated  
> **Domain:** Commercial Rooftop Air Conditioning Units (RTU) Fault Detection & Diagnostics (FDD)  
> **Primary Dataset:** Lawrence Berkeley National Laboratory (LBNL) RTU Dataset (25 Simulated Scenarios + Field & Lab Repositories)  
> **Core Implementations:** From-scratch NumPy implementations for all major curriculum algorithms  

HVAC FaultSense is an explainable machine learning system for analyzing HVAC sensor telemetry, identifying abnormal energy deviations, discovering unlabelled operational regimes, decoding latent temporal states, and diagnosing verified mechanical equipment faults.

---

## 1. Five Curriculum Units & Implementation Scope

| Unit | Syllabus Focus | Custom Algorithm (NumPy from Scratch) | Evaluation Role | Best Metric |
|---|---|---|---|---|
| **Unit 1** | Dataset Profiling & Preprocessing | Statistical moments, IQR outlier filtering, split-safe z-scaling | Data quality audit & leakage prevention | 0% missing values across 3.6M points |
| **Unit 2** | Power Baseline Regression | `LinearRegressionScratch` & `RidgeRegressionScratch` (Gradient Descent) | Continuous expected power estimation & residual tracking | **OLS $R^2 = 0.9983$, RMSE = 38.98 W** |
| **Unit 3** | Operating Regime Discovery | `PCAScratch`, `KMeansScratch`, `GaussianMixtureScratch` (EM) | Unsupervised operating mode partitioning | **GMM Silhouette = 0.4821** (PC1+PC2: 70.91% var) |
| **Unit 4** | Probabilistic Sequence Tracking | `DiscreteHMMViterbiScratch` (Log-space Trellis Dynamic Programming) | Temporal smoothing & chatter reduction | Optimal latent sequence ($\ln P^* = -2805.39$) |
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

---

## 3. Comprehensive Model Evaluation & Benchmarks

All models were evaluated on the strictly held-out test split ($60,000$ test records, $179,925$ training records) using time-aware and scenario-aware partitioning to prevent data leakage.

| Model | Analytical Task | MAE (W) | RMSE (W) | $R^2$ | Accuracy | Precision | Recall | Macro-F1 | Silhouette | Train Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Mean Baseline** | Regression | 794.2322 | 944.5896 | -0.0056 | N/A | N/A | N/A | N/A | N/A | 0.001 |
| **OLS Regression (Scratch)** | Regression | **29.9171** | **38.9816** | **0.9983** | N/A | N/A | N/A | N/A | N/A | 36.101 |
| **Ridge Regression (Scratch)** | Regression | 137.9101 | 169.0907 | 0.9678 | N/A | N/A | N/A | N/A | N/A | 4.271 |
| **Linear Regression (Reference)** | Regression | 0.0000 | 0.0000 | 1.0000 | N/A | N/A | N/A | N/A | N/A | 0.840 |
| **K-Means (Scratch)** | Clustering | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.4726 | 7.263 |
| **GMM / EM (Scratch)** | Clustering | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **0.4821** | 5.316 |
| **Majority Class Baseline** | Classification | N/A | N/A | N/A | 0.2000 | 0.0286 | 0.1429 | 0.0476 | N/A | 0.157 |
| **CART Decision Tree (Scratch)** | Classification | N/A | N/A | N/A | **0.7186** | **0.6766** | **0.5953** | **0.5830** | N/A | 12.033 |
| **AdaBoost SAMME (Scratch)** | Classification | N/A | N/A | N/A | 0.6447 | 0.5242 | 0.5192 | 0.5022 | N/A | 90.155 |
| **CART Decision Tree (Reference)** | Classification | N/A | N/A | N/A | 0.6725 | 0.7025 | 0.5695 | 0.5347 | N/A | 4.318 |
| **AdaBoost (Reference)** | Classification | N/A | N/A | N/A | 0.5986 | 0.4346 | 0.4733 | 0.4285 | N/A | 27.821 |

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
    └── test_no_data_leakage.py
```

---

## 5. Quickstart & Installation

### Step 1: Environment Setup
```bash
# Clone or navigate to the repository
cd "c:\Users\Akash Singh\Desktop\HVAC FaultSense"

# Install dependencies
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```

### Step 2: Download Dataset & Run Pipeline
```bash
# Download official LBNL RTU archive (if not already downloaded)
python src/hvac_faultsense/data/download.py

# Execute end-to-end training and evaluation pipeline
python src/hvac_faultsense/pipeline.py
```

### Step 3: Run Test Suite
```bash
python -m pytest tests
```

### Step 4: Launch Interactive Streamlit Frontend
```bash
streamlit run app/streamlit_app.py
```

---

## 6. Limitations and Responsible Use

- **Decision-Support Only:** HVAC FaultSense is an analytical decision-support prototype. It must not directly control HVAC building automation systems or automatically shut down equipment.
- **Latent State Semantics:** Hidden Markov Model states reflect statistical persistence regimes, not confirmed mechanical degradation or physical health levels.
- **RUL & Future Warnings:** The system does not claim to predict remaining useful life or exact future breakdown dates without historical run-to-failure records.
