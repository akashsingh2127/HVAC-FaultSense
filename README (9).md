# HVAC FaultSense
## Explainable HVAC Fault Detection, Diagnosis & Early-Warning System

> **Project status:** Specification / implementation plan  
> **Primary domain:** Industrial and commercial HVAC fault detection and diagnostics (FDD)  
> **Primary data source:** Lawrence Berkeley National Laboratory (LBNL) Fault Detection and Diagnostics datasets  
> **Initial equipment scope:** Rooftop air-conditioning units (RTUs)

HVAC FaultSense is an explainable machine-learning system for analysing HVAC telemetry, identifying abnormal operating behaviour, diagnosing known equipment faults, and tracking changes in operating state over time.

The project combines statistical analysis, regression, dimensionality reduction, clustering, probabilistic sequence modelling, decision trees, and boosting. Its purpose is both educational and engineering-oriented: every major algorithm should be understandable, testable, and evaluated against appropriate baselines.

**Important distinction:** fault detection, fault classification, anomaly detection, and future failure prediction are different tasks. This project will report them separately. It will not claim to predict a breakdown date or remaining useful life (RUL) unless a defensible future-outcome target and evaluation protocol are added.

---

## 1. Project objectives

The system is intended to:

1. Ingest and validate timestamped HVAC sensor telemetry.
2. Profile sensor distributions, missingness, outliers, correlations, and operating regimes.
3. Estimate expected energy or power behaviour under comparable operating conditions.
4. Detect unusual observations and operating patterns without requiring fault labels.
5. Classify known fault conditions where reliable ground-truth labels are available.
6. Track likely operating states across time to reduce noisy, isolated alerts.
7. Explain model outputs through residuals, sensor contributions, feature importance, and diagnostic evidence.
8. Compare models using leakage-safe evaluation and reproducible experiments.

### Intended outputs

- A data-quality and dataset-profile report.
- A healthy-operation baseline and regression residuals.
- Anomaly scores and unsupervised operating clusters.
- A temporally smoothed operating-state sequence.
- Fault-class predictions for supported labelled scenarios.
- Evaluation metrics, confusion matrices, and feature-importance reports.
- A diagnostic dashboard suitable for reviewing model behaviour.

This is a research/academic decision-support prototype, not a certified building-control product. It must not automatically shut down equipment or replace a qualified HVAC technician.

---

## 2. Dataset strategy

### Primary dataset: LBNL Fault Detection and Diagnostics (FDD)

Use the publicly documented **LBNL FDD datasets**, curated by Lawrence Berkeley National Laboratory with partner national laboratories and research institutions.

- Official dataset portal: <https://faultdetection.lbl.gov/data/>
- RTU dataset details: <https://faultdetection.lbl.gov/dataset/rtu/>
- Dataset catalogue: <https://catalog.data.gov/dataset/lbnl-fault-detection-and-diagnostics-datasets>
- Dataset paper: <https://www.nature.com/articles/s41597-023-02197-w>
- DOI: <https://doi.org/10.25984/1881324>

The LBNL collection covers multiple HVAC equipment types and contains time-series data for fault-free and faulted operating conditions. The records are accompanied by dataset inventories describing equipment, points, fault scenarios, and data-generation methods.

### Initial implementation scope: RTU

Start with the **Rooftop Unit (RTU)** collection rather than mixing unrelated HVAC equipment types in the first experiment.

The official RTU page documents:

- Three RTU data groups: simulated, laboratory, and field data.
- 10 fault types represented at different severity levels.
- 74 CSV files in the simulated RTU collection, covering faulted cases and fault-free cases.
- A sampling interval of one minute.
- Approximately 24–56 measured data points, depending on the dataset.
- A separate inventory describing the scenarios and measurement points.

The official page currently lists the simulated RTU ZIP at approximately 292 MB and the combined RTU resources at approximately 0.98 GB. The download flow may request basic user and research-purpose information.

**Why RTU first?** It provides a manageable, well-documented starting point and includes simulation, laboratory, and field-measured data across the RTU collection. It is more appropriate for a first fault-diagnosis implementation than an arbitrary unlabeled sensor file.

### Dataset selection rules

1. Download the official RTU dataset and its inventory/documentation.
2. Keep the original downloaded files unchanged in the raw-data area.
3. Start with the simulated RTU subset for the reproducible baseline experiment.
4. Treat each CSV file's scenario and fault severity as metadata only after confirming its meaning in the official inventory.
5. Do not infer a fault label from a filename without validating the filename-to-scenario mapping against the inventory.
6. Keep simulated, laboratory, and field data distinguishable. Do not silently combine them.
7. Add laboratory or field data as a separate generalisation experiment after the initial pipeline works.

### Important schema clarification

The dataset described above is **not assumed to contain** the exact 20 columns in the original project description (`high_pressure_1`, `low_pressure_1`, `co2_1`, etc.). Its actual columns must be discovered from the downloaded CSV files and inventory.

The implementation must create a verified data dictionary from the real files. It must not fabricate columns, units, labels, sampling rates, or sensor meanings to match an assumed schema.

The LBNL RTU files are CSV. Parquet may be generated as an optional processed-data format after ingestion; it is not the source format for this selected dataset.

---

## 3. Problem formulation

HVAC FaultSense will implement distinct analytical tasks. Their targets and evaluation methods must not be conflated.

| Task | Learning setup | Output | Ground truth required? |
|---|---|---|---|
| Data-quality analysis | Statistical profiling | Missingness, invalid values, sensor quality | No |
| Anomaly detection | Unsupervised / semi-supervised | Anomaly score or flag | No labels required for fitting |
| Operating-regime discovery | Unsupervised | Cluster assignments | No |
| Power/energy baseline | Regression | Expected value and residual | Target measurement required |
| Fault diagnosis | Supervised classification | Known fault class | Yes |
| Sequential state tracking | Probabilistic sequence model | Most likely latent state path | Sequence model assumptions and ordered observations |
| Future fault warning | Forecasting / horizon classification | Risk within a defined future window | Future-outcome labels required |

### What “predictive maintenance” means in this project

The initial release focuses on **early fault indication and diagnosis**. It does not estimate component remaining useful life.

A future-warning experiment may be added only if the data supports a defensible prediction horizon, such as whether a fault begins within the next *H* minutes or hours. Features must be calculated only from information available before that horizon. Random row splitting is not acceptable for this experiment.

---

## 4. System workflow

```text
Official LBNL RTU files + inventory
                 |
                 v
       Ingestion and validation
                 |
                 v
     Dataset profiling and EDA
                 |
                 v
 Time-aware split + preprocessing
                 |
       +---------+----------+
       |         |          |
       v         v          v
  Regression  Unsupervised  Labelled
  baseline    analysis      diagnosis
       |       |             |
       v       v             v
  Residuals  PCA + K-Means  CART + AdaBoost
       |       |             |
       +---------+-----------+
                 |
                 v
       HMM sequential tracking
                 |
                 v
      Evaluation and explanations
                 |
                 v
       Diagnostic dashboard/reports
```

The exact implementation order should follow the milestones below. A dashboard is not a substitute for validated data processing or model evaluation.

---

## 5. Machine-learning curriculum mapping

The project preserves the five-unit academic structure while making the implementation and claims technically precise.

### Unit 1 — Introduction, data profiling and preprocessing

**Purpose:** Understand the physical measurements and establish a trustworthy modelling dataset.

Implementation:

- Read source CSV files with Pandas.
- Parse timestamps and preserve chronological order.
- Build a dataset manifest containing file, equipment group, scenario, fault type, severity, and provenance where documented.
- Generate a data dictionary containing column name, inferred/verified meaning, unit, data type, missing count, and valid range where known.
- Calculate mean, median, variance, standard deviation, quantiles, skewness, and correlation.
- Visualise sensor distributions, time-series behaviour, missingness, and outliers.
- Detect duplicate timestamps, irregular sampling, constant sensors, and impossible values.
- Separate raw, validated, and model-ready data.

**Deliverables:** dataset inventory, data dictionary, data-quality report, EDA figures, and preprocessing specification.

Do not scale or impute the entire dataset before splitting. Fit preprocessing transformations on training data only.

### Unit 2 — Linear regression and Ridge regression from scratch

**Purpose:** Build an interpretable reference model for expected power or another verified continuous operating variable.

Implement with NumPy:

- Ordinary Least Squares (OLS) linear regression.
- Gradient-descent optimisation, with a documented objective and convergence criteria.
- Ridge regression with an L2 penalty.
- Feature standardisation using training-set statistics only.
- Prediction, residual calculation, and coefficient inspection.

For a regression target such as measured active power, the residual is:

\[
e_t = y_t - \hat{y}_t
\]

Large or persistent residuals can indicate behaviour not explained by the baseline. They are **candidate anomaly evidence**, not proof of a mechanical fault.

Ridge objective:

\[
J(\mathbf{w},b) =
\frac{1}{n}\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
+\lambda\sum_{j=1}^{p}w_j^2
\]

Do not penalise the intercept unless explicitly justified. Select the regularisation strength using training/validation data, not the final test period.

**Evaluation:** MAE, RMSE, and \(R^2\), alongside residual plots and stability checks.

**Interpretation:** Ridge can reduce coefficient magnitude under multicollinearity. It does not automatically guarantee better generalisation; compare it with OLS and a simple baseline.

### Unit 3 — PCA, K-Means and mixture-model concepts

**Purpose:** Explore high-dimensional sensor behaviour and discover operating regimes without relying on fault labels.

Implement:

- PCA using a documented covariance/SVD approach.
- Explained variance ratio and cumulative explained variance.
- Two-dimensional PCA projection for visualisation.
- K-Means clustering with a documented distance metric, initialisation, and stopping rule.
- Cluster-quality analysis using suitable metrics such as silhouette score, where meaningful.
- Optional Gaussian Mixture Model (GMM) with Expectation-Maximisation (EM), if the course requires an explicit mixture-model implementation.

Important distinctions:

- PCA is dimensionality reduction, not a fault classifier.
- K-Means clusters are not inherently “normal” and “anomalous”.
- Cluster meaning must be investigated using sensor patterns, operating context, and known scenario metadata.
- If GMM/EM is included, document its likelihood objective, E-step, M-step, convergence criteria, and numerical-stability safeguards.

**Deliverables:** explained-variance plot, PCA scatter plot, cluster profiles, and a written interpretation of each discovered regime.

### Unit 4 — Hidden Markov Model and Viterbi algorithm

**Purpose:** Model temporal continuity and estimate a likely sequence of hidden operating states.

Implement a discrete Hidden Markov Model (HMM) and Viterbi decoding from scratch, with clearly documented:

- Hidden states.
- Observation symbols or emission model.
- Initial-state probabilities.
- State-transition probabilities.
- Emission probabilities.
- Log-probability calculations to avoid numerical underflow.
- Backtracking to recover the most likely state sequence.

A practical first version may map validated observation groups or cluster IDs to discrete emissions. This mapping must be documented and checked; cluster IDs are arbitrary identifiers, not ordered health levels.

The HMM can smooth noisy state changes and represent persistence. It does **not** prove that a unit is physically degrading, and a transition from one latent state to another must not automatically be described as “healthy to faulty” without supporting labels and validation.

**Evaluation:** compare the decoded sequence with known scenario periods where labels are available; report transition behaviour and state stability. Avoid presenting visual smoothness alone as evidence of accuracy.

### Unit 5 — CART and AdaBoost ensemble classification

**Purpose:** Diagnose known fault categories using labelled scenarios.

Implement:

- A CART-style decision tree classifier.
- Gini impurity or entropy for split selection.
- Maximum depth, minimum samples per split/leaf, and stopping criteria.
- AdaBoost using the documented weak-learner formulation.
- Class-weight handling or an appropriate strategy for class imbalance.
- Feature-importance and error analysis.

The target label must come from verified scenario metadata. Do not create labels from K-Means clusters and then claim that the classifier diagnoses real faults.

**Evaluation:**

- Accuracy, precision, recall, and F1-score.
- Per-class metrics and confusion matrix.
- Macro-F1 for imbalanced multi-class diagnosis.
- Balanced accuracy where class imbalance makes raw accuracy misleading.
- Comparison against a majority-class baseline and a single decision tree.

AdaBoost is not guaranteed to outperform every individual tree or baseline. The project will report measured results rather than promise a particular accuracy.

---

## 6. Evaluation protocol and leakage prevention

Time-series HVAC data can produce misleadingly high scores if adjacent or scenario-related records appear in both training and test sets.

Required safeguards:

1. Preserve timestamps and sort observations chronologically.
2. Split by time period and, where possible, by scenario/file or equipment unit.
3. Keep a final test set untouched until model selection is complete.
4. Fit imputers, scalers, PCA, feature selection, and other learned transformations on training data only.
5. Do not use fault labels, filename-derived labels, or future observations as input features.
6. Keep fault severity and scenario metadata out of the feature matrix unless a specific experiment explicitly justifies their use.
7. Report class counts and the split strategy.
8. Evaluate on a held-out scenario or equipment group when feasible to test generalisation.
9. For early-warning tasks, enforce a temporal gap between feature windows and future outcome windows.
10. Record random seeds, parameters, dataset version, and experiment configuration.

### Required model comparisons

| Component | Minimum comparison |
|---|---|
| Regression | Mean/naive baseline vs OLS vs Ridge |
| Anomaly detection | Residual threshold vs selected unsupervised method |
| Clustering | K-Means configurations; optional GMM |
| Sequence tracking | Raw state sequence vs HMM-decoded sequence |
| Fault classification | Majority baseline vs CART vs AdaBoost |

No accuracy, F1, RMSE, or other result should be written into the README until it has been produced by a reproducible run.

---

## 7. Explainability and diagnostic evidence

Every diagnostic result should provide evidence rather than only a label.

Where applicable, show:

- Observed sensor values and expected/baseline values.
- Regression residual and its recent trend.
- Anomaly score and the threshold or ranking method used.
- Cluster assignment and the sensor profile of that cluster.
- HMM state and transition probability/path context.
- Predicted fault class and class probabilities or model scores, where supported.
- Decision-tree rules or feature-importance summaries.
- Data-quality warnings and missing-sensor indicators.

Feature importance is not causal explanation. A high-importance sensor does not prove that the sensor or associated component caused the fault.

---

## 8. Proposed repository structure

The implementation should be modular and testable rather than one large notebook or one master script.

```text
hvac-faultsense/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
│
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   └── experiments.yaml
│
├── data/
│   ├── raw/                  # Downloaded source files; do not commit
│   ├── interim/              # Parsed/combined intermediate files
│   ├── processed/            # Validated model-ready data
│   └── README.md             # Download, provenance, schema and handling
│
├── docs/
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
│   ├── evaluation/
│   │   ├── evaluation_protocol.md
│   │   └── metrics_and_results.md
│   ├── architecture.md
│   └── decisions/
│
├── src/
│   └── hvac_faultsense/
│       ├── data/
│       │   ├── ingestion.py
│       │   ├── validation.py
│       │   ├── metadata.py
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
├── notebooks/
│   ├── 01_dataset_profiling.ipynb
│   └── 02_experiment_analysis.ipynb
│
├── experiments/
│   └── README.md
│
├── reports/
│   ├── figures/
│   └── results/
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

This is a proposed structure. It can be adjusted during implementation, but dataset documentation, algorithm documentation, tests, and experiment records are first-class project deliverables.

---

## 9. Documentation deliverables

The project must maintain the following Markdown documentation as implementation progresses.

| Document | What it must explain |
|---|---|
| `docs/dataset/dataset_overview.md` | Dataset source, license, equipment type, provenance, files, scenarios, sampling interval, download steps, and limitations |
| `docs/dataset/data_dictionary.md` | Every actual column, verified meaning, unit, type, missingness, and source of interpretation |
| `docs/dataset/data_quality_and_preprocessing.md` | Missing values, duplicates, outliers, timestamp handling, resampling, scaling, and split-safe preprocessing |
| `docs/dataset/label_mapping.md` | How source files/scenarios map to fault labels and severity; unresolved mappings must be explicit |
| `docs/algorithms/*.md` | Intuition, mathematical formulation, assumptions, implementation steps, complexity where useful, parameters, limitations, and tests |
| `docs/evaluation/evaluation_protocol.md` | Split strategy, leakage controls, baselines, metrics, and reproducibility |
| `docs/evaluation/metrics_and_results.md` | Actual experiment results, model comparisons, failure cases, and interpretation |
| `docs/architecture.md` | Data flow, module responsibilities, dependencies, and pipeline boundaries |

Documentation must describe the code that actually exists. Planned features should be marked as planned, not presented as implemented.

---

## 10. Technology choices

| Area | Proposed tools | Reason |
|---|---|---|
| Language | Python | Main ML and data ecosystem |
| Data processing | Pandas, NumPy, PyArrow | CSV ingestion, numerical implementation, optional Parquet |
| Visualisation | Matplotlib, Seaborn | EDA and diagnostic plots |
| Reference ML | scikit-learn | Baselines and independent implementation checks |
| Core syllabus algorithms | NumPy implementations | Demonstrate mathematical understanding |
| Testing | pytest | Unit, integration, and leakage tests |
| Notebooks | Jupyter / Google Colab | Exploratory analysis and demonstrations |
| Configuration | YAML | Reproducible experiment settings |
| Dashboard | Streamlit (optional) | Interactive diagnostic review |

For algorithms required to be implemented from scratch, the project should not call the corresponding scikit-learn estimator internally. Scikit-learn may be used for baselines, metrics, data splitting, and independent verification where appropriate. Any such use must be clearly identified.

---

## 11. Execution plan

The project should be built in verified milestones.

| Milestone | Scope | Completion evidence |
|---|---|---|
| M1 — Dataset foundation | Download, inventory, ingestion, schema validation | Dataset manifest, data dictionary, tests |
| M2 — Profiling | EDA, data quality, preprocessing | Reproducible profiling report |
| M3 — Regression | OLS and Ridge from scratch | Unit tests, metrics, residual analysis |
| M4 — Unsupervised learning | PCA, K-Means, optional GMM/EM | Projection, cluster analysis, tests |
| M5 — Sequential modelling | HMM and Viterbi | Decoded sequence and correctness tests |
| M6 — Fault diagnosis | CART and AdaBoost | Held-out evaluation and error analysis |
| M7 — Integration | End-to-end pipeline and diagnostic outputs | Reproducible run from raw data |
| M8 — Presentation | Dashboard, final report, documentation review | Demonstration and complete project docs |

Do not begin with the dashboard. First establish that the data, labels, splits, and model evaluations are correct.

---

## 12. Running the project

The exact commands will be finalised after the repository and environment are implemented. The intended workflow is:

1. Create and activate a Python virtual environment.
2. Install dependencies from the project dependency file.
3. Download the official LBNL RTU data and inventory using the official portal.
4. Place source files under `data/raw/` without renaming them unless a manifest preserves the original names.
5. Run the ingestion and validation pipeline.
6. Review the generated dataset manifest and data dictionary.
7. Run profiling and preprocessing.
8. Execute each model experiment using the documented configuration.
9. Run the test suite.
10. Generate evaluation reports and, optionally, launch the dashboard.

Example environment setup:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The final CLI commands should be documented once they exist. Do not document commands that have not been implemented and tested.

---

## 13. Definition of done

The project is considered complete only when:

- The dataset source, provenance, licensing, and limitations are documented.
- The actual data schema is discovered from source files rather than assumed.
- Fault labels and severity mappings are traceable to official documentation.
- Preprocessing and evaluation are time-aware and leakage-safe.
- Required from-scratch algorithms have independent unit tests.
- Reference baselines are used to verify the custom implementations.
- All models are compared using appropriate metrics.
- No performance claims are fabricated or based only on training data.
- The end-to-end pipeline can be reproduced from documented inputs.
- The dashboard and reports show evidence and limitations, not just predictions.
- All documentation matches the implemented behaviour.

---

## 14. Limitations and responsible use

- Simulated faults may not represent every real-world fault signature.
- Results from one RTU configuration may not generalise to other equipment, buildings, climates, or control strategies.
- Sensor drift, calibration errors, missing data, and maintenance interventions can affect predictions.
- Fault labels may describe injected or documented scenarios rather than every possible field failure.
- Unsupervised clusters are not ground-truth fault labels.
- A high anomaly score is an investigation signal, not a confirmed diagnosis.
- The HMM latent states are statistical states, not automatically physical health states.
- The system is advisory and requires human review before maintenance decisions.

---

## 15. References

1. **LBNL Fault Detection and Diagnostics datasets** — official dataset portal: <https://faultdetection.lbl.gov/data/>
2. **LBNL RTU dataset** — equipment description, files, sampling interval, and scenario inventory: <https://faultdetection.lbl.gov/dataset/rtu/>
3. **LBNL FDD dataset catalogue, Data.gov** — dataset metadata and access information: <https://catalog.data.gov/dataset/lbnl-fault-detection-and-diagnostics-datasets>
4. Granderson et al. (2023), *A labeled dataset for building HVAC systems operating in faulted and fault-free states*, Scientific Data: <https://www.nature.com/articles/s41597-023-02197-w>
5. Dataset DOI: <https://doi.org/10.25984/1881324>

---

**Development principle:** trustworthy data, explicit assumptions, mathematically correct implementations, leakage-safe evaluation, and evidence-backed conclusions take priority over the number of algorithms or dashboard features.
