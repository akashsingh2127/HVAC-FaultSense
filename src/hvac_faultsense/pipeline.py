"""HVAC FaultSense Master Pipeline Orchestrator.

Executes data profiling, leakage-safe preprocessing, from-scratch model training
(OLS, Ridge, PCA, K-Means, GMM, HMM-Viterbi, CART, AdaBoost), reference benchmarking,
and persists experiment artifacts for the Streamlit dashboard.
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.linear_model import LinearRegression as SklearnLinearRegression, Ridge as SklearnRidge
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture as SklearnGMM
from sklearn.tree import DecisionTreeClassifier as SklearnCART

from hvac_faultsense.data.ingestion import load_simulated_dataset
from hvac_faultsense.data.metadata import build_dataset_manifest, get_data_dictionary
from hvac_faultsense.data.preprocessing import (
    StandardScalerScratch,
    prepare_feature_matrices,
    split_time_and_scenario_aware,
)
from hvac_faultsense.data.validation import generate_profile_markdown, profile_dataset
from hvac_faultsense.evaluation.classification_metrics import (
    MajorityBaselineClassifier,
    evaluate_classification,
)
from hvac_faultsense.evaluation.regression_metrics import (
    MeanBaselineRegressor,
    evaluate_regression,
)
from hvac_faultsense.evaluation.validation import ModelComparisonRegistry, ModelResultRecord
from hvac_faultsense.features.engineering import compute_domain_features, get_feature_column_names
from hvac_faultsense.models.adaboost_classifier import AdaBoostClassifierScratch
from hvac_faultsense.models.cart_classifier import DecisionTreeClassifierScratch
from hvac_faultsense.models.gmm_em_scratch import GaussianMixtureScratch
from hvac_faultsense.models.hmm_viterbi_scratch import DiscreteHMMViterbiScratch
from hvac_faultsense.models.kmeans_scratch import KMeansScratch
from hvac_faultsense.models.pca_scratch import PCAScratch
from hvac_faultsense.models.regression_scratch import LinearRegressionScratch
from hvac_faultsense.models.ridge_scratch import RidgeRegressionScratch


def run_pipeline(
    data_dir: Path,
    reports_dir: Path,
    experiments_dir: Path,
    stride: int = 10, # Subsampling factor for fast, representative execution
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end HVAC FaultSense pipeline."""
    print("=" * 70)
    print("HVAC FaultSense: Executing End-to-End Analytical Pipeline")
    print("=" * 70)

    # 1. Manifest and Data Dictionary
    print("\n[Step 1/8] Generating Dataset Manifest & Data Dictionary...")
    manifest_csv = data_dir / "interim" / "dataset_manifest.csv"
    dict_csv = data_dir / "interim" / "data_dictionary.csv"
    manifest = build_dataset_manifest(
        raw_sim_dir=data_dir / "raw" / "simulated",
        raw_field_dir=data_dir / "raw" / "field",
        raw_lab_dir=data_dir / "raw" / "laboratory",
        output_csv=manifest_csv,
    )
    data_dict = get_data_dictionary(output_csv=dict_csv)

    # 2. Ingestion
    print(f"\n[Step 2/8] Ingesting Simulated RTU Dataset (stride={stride})...")
    raw_sim_dir = data_dir / "raw" / "simulated"
    df_raw = load_simulated_dataset(raw_sim_dir, stride=stride)
    print(f"Loaded {len(df_raw):,} records across {df_raw['scenario_name'].nunique()} scenarios.")

    # 3. Profiling
    print("\n[Step 3/8] Running Statistical Profiling and Quality Audit...")
    profile = profile_dataset(df_raw)
    profile_md = generate_profile_markdown(profile)
    with open(reports_dir / "results" / "data_profile.json", "w") as f:
        json.dump(profile, f, indent=2)
    with open(reports_dir / "results" / "data_profile.md", "w") as f:
        f.write(profile_md)
    print("Data profile saved to reports/results/data_profile.md.")

    # 4. Feature Engineering
    print("\n[Step 4/8] Computing Domain Thermodynamic Features...")
    df_engineered = compute_domain_features(df_raw)
    feature_cols = get_feature_column_names(df_engineered)
    target_col = "RTU_TOT_WATT"
    label_col = "fault_type"
    print(f"Engineered feature set contains {len(feature_cols)} predictor features.")

    # 5. Leakage-Safe Splitting
    print("\n[Step 5/8] Splitting Train and Test Sets (Chronological & Scenario-Aware)...")
    df_train, df_test = split_time_and_scenario_aware(df_engineered, test_ratio=0.25)
    print(f"Training samples: {len(df_train):,} | Held-out test samples: {len(df_test):,}")

    X_train_scaled, X_test_scaled, y_train_reg, y_test_reg, y_train_cls, y_test_cls, scaler = prepare_feature_matrices(
        df_train, df_test, feature_cols, target_col=target_col, label_col=label_col
    )

    registry = ModelComparisonRegistry()
    trained_artifacts = {
        "scaler": scaler,
        "feature_cols": feature_cols,
        "classes": np.unique(y_train_cls).tolist(),
    }

    # 6. Unit 2: Regression Experiments
    print("\n[Step 6/8] Unit 2: Evaluating OLS and Ridge Regression from Scratch...")

    # A. Mean Baseline
    t0 = time.time()
    baseline_reg = MeanBaselineRegressor().fit(X_train_scaled, y_train_reg)
    t_base = time.time() - t0
    y_pred_base = baseline_reg.predict(X_test_scaled)
    m_base = evaluate_regression(y_test_reg, y_pred_base)
    registry.log_result(
        ModelResultRecord(
            model_name="Mean Baseline",
            task="Regression",
            mae=m_base["MAE"],
            rmse=m_base["RMSE"],
            r2=m_base["R2"],
            training_time_sec=t_base,
            n_train_samples=len(X_train_scaled),
            n_test_samples=len(X_test_scaled),
        )
    )

    # B. OLS Linear Regression Scratch
    t0 = time.time()
    ols_scratch = LinearRegressionScratch(learning_rate=0.02, max_iter=1500, solver="gradient_descent").fit(
        X_train_scaled, y_train_reg
    )
    t_ols = time.time() - t0
    y_pred_ols = ols_scratch.predict(X_test_scaled)
    m_ols = evaluate_regression(y_test_reg, y_pred_ols)
    registry.log_result(
        ModelResultRecord(
            model_name="OLS Regression (Scratch)",
            task="Regression",
            mae=m_ols["MAE"],
            rmse=m_ols["RMSE"],
            r2=m_ols["R2"],
            training_time_sec=t_ols,
            n_train_samples=len(X_train_scaled),
            n_test_samples=len(X_test_scaled),
            hyperparameters={"learning_rate": 0.02, "max_iter": 1500},
        )
    )
    trained_artifacts["ols_scratch"] = ols_scratch

    # C. Ridge Regression Scratch
    t0 = time.time()
    ridge_scratch = RidgeRegressionScratch(lambda_param=1.0, learning_rate=0.02, max_iter=1500).fit(
        X_train_scaled, y_train_reg
    )
    t_ridge = time.time() - t0
    y_pred_ridge = ridge_scratch.predict(X_test_scaled)
    m_ridge = evaluate_regression(y_test_reg, y_pred_ridge)
    registry.log_result(
        ModelResultRecord(
            model_name="Ridge Regression (Scratch)",
            task="Regression",
            mae=m_ridge["MAE"],
            rmse=m_ridge["RMSE"],
            r2=m_ridge["R2"],
            training_time_sec=t_ridge,
            n_train_samples=len(X_train_scaled),
            n_test_samples=len(X_test_scaled),
            hyperparameters={"lambda_param": 1.0, "learning_rate": 0.02},
        )
    )
    trained_artifacts["ridge_scratch"] = ridge_scratch

    # D. Scikit-learn Linear Regression (Reference)
    t0 = time.time()
    sk_lr = SklearnLinearRegression().fit(X_train_scaled, y_train_reg)
    t_sk_lr = time.time() - t0
    m_sk_lr = evaluate_regression(y_test_reg, sk_lr.predict(X_test_scaled))
    registry.log_result(
        ModelResultRecord(
            model_name="Linear Regression (Reference)",
            task="Regression",
            mae=m_sk_lr["MAE"],
            rmse=m_sk_lr["RMSE"],
            r2=m_sk_lr["R2"],
            training_time_sec=t_sk_lr,
            n_train_samples=len(X_train_scaled),
            n_test_samples=len(X_test_scaled),
        )
    )

    # 7. Unit 3 & 4: Unsupervised Learning & Sequential Tracking
    print("\n[Step 7/8] Unit 3 & 4: Evaluating PCA, K-Means, GMM, and HMM Viterbi from Scratch...")

    # PCA from Scratch
    pca_scratch = PCAScratch(n_components=2).fit(X_train_scaled)
    X_train_pca = pca_scratch.transform(X_train_scaled)
    X_test_pca = pca_scratch.transform(X_test_scaled)
    trained_artifacts["pca_scratch"] = pca_scratch
    print(f"PCA explained variance ratio: PC1={pca_scratch.explained_variance_ratio_[0]:.2%}, PC2={pca_scratch.explained_variance_ratio_[1]:.2%}")

    # Subsample for fast clustering metrics if test set is large
    eval_subsample_idx = np.random.default_rng(random_seed).choice(
        len(X_test_pca), size=min(3000, len(X_test_pca)), replace=False
    )
    X_sub_eval = X_test_pca[eval_subsample_idx]

    # K-Means from Scratch
    t0 = time.time()
    kmeans_scratch = KMeansScratch(n_clusters=4, max_iter=80, n_init=3, random_state=random_seed).fit(X_train_pca)
    t_km = time.time() - t0
    km_test_labels = kmeans_scratch.predict(X_test_pca)
    km_sub_labels = km_test_labels[eval_subsample_idx]
    sil_km = float(silhouette_score(X_sub_eval, km_sub_labels))
    registry.log_result(
        ModelResultRecord(
            model_name="K-Means (Scratch)",
            task="Clustering",
            silhouette=sil_km,
            training_time_sec=t_km,
            n_train_samples=len(X_train_pca),
            n_test_samples=len(X_test_pca),
            hyperparameters={"n_clusters": 4, "init": "k-means++"},
        )
    )
    trained_artifacts["kmeans_scratch"] = kmeans_scratch

    # GMM from Scratch
    t0 = time.time()
    gmm_scratch = GaussianMixtureScratch(n_components=4, max_iter=40, random_state=random_seed).fit(X_train_pca)
    t_gmm = time.time() - t0
    gmm_test_labels = gmm_scratch.predict(X_test_pca)
    gmm_sub_labels = gmm_test_labels[eval_subsample_idx]
    sil_gmm = float(silhouette_score(X_sub_eval, gmm_sub_labels))
    registry.log_result(
        ModelResultRecord(
            model_name="GMM / EM (Scratch)",
            task="Clustering",
            silhouette=sil_gmm,
            training_time_sec=t_gmm,
            n_train_samples=len(X_train_pca),
            n_test_samples=len(X_test_pca),
            hyperparameters={"n_components": 4, "reg_covar": 1e-4},
        )
    )
    trained_artifacts["gmm_scratch"] = gmm_scratch

    # HMM Viterbi from Scratch
    hmm_scratch = DiscreteHMMViterbiScratch(n_states=4, n_observations=4).initialize_uniform_or_random(random_state=random_seed)
    # Decode state sequence for a continuous slice of held-out test data
    test_slice_len = min(2000, len(km_test_labels))
    raw_obs_slice = km_test_labels[:test_slice_len]
    decoded_states, log_prob = hmm_scratch.viterbi_decode(raw_obs_slice)
    trained_artifacts["hmm_scratch"] = hmm_scratch
    print(f"HMM decoded {test_slice_len} observations into optimal state path (log_prob = {log_prob:.2f}).")

    # 8. Unit 5: Fault Classification Experiments
    print("\n[Step 8/8] Unit 5: Evaluating CART Decision Tree and AdaBoost from Scratch...")

    # A. Majority Baseline
    t0 = time.time()
    majority_cls = MajorityBaselineClassifier().fit(X_train_scaled, y_train_cls)
    t_maj = time.time() - t0
    m_maj = evaluate_classification(y_test_cls, majority_cls.predict(X_test_scaled))
    registry.log_result(
        ModelResultRecord(
            model_name="Majority Class Baseline",
            task="Classification",
            accuracy=m_maj["Accuracy"],
            precision=m_maj["Precision"],
            recall=m_maj["Recall"],
            macro_f1=m_maj["Macro-F1"],
            training_time_sec=t_maj,
            n_train_samples=len(X_train_scaled),
            n_test_samples=len(X_test_scaled),
        )
    )

    # B. CART Decision Tree from Scratch
    t0 = time.time()
    cart_scratch = DecisionTreeClassifierScratch(max_depth=6, min_samples_split=10, min_samples_leaf=5).fit(
        X_train_scaled, y_train_cls
    )
    t_cart = time.time() - t0
    y_pred_cart = cart_scratch.predict(X_test_scaled)
    m_cart = evaluate_classification(y_test_cls, y_pred_cart)
    registry.log_result(
        ModelResultRecord(
            model_name="CART Decision Tree (Scratch)",
            task="Classification",
            accuracy=m_cart["Accuracy"],
            precision=m_cart["Precision"],
            recall=m_cart["Recall"],
            macro_f1=m_cart["Macro-F1"],
            training_time_sec=t_cart,
            n_train_samples=len(X_train_scaled),
            n_test_samples=len(X_test_scaled),
            hyperparameters={"max_depth": 6, "criterion": "gini"},
        )
    )
    trained_artifacts["cart_scratch"] = cart_scratch

    # C. AdaBoost from Scratch (SAMME multi-class)
    t0 = time.time()
    adaboost_scratch = AdaBoostClassifierScratch(n_estimators=15, learning_rate=0.5, weak_learner_max_depth=2).fit(
        X_train_scaled, y_train_cls
    )
    t_ada = time.time() - t0
    y_pred_ada = adaboost_scratch.predict(X_test_scaled)
    m_ada = evaluate_classification(y_test_cls, y_pred_ada)
    registry.log_result(
        ModelResultRecord(
            model_name="AdaBoost SAMME (Scratch)",
            task="Classification",
            accuracy=m_ada["Accuracy"],
            precision=m_ada["Precision"],
            recall=m_ada["Recall"],
            macro_f1=m_ada["Macro-F1"],
            training_time_sec=t_ada,
            n_train_samples=len(X_train_scaled),
            n_test_samples=len(X_test_scaled),
            hyperparameters={"n_estimators": 15, "learning_rate": 0.5},
        )
    )
    trained_artifacts["adaboost_scratch"] = adaboost_scratch

    # D. Scikit-learn Reference Models
    t0 = time.time()
    sk_cart = SklearnCART(max_depth=6, random_state=random_seed).fit(X_train_scaled, y_train_cls)
    t_sk_c = time.time() - t0
    m_sk_c = evaluate_classification(y_test_cls, sk_cart.predict(X_test_scaled))
    registry.log_result(
        ModelResultRecord(
            model_name="CART Decision Tree (Reference)",
            task="Classification",
            accuracy=m_sk_c["Accuracy"],
            precision=m_sk_c["Precision"],
            recall=m_sk_c["Recall"],
            macro_f1=m_sk_c["Macro-F1"],
            training_time_sec=t_sk_c,
            n_train_samples=len(X_train_scaled),
            n_test_samples=len(X_test_scaled),
        )
    )

    t0 = time.time()
    sk_ada = SklearnAdaBoost(n_estimators=15, random_state=random_seed).fit(X_train_scaled, y_train_cls)
    t_sk_a = time.time() - t0
    m_sk_a = evaluate_classification(y_test_cls, sk_ada.predict(X_test_scaled))
    registry.log_result(
        ModelResultRecord(
            model_name="AdaBoost (Reference)",
            task="Classification",
            accuracy=m_sk_a["Accuracy"],
            precision=m_sk_a["Precision"],
            recall=m_sk_a["Recall"],
            macro_f1=m_sk_a["Macro-F1"],
            training_time_sec=t_sk_a,
            n_train_samples=len(X_train_scaled),
            n_test_samples=len(X_test_scaled),
        )
    )

    # 9. Consolidation and Persistence
    print("\n[Consolidation] Saving Experiment Results, Benchmark Table & Artifacts...")
    df_comparison = registry.to_dataframe()
    print("\n" + df_comparison.to_string(index=False))

    bench_csv = reports_dir / "results" / "benchmark_summary.csv"
    df_comparison.to_csv(bench_csv, index=False)

    best_models = registry.identify_best_models()
    print("\n=== Best Performing Models by Task ===")
    for task_name, info in best_models.items():
        print(f"Task: {task_name:15} | Winner: {info['winner']:30} | {info['reason']}")

    results_payload = {
        "benchmark_summary": df_comparison.to_dict(orient="records"),
        "best_models": best_models,
        "classification_metrics": {
            "cart": m_cart,
            "adaboost": m_ada,
        },
        "regression_metrics": {
            "ols": m_ols,
            "ridge": m_ridge,
            "baseline": m_base,
        },
        "pca_explained_variance_ratio": pca_scratch.explained_variance_ratio_.tolist(),
        "clustering": {
            "kmeans_silhouette": sil_km,
            "gmm_silhouette": sil_gmm,
        },
    }

    with open(reports_dir / "results" / "experiment_results.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    # Save trained objects for Streamlit dashboard fast loading
    with open(experiments_dir / "trained_pipeline.pkl", "wb") as f:
        pickle.dump(trained_artifacts, f)

    # Save sample test dataframe with predictions and residuals for fast visual dashboard inspection
    df_test_sample = df_test.iloc[:2000].copy()
    df_test_sample["pred_power_ols"] = y_pred_ols[:2000]
    df_test_sample["pred_power_ridge"] = y_pred_ridge[:2000]
    df_test_sample["residual_ols"] = df_test_sample["RTU_TOT_WATT"] - df_test_sample["pred_power_ols"]
    df_test_sample["pc1"] = X_test_pca[:2000, 0]
    df_test_sample["pc2"] = X_test_pca[:2000, 1]
    df_test_sample["kmeans_cluster"] = km_test_labels[:2000]
    df_test_sample["gmm_cluster"] = gmm_test_labels[:2000]
    df_test_sample["hmm_state"] = decoded_states[:2000]
    df_test_sample["pred_fault_cart"] = y_pred_cart[:2000]
    df_test_sample["pred_fault_ada"] = y_pred_ada[:2000]

    try:
        df_test_sample.to_parquet(data_dir / "processed" / "test_evaluation_sample.parquet", index=False)
        print("Test evaluation sample saved to data/processed/test_evaluation_sample.parquet.")
    except Exception as e:
        print(f"Notice regarding parquet export: {e}")
    df_test_sample.to_csv(data_dir / "processed" / "test_evaluation_sample.csv", index=False)
    print("Test evaluation sample saved to data/processed/test_evaluation_sample.csv.")
    print("=" * 70)
    print("Pipeline Execution Completed Successfully!")
    print("=" * 70)

    return results_payload


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    run_pipeline(
        data_dir=base_dir / "data",
        reports_dir=base_dir / "reports",
        experiments_dir=base_dir / "experiments",
        stride=15, # 15-minute stride for smooth, fast, and balanced training
    )
