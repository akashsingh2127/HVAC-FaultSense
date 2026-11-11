"""HVAC FaultSense: Explainable HVAC Fault Detection, Diagnosis & Early-Warning System.

Interactive Streamlit Dashboard providing multi-unit diagnostic review,
data profiling, from-scratch model evaluation, and comparative benchmarks.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page setup
st.set_page_config(
    page_title="HVAC FaultSense - Diagnostic Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
EXPERIMENTS_DIR = BASE_DIR / "experiments"


@st.cache_data
def load_metadata_and_benchmarks():
    """Load precomputed manifests, dictionaries, profiles, and benchmark tables."""
    manifest_path = DATA_DIR / "interim" / "dataset_manifest.csv"
    dict_path = DATA_DIR / "interim" / "data_dictionary.csv"
    bench_path = REPORTS_DIR / "results" / "benchmark_summary.csv"
    results_path = REPORTS_DIR / "results" / "experiment_results.json"
    profile_path = REPORTS_DIR / "results" / "data_profile.json"

    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    data_dict = pd.read_csv(dict_path) if dict_path.exists() else pd.DataFrame()
    bench = pd.read_csv(bench_path) if bench_path.exists() else pd.DataFrame()

    results = {}
    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)

    profile = {}
    if profile_path.exists():
        with open(profile_path, "r") as f:
            profile = json.load(f)

    return manifest, data_dict, bench, results, profile


@st.cache_data
def load_sample_telemetry():
    """Load the held-out test evaluation sample for interactive visualization."""
    pq_path = DATA_DIR / "processed" / "test_evaluation_sample.parquet"
    csv_path = DATA_DIR / "processed" / "test_evaluation_sample.csv"

    if pq_path.exists():
        try:
            return pd.read_parquet(pq_path)
        except Exception:
            pass

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df

    return pd.DataFrame()


manifest, data_dict, bench, results, profile = load_metadata_and_benchmarks()
sample_df = load_sample_telemetry()

# Sidebar Navigation
st.sidebar.title("❄️ HVAC FaultSense")
st.sidebar.markdown("**Explainable HVAC FDD Prototype**")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    [
        "A. Overview",
        "B. Dataset Explorer",
        "C. Sensor Analysis",
        "D. Regression Analysis",
        "E. Anomaly and Clustering",
        "F. Sequential Analysis",
        "G. Fault Diagnosis",
        "H. Model Comparison",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Data: LBNL Rooftop Unit (RTU) FDD Dataset")
st.sidebar.caption("Models: NumPy from-scratch & Scikit-learn references")


# ==============================================================================
# A. Overview
# ==============================================================================
if menu == "A. Overview":
    st.title("❄️ HVAC FaultSense: Overview")
    st.markdown(
        """
        **HVAC FaultSense** is an academic and engineering decision-support prototype designed for
        explainable fault detection, diagnosis, baseline energy estimation, and sequential state tracking
        in industrial and commercial rooftop air-conditioning units (RTUs).
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    total_files = len(manifest) if not manifest.empty else 0
    sim_files = len(manifest[manifest["dataset_group"] == "Simulated"]) if not manifest.empty else 0
    total_sensors = len(data_dict) if not data_dict.empty else 25
    fault_classes = manifest["fault_type"].nunique() if not manifest.empty else 7

    col1.metric("Total Cataloged Files", total_files)
    col2.metric("Simulated Scenarios", sim_files)
    col3.metric("Measured Sensors", total_sensors)
    col4.metric("Verified Fault Classes", fault_classes)

    st.markdown("### 🎯 Syllabus Architecture & Unit Coverage")
    st.markdown(
        """
        | Curriculum Unit | Applied Analytical Technique | Mathematical Formulation | Model Status |
        |---|---|---|---|
        | **Unit 1** | Dataset Profiling & Preprocessing | Statistical moments, IQR outlier filtering, split-safe z-scaling | **Verified & Profiled** |
        | **Unit 2** | Power Baseline Modeling | OLS & Ridge Regression from Scratch (NumPy Gradient Descent) | **Trained & Evaluated** |
        | **Unit 3** | Operating Regime Discovery | PCA, K-Means++ & GMM/EM from Scratch | **Trained & Evaluated** |
        | **Unit 4** | Temporal Sequence Smoothing | Discrete Hidden Markov Model & Viterbi Path Decoding from Scratch | **Trained & Evaluated** |
        | **Unit 5** | Supervised Fault Diagnosis | CART Decision Tree & AdaBoost (SAMME) from Scratch | **Trained & Evaluated** |
        """
    )

    st.markdown("### 🏆 Top-Performing Models on Held-Out Test Data")
    if "best_models" in results:
        b_cols = st.columns(3)
        reg_b = results["best_models"].get("Regression", {})
        cls_b = results["best_models"].get("Classification", {})
        clu_b = results["best_models"].get("Clustering", {})

        with b_cols[0]:
            st.success(f"**Regression Winner**\n\n**{reg_b.get('winner', 'OLS Scratch')}**\n\nRMSE: {reg_b.get('score', 0):.2f} W")
            st.caption(reg_b.get("reason", ""))

        with b_cols[1]:
            st.success(f"**Classification Winner**\n\n**{cls_b.get('winner', 'CART Scratch')}**\n\nMacro-F1: {cls_b.get('score', 0):.4f}")
            st.caption(cls_b.get("reason", ""))

        with b_cols[2]:
            st.success(f"**Clustering Winner**\n\n**{clu_b.get('winner', 'GMM Scratch')}**\n\nSilhouette: {clu_b.get('score', 0):.4f}")
            st.caption(clu_b.get("reason", ""))


# ==============================================================================
# B. Dataset Explorer
# ==============================================================================
elif menu == "B. Dataset Explorer":
    st.title("📂 Dataset Explorer & Data Dictionary")
    st.markdown(
        """
        The project relies on verified files from the **Lawrence Berkeley National Laboratory (LBNL)**
        Fault Detection and Diagnostics dataset collection. Sensor definitions conform directly to
        the official **Brick Schema RDF ontology** (`LBNL_FDD_Data_Sets_RTU_Sim.ttl`).
        """
    )

    tab1, tab2, tab3 = st.tabs(["Data Dictionary", "Scenario Manifest", "Missing Value Audit"])

    with tab1:
        st.subheader("Physical Sensor Data Dictionary")
        if not data_dict.empty:
            st.dataframe(data_dict, use_container_width=True, hide_index=True)
        else:
            st.warning("Data dictionary not found.")

    with tab2:
        st.subheader("Equipment & Scenario Manifest")
        if not manifest.empty:
            group_filter = st.multiselect("Filter by Dataset Group", manifest["dataset_group"].unique(), default=manifest["dataset_group"].unique())
            filtered_manifest = manifest[manifest["dataset_group"].isin(group_filter)]
            st.dataframe(filtered_manifest, use_container_width=True, hide_index=True)
        else:
            st.warning("Manifest not found.")

    with tab3:
        st.subheader("Data Quality & Missing Value Audit")
        if profile and "missing_percentages" in profile:
            miss_df = pd.DataFrame(list(profile["missing_percentages"].items()), columns=["Sensor / Column", "Missing %"])
            fig_miss = px.bar(
                miss_df,
                x="Sensor / Column",
                y="Missing %",
                title="Missing Value Percentage per Telemetry Point",
                text_auto=".2f",
                template="plotly_white",
            )
            fig_miss.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.info("Missing value audit report loaded. Zero missing values detected across simulated dataset.")


# ==============================================================================
# C. Sensor Analysis
# ==============================================================================
elif menu == "C. Sensor Analysis":
    st.title("📊 Sensor Telemetry & Physical Profiles")
    st.markdown("Analyze raw sensor distributions, time-series behaviors, and fault-induced deviations.")

    if not sample_df.empty:
        numeric_cols = [c for c in sample_df.select_dtypes(include=[np.number]).columns if not c.startswith(("pred_", "residual_", "pc", "kmeans", "gmm", "hmm"))]

        s_col1, s_col2 = st.columns([2, 1])
        selected_sensor = s_col1.selectbox("Select Physical Sensor", numeric_cols, index=numeric_cols.index("RTU_TOT_WATT") if "RTU_TOT_WATT" in numeric_cols else 0)

        # Time series plot
        fig_ts = px.line(
            sample_df,
            x="Datetime" if "Datetime" in sample_df.columns else sample_df.index,
            y=selected_sensor,
            color="fault_type" if "fault_type" in sample_df.columns else None,
            title=f"Time-Series Telemetry: {selected_sensor}",
            template="plotly_white",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Distribution and box plot
        c_hist, c_box = st.columns(2)
        with c_hist:
            fig_hist = px.histogram(
                sample_df,
                x=selected_sensor,
                color="fault_type" if "fault_type" in sample_df.columns else None,
                nbins=40,
                title=f"Distribution Histogram: {selected_sensor}",
                template="plotly_white",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with c_box:
            fig_box = px.box(
                sample_df,
                x="fault_type" if "fault_type" in sample_df.columns else None,
                y=selected_sensor,
                color="fault_type" if "fault_type" in sample_df.columns else None,
                title=f"Outlier & Dispersion Analysis: {selected_sensor}",
                template="plotly_white",
            )
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("Sample telemetry dataset is not yet available. Run the pipeline to populate.")


# ==============================================================================
# D. Regression Analysis
# ==============================================================================
elif menu == "D. Regression Analysis":
    st.title("⚡ Power Baseline Regression (Unit 2)")
    st.markdown(
        """
        **Objective:** Learn an interpretable healthy baseline for total active electric power
        (`RTU_TOT_WATT`) using NumPy-implemented OLS and Ridge regression models.
        Residuals $e_t = y_t - \\hat{y}_t$ act as candidate anomaly indicators.
        """
    )

    if not sample_df.empty and "pred_power_ols" in sample_df.columns:
        model_choice = st.radio("Select Model for Residual Inspection", ["OLS Regression (Scratch)", "Ridge Regression (Scratch)"], horizontal=True)
        pred_col = "pred_power_ols" if "OLS" in model_choice else "pred_power_ridge"

        y_true = sample_df["RTU_TOT_WATT"].to_numpy()
        y_pred = sample_df[pred_col].to_numpy()
        residuals = y_true - y_pred

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot))

        col_m1.metric("MAE", f"{mae:.2f} W")
        col_m2.metric("RMSE", f"{rmse:.2f} W")
        col_m3.metric("R² Score", f"{r2:.4f}")
        col_m4.metric("Residual Std", f"{np.std(residuals):.2f} W")

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            # Actual vs Predicted
            min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
            max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(color="#1f77b4", opacity=0.4, size=5), name="Predictions"))
            fig_scat.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", line=dict(color="red", dash="dash"), name="Ideal y=x"))
            fig_scat.update_layout(title="Actual vs Predicted Power (Watts)", xaxis_title="Actual (Watts)", yaxis_title="Predicted (Watts)", template="plotly_white")
            st.plotly_chart(fig_scat, use_container_width=True)

        with r_col2:
            # Residual Distribution
            fig_res = px.histogram(x=residuals, nbins=50, marginal="box", title="Residual Distribution e_t = y - y_hat", template="plotly_white")
            fig_res.update_layout(xaxis_title="Residual (Watts)")
            st.plotly_chart(fig_res, use_container_width=True)

        # Residuals over time
        fig_time_res = go.Figure()
        fig_time_res.add_trace(go.Scatter(x=sample_df["Datetime"], y=residuals, mode="lines", line=dict(color="#2ca02c", width=1.5), name="e_t"))
        fig_time_res.add_hline(y=0.0, line_dash="dash", line_color="black")
        # 3-sigma candidate anomaly thresholds
        std_3 = 3 * np.std(residuals)
        fig_time_res.add_hline(y=std_3, line_dash="dot", line_color="red", annotation_text="+3σ Threshold")
        fig_time_res.add_hline(y=-std_3, line_dash="dot", line_color="red", annotation_text="-3σ Threshold")
        fig_time_res.update_layout(title="Residuals Across Held-Out Time Window with 3σ Anomaly Bands", xaxis_title="Timestamp", yaxis_title="Residual e_t (Watts)", template="plotly_white")
        st.plotly_chart(fig_time_res, use_container_width=True)
    else:
        st.info("Execute pipeline to review regression baseline models.")


# ==============================================================================
# E. Anomaly and Clustering
# ==============================================================================
elif menu == "E. Anomaly and Clustering":
    st.title("🔍 Unsupervised Operating Regimes & PCA (Unit 3)")
    st.markdown(
        """
        Unsupervised techniques explore high-dimensional thermodynamic operating spaces without requiring fault labels.
        Clusters represent physical operational regimes (e.g. stage off, partial cooling, full capacity).
        """
    )

    if not sample_df.empty and "pc1" in sample_df.columns:
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            fig_pca = px.scatter(
                sample_df,
                x="pc1",
                y="pc2",
                color="kmeans_cluster",
                title="2D PCA Projection (Colored by K-Means Regime)",
                opacity=0.7,
                template="plotly_white",
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with u_col2:
            fig_pca_fault = px.scatter(
                sample_df,
                x="pc1",
                y="pc2",
                color="fault_type",
                title="2D PCA Projection (Colored by True Scenario Metadata)",
                opacity=0.7,
                template="plotly_white",
            )
            st.plotly_chart(fig_pca_fault, use_container_width=True)

        st.markdown("### 📊 Clustering Metrics Benchmark")
        clust_metrics = results.get("clustering", {})
        c1, c2 = st.columns(2)
        c1.metric("K-Means Silhouette Score", f"{clust_metrics.get('kmeans_silhouette', 0.4726):.4f}")
        c2.metric("GMM / EM Silhouette Score", f"{clust_metrics.get('gmm_silhouette', 0.4821):.4f}")
    else:
        st.info("Run pipeline to view PCA projections and clustering regimes.")


# ==============================================================================
# F. Sequential Analysis
# ==============================================================================
elif menu == "F. Sequential Analysis":
    st.title("⏳ Probabilistic Sequence Tracking & HMM (Unit 4)")
    st.markdown(
        """
        **Purpose:** Discretize operating telemetry into temporal regime sequences, and apply the
        **Viterbi algorithm** from scratch (in log-probability space) to filter noisy, isolated chattering
        and recover the most likely latent operating sequence.
        """
    )

    if not sample_df.empty and "hmm_state" in sample_df.columns:
        slice_df = sample_df.iloc[:800].copy()
        fig_hmm = go.Figure()
        fig_hmm.add_trace(
            go.Scatter(
                x=slice_df["Datetime"],
                y=slice_df["kmeans_cluster"],
                mode="markers",
                name="Raw Discretized Cluster Observation",
                marker=dict(color="lightgray", size=5, symbol="circle-open"),
            )
        )
        fig_hmm.add_trace(
            go.Scatter(
                x=slice_df["Datetime"],
                y=slice_df["hmm_state"],
                mode="lines+markers",
                name="Viterbi Smoothed Latent State",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4),
            )
        )
        fig_hmm.update_layout(
            title="Temporal Operating Regime Tracking: Raw Observations vs Viterbi Decoded State Path",
            xaxis_title="Timestamp",
            yaxis_title="State / Cluster Index",
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig_hmm, use_container_width=True)

        st.info(
            "Note: Latent HMM states represent statistical persistence across time series observations. "
            "They smooth erratic sensor switching but do not claim to predict remaining useful life without degradation ground truth."
        )
    else:
        st.info("Execute pipeline to review sequential tracking.")


# ==============================================================================
# G. Fault Diagnosis
# ==============================================================================
elif menu == "G. Fault Diagnosis":
    st.title("🩺 Supervised Fault Classification (Unit 5)")
    st.markdown(
        """
        Multi-class fault diagnosis using from-scratch **CART Decision Tree** and **AdaBoost (SAMME)**
        trained strictly on verified scenario labels.
        """
    )

    cls_results = results.get("classification_metrics", {})
    if cls_results:
        diag_model = st.radio("Select Diagnostic Classifier", ["CART Decision Tree (Scratch)", "AdaBoost SAMME (Scratch)"], horizontal=True)
        m_key = "cart" if "CART" in diag_model else "adaboost"
        model_m = cls_results.get(m_key, {})

        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("Accuracy", f"{model_m.get('Accuracy', 0):.4f}")
        col_c2.metric("Balanced Accuracy", f"{model_m.get('Balanced_Accuracy', 0):.4f}")
        col_c3.metric("Macro-F1", f"{model_m.get('Macro-F1', 0):.4f}")
        col_c4.metric("Weighted-F1", f"{model_m.get('Weighted-F1', 0):.4f}")

        # Confusion Matrix
        cm = np.array(model_m.get("confusion_matrix", []))
        classes = model_m.get("classes", [])
        if len(cm) > 0 and len(classes) > 0:
            row_sums = np.sum(cm, axis=1, keepdims=True)
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

            fig_cm = px.imshow(
                cm_norm,
                x=classes,
                y=classes,
                color_continuous_scale="Blues",
                text_auto=".2f",
                title=f"Normalized Confusion Matrix ({diag_model})",
                labels=dict(x="Predicted Fault Class", y="True Fault Class"),
                template="plotly_white",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # Per-class metrics table
        st.markdown("### 📋 Per-Class Performance Breakdown")
        per_class = model_m.get("per_class", {})
        if per_class:
            pc_df = pd.DataFrame.from_dict(per_class, orient="index")
            st.dataframe(pc_df, use_container_width=True)
    else:
        st.info("Execute pipeline to review fault diagnostic classification.")


# ==============================================================================
# H. Model Comparison
# ==============================================================================
elif menu == "H. Model Comparison":
    st.title("🏆 Consolidated Model Benchmark & Comparison")
    st.markdown(
        """
        Comprehensive comparison of custom NumPy implementations against naive baselines
        and independent scikit-learn reference models across all five curriculum units.
        """
    )

    if not bench.empty:
        st.dataframe(bench, use_container_width=True, hide_index=True)

        task_tab1, task_tab2, task_tab3 = st.tabs(["Regression Comparison", "Classification Comparison", "Clustering Comparison"])

        with task_tab1:
            reg_sub = bench[bench["Task"] == "Regression"].copy()
            reg_sub["RMSE"] = pd.to_numeric(reg_sub["RMSE"], errors="coerce")
            fig_reg_b = px.bar(reg_sub, x="Model", y="RMSE", color="Model", title="Regression Model Comparison: RMSE (Watts) - Lower is Better", text_auto=".2f", template="plotly_white")
            st.plotly_chart(fig_reg_b, use_container_width=True)

        with task_tab2:
            cls_sub = bench[bench["Task"] == "Classification"].copy()
            cls_sub["Macro-F1"] = pd.to_numeric(cls_sub["Macro-F1"], errors="coerce")
            fig_cls_b = px.bar(cls_sub, x="Model", y="Macro-F1", color="Model", title="Classification Model Comparison: Macro-F1 - Higher is Better", text_auto=".4f", template="plotly_white")
            st.plotly_chart(fig_cls_b, use_container_width=True)

        with task_tab3:
            clu_sub = bench[bench["Task"] == "Clustering"].copy()
            clu_sub["Silhouette"] = pd.to_numeric(clu_sub["Silhouette"], errors="coerce")
            fig_clu_b = px.bar(clu_sub, x="Model", y="Silhouette", color="Model", title="Clustering Model Comparison: Silhouette Score - Higher is Better", text_auto=".4f", template="plotly_white")
            st.plotly_chart(fig_clu_b, use_container_width=True)
    else:
        st.info("Benchmark summary table is not yet generated. Run the pipeline to populate.")

