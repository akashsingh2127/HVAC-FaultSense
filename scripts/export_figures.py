"""Export standalone interactive Plotly figures to reports/figures/."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from hvac_faultsense.visualization.plots import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_residuals_over_time,
    plot_pca_variance,
    plot_pca_2d_projection,
    plot_hmm_sequence,
    plot_confusion_matrix,
    plot_benchmark_bar,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load test evaluation sample
sample_path = DATA_DIR / "processed" / "test_evaluation_sample.csv"
if not sample_path.exists():
    sample_path = DATA_DIR / "processed" / "test_evaluation_sample.parquet"

if sample_path.suffix == ".csv":
    df_sample = pd.read_csv(sample_path)
    df_sample["Datetime"] = pd.to_datetime(df_sample["Datetime"])
else:
    df_sample = pd.read_parquet(sample_path)

# Load benchmark results
bench_path = REPORTS_DIR / "results" / "benchmark_summary.csv"
df_bench = pd.read_csv(bench_path)

results_path = REPORTS_DIR / "results" / "experiment_results.json"
with open(results_path, "r") as f:
    results = json.load(f)

print("Exporting Plotly figures to reports/figures/...")

# 1. Regression Plots
y_true = df_sample["RTU_TOT_WATT"].to_numpy()
y_pred = df_sample["pred_power_ols"].to_numpy()
fig1 = plot_actual_vs_predicted(y_true, y_pred, model_name="OLS Regression (Scratch)")
fig1.write_html(str(FIGURES_DIR / "actual_vs_predicted_ols.html"))

fig2 = plot_residuals(y_true, y_pred, model_name="OLS Regression (Scratch)")
fig2.write_html(str(FIGURES_DIR / "residual_distribution_ols.html"))

residuals = df_sample["residual_ols"].to_numpy()
fig3 = plot_residuals_over_time(df_sample["Datetime"], residuals, model_name="OLS Regression (Scratch)")
fig3.write_html(str(FIGURES_DIR / "residuals_over_time.html"))

# 2. PCA & Clustering Plots
pca_var = np.array(results.get("pca_explained_variance_ratio", [0.5425, 0.1666]))
fig4 = plot_pca_variance(pca_var)
fig4.write_html(str(FIGURES_DIR / "pca_explained_variance.html"))

X_pca = df_sample[["pc1", "pc2"]].to_numpy()
fig5 = plot_pca_2d_projection(X_pca, df_sample["kmeans_cluster"].astype(str), color_title="K-Means Regime")
fig5.write_html(str(FIGURES_DIR / "pca_2d_projection_clusters.html"))

fig6 = plot_pca_2d_projection(X_pca, df_sample["fault_type"], color_title="True Fault Scenario")
fig6.write_html(str(FIGURES_DIR / "pca_2d_projection_faults.html"))

# 3. HMM Sequence Plot
fig7 = plot_hmm_sequence(df_sample["Datetime"], df_sample["kmeans_cluster"].to_numpy(), df_sample["hmm_state"].to_numpy(), max_points=800)
fig7.write_html(str(FIGURES_DIR / "hmm_viterbi_sequence.html"))

# 4. Classification Plots
cart_m = results["classification_metrics"]["cart"]
cm = np.array(cart_m["confusion_matrix"])
classes = cart_m["classes"]
fig8 = plot_confusion_matrix(cm, classes, model_name="CART Decision Tree (Scratch)")
fig8.write_html(str(FIGURES_DIR / "confusion_matrix_cart.html"))

# 5. Benchmark Comparison Plots
fig9 = plot_benchmark_bar(df_bench, metric_col="RMSE", task="Regression")
fig9.write_html(str(FIGURES_DIR / "benchmark_rmse_comparison.html"))

fig10 = plot_benchmark_bar(df_bench, metric_col="Macro-F1", task="Classification")
fig10.write_html(str(FIGURES_DIR / "benchmark_macro_f1_comparison.html"))

print(f"Exported 10 standalone figures to {FIGURES_DIR}.")

