"""Plotly Visualizations for HVAC FaultSense.

Generates interactive, publication-quality Plotly figures for data profiling,
regression residuals, PCA projections, cluster regimes, HMM sequences, and classifier benchmarks.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_missing_values(missing_dict: Dict[str, float]) -> go.Figure:
    """Bar chart displaying missing value percentages per column."""
    df_missing = pd.DataFrame(list(missing_dict.items()), columns=["Column", "Missing_Pct"])
    fig = px.bar(
        df_missing,
        x="Column",
        y="Missing_Pct",
        title="Missing Value Percentage per Telemetry Point",
        labels={"Missing_Pct": "Missing (%)", "Column": "Sensor / Point"},
        text_auto=".2f",
    )
    fig.update_layout(xaxis_tickangle=-45, template="plotly_white")
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Interactive correlation heatmap for physical sensors."""
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="HVAC Sensor Correlation Matrix",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_sensor_timeseries(
    df: pd.DataFrame,
    sensor_col: str,
    time_col: str = "Datetime",
    color_col: Optional[str] = "fault_type",
) -> go.Figure:
    """Time-series line chart for a specific sensor, optionally grouped by fault type."""
    fig = px.line(
        df,
        x=time_col,
        y=sensor_col,
        color=color_col if color_col in df.columns else None,
        title=f"Time-Series Telemetry: {sensor_col}",
        labels={sensor_col: f"{sensor_col}", time_col: "Time (1-min Interval)"},
    )
    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


def plot_sensor_distribution(
    df: pd.DataFrame,
    sensor_col: str,
    color_col: Optional[str] = "fault_type",
) -> go.Figure:
    """Histogram and marginal box plot showing distribution and outliers."""
    fig = px.histogram(
        df,
        x=sensor_col,
        color=color_col if color_col in df.columns else None,
        marginal="box",
        nbins=50,
        title=f"Distribution and Outliers: {sensor_col}",
        template="plotly_white",
    )
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    target_unit: str = "Watts",
) -> go.Figure:
    """Scatter plot of actual versus predicted values with ideal y=x reference line."""
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="Predictions",
            marker=dict(color="#1f77b4", opacity=0.5, size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal Line (y = x)",
            line=dict(color="red", dash="dash", width=2),
        )
    )
    fig.update_layout(
        title=f"Actual vs Predicted Power Demand ({model_name})",
        xaxis_title=f"Actual Measured Target ({target_unit})",
        yaxis_title=f"Predicted Target ({target_unit})",
        template="plotly_white",
    )
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Residual distribution histogram and normal curve reference."""
    residuals = y_true - y_pred
    fig = px.histogram(
        x=residuals,
        nbins=60,
        marginal="box",
        title=f"Prediction Residual Distribution ({model_name})",
        labels={"x": "Residual (e = y - y_hat) [Watts]"},
        template="plotly_white",
    )
    return fig


def plot_residuals_over_time(
    timestamps: pd.Series,
    residuals: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Residuals plotted across time with zero-error baseline."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=residuals,
            mode="lines",
            name="Residual (e_t)",
            line=dict(color="#2ca02c", width=1.5),
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="black")
    fig.update_layout(
        title=f"Residual Trajectory Over Time ({model_name})",
        xaxis_title="Timestamp",
        yaxis_title="Residual e_t (Watts)",
        template="plotly_white",
    )
    return fig


def plot_pca_variance(explained_variance_ratio: np.ndarray) -> go.Figure:
    """Plot individual and cumulative explained variance ratios."""
    k = len(explained_variance_ratio)
    comp_labels = [f"PC{i+1}" for i in range(k)]
    cum_var = np.cumsum(explained_variance_ratio)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=comp_labels,
            y=explained_variance_ratio,
            name="Individual Variance",
            text=[f"{v:.1%}" for v in explained_variance_ratio],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comp_labels,
            y=cum_var,
            name="Cumulative Variance",
            mode="lines+markers",
            line=dict(color="orange", width=2),
        )
    )
    fig.update_layout(
        title="PCA Explained Variance Ratio",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance Ratio",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
    )
    return fig


def plot_pca_2d_projection(
    X_pca: np.ndarray,
    color_labels: pd.Series,
    color_title: str = "Operating Regime",
) -> go.Figure:
    """2D scatter plot of principal component projection."""
    df_pca = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        color_title: color_labels,
    })
    fig = px.scatter(
        df_pca,
        x="PC1",
        y="PC2",
        color=color_title,
        title=f"2D PCA Projection (Colored by {color_title})",
        opacity=0.7,
        template="plotly_white",
    )
    return fig


def plot_hmm_sequence(
    timestamps: pd.Series,
    obs_sequence: np.ndarray,
    decoded_states: np.ndarray,
    max_points: int = 1000,
) -> go.Figure:
    """Compare raw discrete observation stream with temporally smoothed HMM state sequence."""
    ts = timestamps.iloc[:max_points]
    obs = obs_sequence[:max_points]
    states = decoded_states[:max_points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=obs,
            mode="markers",
            name="Raw Discretized Cluster / Regime",
            marker=dict(color="lightgray", size=5, symbol="circle-open"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=states,
            mode="lines+markers",
            name="Viterbi Smoothed Latent State",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        title="Temporal Operating Regime Tracking: Raw Observations vs HMM Viterbi Path",
        xaxis_title="Timestamp",
        yaxis_title="State / Cluster Index",
        template="plotly_white",
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], model_name: str = "Model") -> go.Figure:
    """Interactive normalized confusion matrix heatmap."""
    row_sums = np.sum(cm, axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig = px.imshow(
        cm_norm,
        x=classes,
        y=classes,
        color_continuous_scale="Blues",
        text_auto=".2f",
        labels=dict(x="Predicted Class", y="True Ground-Truth Class"),
        title=f"Confusion Matrix (Normalized) - {model_name}",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_feature_importances(feature_names: List[str], importances: np.ndarray, top_k: int = 15) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values("Importance", ascending=True).tail(top_k)

    fig = px.bar(
        df_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_k} Feature Importances",
        text_auto=".3f",
        template="plotly_white",
    )
    return fig


def plot_benchmark_bar(
    df_bench: pd.DataFrame,
    metric_col: str = "Macro-F1",
    task: str = "Classification",
) -> go.Figure:
    """Bar chart comparing models on a specific metric."""
    df_sub = df_bench[df_bench["Task"] == task].copy()
    if metric_col not in df_sub.columns or df_sub.empty:
        return go.Figure()

    # Convert metric to float
    df_sub[metric_col] = pd.to_numeric(df_sub[metric_col], errors="coerce")
    fig = px.bar(
        df_sub,
        x="Model",
        y=metric_col,
        color="Model",
        title=f"{task} Model Comparison: {metric_col}",
        text_auto=".4f",
        template="plotly_white",
    )
    return fig

