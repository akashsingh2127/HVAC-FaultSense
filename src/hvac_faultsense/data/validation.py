"""Data Validation and Statistical Profiling Module.

Performs rigorous statistical profiling, data quality auditing,
correlation analysis, and outlier inspection on HVAC telemetry.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def profile_dataset(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    variance_threshold: float = 1e-5,
) -> Dict[str, Any]:
    """Generate a comprehensive statistical and data quality profile.

    Args:
        df: Input DataFrame containing HVAC telemetry.
        sensor_cols: Specific columns to treat as sensor features (defaults to all numeric).
        variance_threshold: Threshold below which a feature is flagged as near-constant.

    Returns:
        Structured dictionary containing full profile metrics.
    """
    total_records = len(df)
    total_columns = len(df.columns)

    if sensor_cols is None:
        sensor_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["is_fault", "fault_severity"]]

    # 1. Missing values and data types
    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = ((df.isnull().sum() / max(total_records, 1)) * 100).round(3).to_dict()
    dtypes = {c: str(df[c].dtype) for c in df.columns}

    # 2. Duplicate analysis
    duplicate_rows = int(df.duplicated().sum())
    duplicate_timestamps = int(df["Datetime"].duplicated().sum()) if "Datetime" in df.columns else 0

    # 3. Statistical summary for sensor columns
    stat_summary = {}
    constant_columns = []
    near_constant_columns = []
    outlier_counts = {}

    for col in sensor_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        c_mean = float(series.mean())
        c_median = float(series.median())
        c_mode = float(series.mode().iloc[0]) if not series.mode().empty else c_mean
        c_var = float(series.var()) if len(series) > 1 else 0.0
        c_std = float(series.std()) if len(series) > 1 else 0.0
        c_min = float(series.min())
        c_max = float(series.max())
        q25 = float(series.quantile(0.25))
        q50 = float(series.quantile(0.50))
        q75 = float(series.quantile(0.75))
        c_skew = float(series.skew()) if len(series) > 2 else 0.0
        c_unique = int(series.nunique())

        if c_unique <= 1 or c_var == 0.0:
            constant_columns.append(col)
        elif c_var < variance_threshold:
            near_constant_columns.append(col)

        # IQR-based outlier detection
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        num_outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
        outlier_counts[col] = num_outliers

        stat_summary[col] = {
            "mean": round(c_mean, 4),
            "median": round(c_median, 4),
            "mode": round(c_mode, 4),
            "variance": round(c_var, 4),
            "std": round(c_std, 4),
            "min": round(c_min, 4),
            "q25": round(q25, 4),
            "q50": round(q50, 4),
            "q75": round(q75, 4),
            "max": round(c_max, 4),
            "skewness": round(c_skew, 4),
            "unique_values": c_unique,
            "outlier_count": num_outliers,
            "outlier_pct": round((num_outliers / max(len(series), 1)) * 100, 2),
        }

    # 4. Correlation matrix among sensor features
    corr_df = df[sensor_cols].corr()
    corr_matrix = corr_df.round(4).to_dict()

    # 5. Label and scenario distribution if present
    class_distribution = {}
    severity_distribution = {}
    if "fault_type" in df.columns:
        class_distribution = df["fault_type"].value_counts().to_dict()
    if "fault_severity" in df.columns:
        severity_distribution = df["fault_severity"].value_counts().to_dict()

    profile = {
        "dataset_dimensions": {"rows": total_records, "columns": total_columns},
        "sensor_columns": sensor_cols,
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "missing_percentages": missing_pcts,
        "duplicate_rows": duplicate_rows,
        "duplicate_timestamps": duplicate_timestamps,
        "constant_columns": constant_columns,
        "near_constant_columns": near_constant_columns,
        "statistics": stat_summary,
        "correlations": corr_matrix,
        "class_distribution": class_distribution,
        "severity_distribution": severity_distribution,
    }
    return profile


def generate_profile_markdown(profile: Dict[str, Any]) -> str:
    """Format the profiling results into a readable Markdown report."""
    dim = profile["dataset_dimensions"]
    md = [
        "# HVAC FaultSense - Dataset Quality and Statistical Profile",
        "",
        "## 1. Dimensions & Structure",
        f"- **Total Rows:** {dim['rows']:,}",
        f"- **Total Columns:** {dim['columns']}",
        f"- **Duplicate Rows:** {profile['duplicate_rows']}",
        f"- **Duplicate Timestamps:** {profile['duplicate_timestamps']}",
        f"- **Constant Columns:** {profile['constant_columns'] if profile['constant_columns'] else 'None'}",
        f"- **Near-Constant Columns:** {profile['near_constant_columns'] if profile['near_constant_columns'] else 'None'}",
        "",
        "## 2. Statistical Summary",
        "| Sensor Column | Mean | Std | Min | 25% | Median | 75% | Max | Skewness | Outliers (%) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for col, s in profile["statistics"].items():
        md.append(
            f"| `{col}` | {s['mean']} | {s['std']} | {s['min']} | {s['q25']} | {s['median']} | {s['q75']} | {s['max']} | {s['skewness']} | {s['outlier_count']} ({s['outlier_pct']}%) |"
        )

    if profile["class_distribution"]:
        md.extend([
            "",
            "## 3. Fault Class Distribution",
            "| Fault Class | Record Count | Percentage |",
            "|---|---|---|",
        ])
        total = sum(profile["class_distribution"].values())
        for cls_name, count in profile["class_distribution"].items():
            pct = round((count / total) * 100, 2)
            md.append(f"| **{cls_name}** | {count:,} | {pct}% |")

    return "\n".join(md)

