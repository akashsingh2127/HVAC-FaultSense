"""Feature Engineering for HVAC Thermodynamic Telemetry.

Derives physically meaningful thermodynamic indicators (temperature differences,
compressor pressure ratios, coil splits) based on HVAC engineering principles,
with strict guarantees against target and component power leakage.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def compute_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical HVAC diagnostic indicators from raw sensor readings.

    Thermodynamic relationships:
    1. delta_T_air = RTU_RA_TEMP - RTU_SA_TEMP (Air temperature drop across cooling coil)
    2. delta_T_cond = RTU_REFG_COND_TEMP - RTU_OA_TEMP (Condenser split temperature)
    3. delta_T_disc = RTU_REFG_DISC_TEMP - RTU_REFG_COND_TEMP (Discharge gas de-superheating split)
    4. compression_ratio = RTU_REFG_DISC_PRES / max(RTU_REFG_SUCT_PRES, 1e-3) (Compressor compression ratio)
    5. air_cooling_load_proxy = (RTU_RA_TEMP - RTU_SA_TEMP) * RTU_SA_FLOW (Sensible thermal heat removal proxy)

    NOTE: No target variables (RTU_TOT_WATT) or sub-component power ratios are computed here
    to prevent target leakage.
    """
    out = df.copy()

    # 1. Temperature differences across heat exchangers
    if "RTU_RA_TEMP" in out.columns and "RTU_SA_TEMP" in out.columns:
        out["feat_delta_T_air"] = out["RTU_RA_TEMP"] - out["RTU_SA_TEMP"]

    if "RTU_REFG_COND_TEMP" in out.columns and "RTU_OA_TEMP" in out.columns:
        out["feat_delta_T_cond"] = out["RTU_REFG_COND_TEMP"] - out["RTU_OA_TEMP"]

    if "RTU_REFG_DISC_TEMP" in out.columns and "RTU_REFG_COND_TEMP" in out.columns:
        out["feat_delta_T_disc"] = out["RTU_REFG_DISC_TEMP"] - out["RTU_REFG_COND_TEMP"]

    # 2. Refrigeration cycle pressure ratios
    if "RTU_REFG_DISC_PRES" in out.columns and "RTU_REFG_SUCT_PRES" in out.columns:
        denom = out["RTU_REFG_SUCT_PRES"].replace(0, np.nan)
        out["feat_compression_ratio"] = (out["RTU_REFG_DISC_PRES"] / denom).fillna(1.0)

    # 3. Sensible air heat exchange proxy: Q_sensible ~ m_dot * c_p * delta_T
    if "feat_delta_T_air" in out.columns and "RTU_SA_FLOW" in out.columns:
        out["feat_air_cooling_load_proxy"] = out["feat_delta_T_air"] * out["RTU_SA_FLOW"]

    return out


def get_feature_column_names(
    df: pd.DataFrame,
    task: str = "classification",
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """Retrieve numeric feature columns with task-specific leakage protection.

    Args:
        df: Input DataFrame.
        task: 'regression' or 'classification'.
            - In 'regression' (predicting RTU_TOT_WATT): strictly excludes RTU_TOT_WATT,
              RTU_COMP_WATT, RTU_SA_FAN_WATT, RTU_SEN_CAPA, RTU_TOT_CAPA to prevent
              tautological power identity leakage.
            - In 'classification': uses full sensor suite while strictly excluding
              labels and metadata.
        exclude_cols: Optional explicit list of columns to exclude.
    """
    base_exclusions = [
        "Datetime",
        "scenario_name",
        "fault_type",
        "fault_severity",
        "is_fault",
    ]

    if task.lower() == "regression":
        # Target leakage & component power identity exclusions:
        # We must predict total electrical power strictly from boundary and operating conditions!
        regression_exclusions = base_exclusions + [
            "RTU_TOT_WATT",  # Target
            "RTU_COMP_WATT", # Subsystem electrical power
            "RTU_SA_FAN_WATT", # Subsystem electrical power
            "RTU_SEN_CAPA",  # Direct thermal capacity measurement
            "RTU_TOT_CAPA",  # Direct thermal capacity measurement
        ]
        exclusions = set(regression_exclusions if exclude_cols is None else exclude_cols + regression_exclusions)
    else:
        exclusions = set(base_exclusions if exclude_cols is None else exclude_cols + base_exclusions)

    cols = [c for c in df.columns if c not in exclusions and pd.api.types.is_numeric_dtype(df[c])]
    return cols
