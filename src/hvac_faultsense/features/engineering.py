"""Feature Engineering for HVAC Thermodynamic Telemetry.

Derives physically meaningful thermodynamic indicators (temperature differences,
compressor pressure ratios, coil splits) based on HVAC engineering principles.
"""

from typing import List
import numpy as np
import pandas as pd


def compute_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical HVAC diagnostic indicators from raw sensor readings.

    Thermodynamic relationships:
    1. delta_T_air = RTU_RA_TEMP - RTU_SA_TEMP (Air temperature drop across evaporator cooling coil)
    2. delta_T_cond = RTU_REFG_COND_TEMP - RTU_OA_TEMP (Condenser split temperature)
    3. delta_T_disc = RTU_REFG_DISC_TEMP - RTU_REFG_COND_TEMP (Discharge gas de-superheating split)
    4. compression_ratio = RTU_REFG_DISC_PRES / max(RTU_REFG_SUCT_PRES, 1e-3) (Compressor compression ratio)
    5. compressor_power_ratio = RTU_COMP_WATT / max(RTU_TOT_WATT, 1.0) (Compressor electrical fraction)
    6. fan_power_ratio = RTU_SA_FAN_WATT / max(RTU_TOT_WATT, 1.0) (Fan electrical fraction)
    """
    out = df.copy()

    # Temperature differences
    if "RTU_RA_TEMP" in out.columns and "RTU_SA_TEMP" in out.columns:
        out["feat_delta_T_air"] = out["RTU_RA_TEMP"] - out["RTU_SA_TEMP"]

    if "RTU_REFG_COND_TEMP" in out.columns and "RTU_OA_TEMP" in out.columns:
        out["feat_delta_T_cond"] = out["RTU_REFG_COND_TEMP"] - out["RTU_OA_TEMP"]

    if "RTU_REFG_DISC_TEMP" in out.columns and "RTU_REFG_COND_TEMP" in out.columns:
        out["feat_delta_T_disc"] = out["RTU_REFG_DISC_TEMP"] - out["RTU_REFG_COND_TEMP"]

    # Pressure ratios
    if "RTU_REFG_DISC_PRES" in out.columns and "RTU_REFG_SUCT_PRES" in out.columns:
        denom = out["RTU_REFG_SUCT_PRES"].replace(0, np.nan)
        out["feat_compression_ratio"] = out["RTU_REFG_DISC_PRES"] / denom
        out["feat_compression_ratio"] = out["feat_compression_ratio"].fillna(1.0)

    # Electrical ratios
    if "RTU_COMP_WATT" in out.columns and "RTU_TOT_WATT" in out.columns:
        denom = out["RTU_TOT_WATT"].replace(0, np.nan)
        out["feat_comp_power_fraction"] = (out["RTU_COMP_WATT"] / denom).fillna(0.0)

    if "RTU_SA_FAN_WATT" in out.columns and "RTU_TOT_WATT" in out.columns:
        denom = out["RTU_TOT_WATT"].replace(0, np.nan)
        out["feat_fan_power_fraction"] = (out["RTU_SA_FAN_WATT"] / denom).fillna(0.0)

    return out


def get_feature_column_names(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """Retrieve list of numeric feature columns excluding labels, identifiers, and targets."""
    if exclude_cols is None:
        exclude_cols = [
            "Datetime",
            "scenario_name",
            "fault_type",
            "fault_severity",
            "is_fault",
            "RTU_TOT_WATT", # Continuous target
        ]
    cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    return cols

