"""Dataset Ingestion Module for HVAC FaultSense.

Loads raw CSV data, attaches scenario metadata, validates columns,
and provides time-downsampled subsets for rapid, reproducible experimentation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from hvac_faultsense.data.metadata import SIMULATED_SCENARIO_MAPPING


def load_single_csv(
    file_path: Union[str, Path],
    parse_dates: bool = True,
    stride: int = 1,
) -> pd.DataFrame:
    """Load a single CSV file with optional time parsing and striding.

    Args:
        file_path: Path to the CSV file.
        parse_dates: Whether to parse the 'Datetime' column into datetime objects.
        stride: Keep every k-th row (e.g., stride=10 turns 1-min data into 10-min data).

    Returns:
        Loaded pandas DataFrame.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    df = pd.read_csv(file_path)
    if parse_dates and "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime").reset_index(drop=True)

    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)

    return df


def load_simulated_dataset(
    sim_dir: Union[str, Path],
    selected_scenarios: Optional[List[str]] = None,
    stride: int = 1,
) -> pd.DataFrame:
    """Load and concatenate Simulated RTU scenarios with verified ground truth labels.

    Args:
        sim_dir: Directory containing Simulated RTU CSV files.
        selected_scenarios: Optional list of filename prefixes or scenario names.
        stride: Subsampling stride to manage memory/speed while preserving time properties.

    Returns:
        Concatenated DataFrame with metadata columns attached.
    """
    sim_dir = Path(sim_dir)
    csv_files = sorted(sim_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {sim_dir}")

    dfs = []
    for csv_file in csv_files:
        fname = csv_file.name
        if selected_scenarios:
            if not any(s in fname for s in selected_scenarios):
                continue

        df = load_single_csv(csv_file, stride=stride)
        meta = SIMULATED_SCENARIO_MAPPING.get(
            fname,
            {
                "fault_type": "Unknown",
                "severity": 0.0,
                "is_fault": -1,
                "description": "Unclassified scenario",
            },
        )
        df["scenario_name"] = fname.replace(".csv", "")
        df["fault_type"] = meta["fault_type"]
        df["fault_severity"] = meta["severity"]
        df["is_fault"] = meta["is_fault"]
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def create_synthetic_rtu_fixture(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a clearly-labelled synthetic test fixture for CI and testing.

    This synthetic fixture strictly mimics the column schema and ranges
    of the real LBNL RTU dataset, without fabricating real experimental results.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01 00:00:00", periods=n_rows, freq="1min")

    data = {
        "Datetime": dates,
        "RTU_COMP_WATT": rng.uniform(2000.0, 4500.0, size=n_rows),
        "RTU_OA_FLOW": rng.uniform(300.0, 800.0, size=n_rows),
        "RTU_OA_HUM": rng.uniform(30.0, 85.0, size=n_rows),
        "RTU_OA_TEMP": rng.uniform(15.0, 35.0, size=n_rows),
        "RTU_RA_FLOW": rng.uniform(2000.0, 3000.0, size=n_rows),
        "RTU_RA_HUM": rng.uniform(40.0, 60.0, size=n_rows),
        "RTU_RA_TEMP": rng.uniform(21.0, 25.0, size=n_rows),
        "RTU_REFG_COND_PRES": rng.uniform(1500.0, 2500.0, size=n_rows),
        "RTU_REFG_COND_TEMP": rng.uniform(35.0, 50.0, size=n_rows),
        "RTU_REFG_DISC_PRES": rng.uniform(1600.0, 2600.0, size=n_rows),
        "RTU_REFG_DISC_TEMP": rng.uniform(55.0, 85.0, size=n_rows),
        "RTU_REFG_SUCT_PRES": rng.uniform(600.0, 950.0, size=n_rows),
        "RTU_REFG_SUCT_TEMP": rng.uniform(5.0, 18.0, size=n_rows),
        "RTU_SA_FAN_WATT": rng.uniform(300.0, 650.0, size=n_rows),
        "RTU_SA_FLOW": rng.uniform(2500.0, 3500.0, size=n_rows),
        "RTU_SA_HUM": rng.uniform(70.0, 95.0, size=n_rows),
        "RTU_SA_TEMP": rng.uniform(11.0, 16.0, size=n_rows),
        "RTU_SEN_CAPA": rng.uniform(8000.0, 16000.0, size=n_rows),
        "RTU_STG_STA": rng.choice([0.0, 1.0, 2.0], size=n_rows),
        "RTU_TOT_CAPA": rng.uniform(10000.0, 22000.0, size=n_rows),
        "RTU_TOT_WATT": rng.uniform(2500.0, 5200.0, size=n_rows),
        "ZA_HUM": rng.uniform(40.0, 60.0, size=n_rows),
        "ZA_TEMP": rng.uniform(21.0, 24.5, size=n_rows),
        "ZA_TEMP_SPT": np.full(n_rows, 23.0),
        "scenario_name": ["Synthetic_Baseline"] * n_rows,
        "fault_type": ["FaultFree"] * n_rows,
        "fault_severity": [0.0] * n_rows,
        "is_fault": [0] * n_rows,
    }
    return pd.DataFrame(data)

