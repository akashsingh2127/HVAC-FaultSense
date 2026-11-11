"""Unit tests for dataset ingestion, manifest building, and synthetic fixture generation."""

import pytest
import pandas as pd
from pathlib import Path

from hvac_faultsense.data.ingestion import load_single_csv, create_synthetic_rtu_fixture
from hvac_faultsense.data.metadata import build_dataset_manifest, get_data_dictionary


def test_create_synthetic_fixture():
    df = create_synthetic_rtu_fixture(n_rows=50, seed=42)
    assert len(df) == 50
    assert "Datetime" in df.columns
    assert "RTU_TOT_WATT" in df.columns
    assert "fault_type" in df.columns
    assert df["fault_type"].iloc[0] == "FaultFree"
    assert df["is_fault"].iloc[0] == 0


def test_data_dictionary_schema():
    df_dict = get_data_dictionary()
    assert len(df_dict) == 25
    expected_cols = {"column_name", "physical_interpretation", "measurement_unit", "data_type", "brick_class"}
    assert expected_cols.issubset(set(df_dict.columns))


def test_load_nonexistent_file_raises_error(tmp_path):
    missing_file = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        load_single_csv(missing_file)


def test_load_single_csv_with_stride(tmp_path):
    df_sample = pd.DataFrame({
        "Datetime": pd.date_range("2021-01-01", periods=20, freq="1min"),
        "RTU_TOT_WATT": range(20),
    })
    csv_file = tmp_path / "sample.csv"
    df_sample.to_csv(csv_file, index=False)

    df_loaded = load_single_csv(csv_file, stride=5)
    assert len(df_loaded) == 4
    assert pd.api.types.is_datetime64_any_dtype(df_loaded["Datetime"])

