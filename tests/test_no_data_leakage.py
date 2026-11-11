"""Strict verification of time-series isolation and data leakage safeguards."""

import numpy as np
import pandas as pd
import pytest

from hvac_faultsense.data.preprocessing import (
    StandardScalerScratch,
    prepare_feature_matrices,
    split_time_and_scenario_aware,
)
from hvac_faultsense.features.engineering import compute_domain_features, get_feature_column_names


def test_chronological_ordering_guarantee():
    """Ensure training data strictly precedes held-out test data in every scenario."""
    dates = pd.date_range("2021-01-01 00:00:00", periods=100, freq="1min")
    df = pd.DataFrame({
        "Datetime": dates,
        "scenario_name": ["Scen1"] * 50 + ["Scen2"] * 50,
        "RTU_TOT_WATT": np.random.uniform(2000, 4000, size=100),
        "RTU_OA_TEMP": np.random.uniform(15, 30, size=100),
        "fault_type": ["FaultFree"] * 100,
    })

    df_train, df_test = split_time_and_scenario_aware(df, test_ratio=0.30)

    for scen in ["Scen1", "Scen2"]:
        train_max_time = df_train[df_train["scenario_name"] == scen]["Datetime"].max()
        test_min_time = df_test[df_test["scenario_name"] == scen]["Datetime"].min()
        # Train timestamp must be strictly less than test timestamp
        assert train_max_time < test_min_time, f"Leakage detected: {train_max_time} >= {test_min_time} in {scen}"


def test_scaler_zero_leakage():
    """Verify scaler state depends exclusively on training set mean/std."""
    X_train = np.array([[10.0], [20.0], [30.0]]) # Mean = 20.0
    X_test = np.array([[1000.0], [2000.0]]) # Very high test values

    scaler = StandardScalerScratch()
    scaler.fit(X_train)

    # Scaler mean must equal 20.0 and must not be influenced by test data
    assert np.isclose(scaler.state.means["f_0"], 20.0)

    X_test_scaled = scaler.transform(X_test)
    # 1000 - 20 / std
    expected_scaled = (1000.0 - 20.0) / scaler.state.stds["f_0"]
    assert np.isclose(X_test_scaled[0, 0], expected_scaled)


def test_no_target_or_label_in_feature_columns():
    """Ensure labels, scenario names, and target are stripped from feature matrices."""
    df = pd.DataFrame({
        "Datetime": pd.date_range("2021-01-01", periods=10, freq="1min"),
        "scenario_name": ["S1"] * 10,
        "fault_type": ["FaultFree"] * 10,
        "fault_severity": [0.0] * 10,
        "is_fault": [0] * 10,
        "RTU_TOT_WATT": range(10), # Target
        "RTU_OA_TEMP": range(10), # Valid feature
    })
    feat_cols = get_feature_column_names(df)
    assert "fault_type" not in feat_cols
    assert "scenario_name" not in feat_cols
    assert "is_fault" not in feat_cols
    assert "RTU_TOT_WATT" not in feat_cols
    assert "RTU_OA_TEMP" in feat_cols

