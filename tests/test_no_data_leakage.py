"""Strict verification of time-series isolation and data leakage safeguards."""

import numpy as np
import pandas as pd
import pytest

from hvac_faultsense.data.preprocessing import (
    StandardScalerScratch,
    prepare_feature_matrices,
    split_scenario_holdout,
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
    expected_scaled = (1000.0 - 20.0) / scaler.state.stds["f_0"]
    assert np.isclose(X_test_scaled[0, 0], expected_scaled)


def test_no_target_or_component_power_leakage_in_regression():
    """Ensure target variable AND sub-component powers are strictly excluded from regression features."""
    df = pd.DataFrame({
        "Datetime": pd.date_range("2021-01-01", periods=10, freq="1min"),
        "scenario_name": ["S1"] * 10,
        "fault_type": ["FaultFree"] * 10,
        "fault_severity": [0.0] * 10,
        "is_fault": [0] * 10,
        "RTU_TOT_WATT": range(10), # Target
        "RTU_COMP_WATT": range(10), # Sub-component power (leakage)
        "RTU_SA_FAN_WATT": range(10), # Sub-component power (leakage)
        "RTU_OA_TEMP": range(10), # Valid boundary feature
        "RTU_OA_HUM": range(10), # Valid boundary feature
    })
    reg_feats = get_feature_column_names(df, task="regression")

    # Anti-leakage assertions
    assert "RTU_TOT_WATT" not in reg_feats
    assert "RTU_COMP_WATT" not in reg_feats
    assert "RTU_SA_FAN_WATT" not in reg_feats
    assert "fault_type" not in reg_feats
    assert "scenario_name" not in reg_feats

    # Valid predictors must remain
    assert "RTU_OA_TEMP" in reg_feats
    assert "RTU_OA_HUM" in reg_feats


def test_scenario_holdout_split():
    """Verify scenario-aware holdout splits partition full scenarios without overlap."""
    df = pd.DataFrame({
        "scenario_name": ["ScenA"] * 20 + ["ScenB"] * 20 + ["ScenC"] * 20,
        "val": range(60),
    })
    df_train, df_test = split_scenario_holdout(df, holdout_scenarios=["ScenB"])
    assert "ScenB" not in df_train["scenario_name"].values
    assert set(df_test["scenario_name"].unique()) == {"ScenB"}
    assert len(df_train) == 40
    assert len(df_test) == 20
