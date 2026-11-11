"""Unit tests for preprocessing, StandardScaler from scratch, and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from hvac_faultsense.data.preprocessing import (
    StandardScalerScratch,
    split_scenario_holdout,
    split_time_and_scenario_aware,
)
from hvac_faultsense.features.engineering import compute_domain_features, get_feature_column_names


def test_standard_scaler_scratch():
    X = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    scaler = StandardScalerScratch()
    X_scaled = scaler.fit_transform(X)

    # Check scaled mean ~ 0 and std ~ 1
    assert np.allclose(np.mean(X_scaled, axis=0), [0.0, 0.0], atol=1e-7)
    assert np.allclose(np.std(X_scaled, axis=0), [1.0, 1.0], atol=1e-7)

    # Check inverse transform
    X_inv = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_inv, atol=1e-7)


def test_standard_scaler_constant_column():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    scaler = StandardScalerScratch()
    X_scaled = scaler.fit_transform(X)
    assert not np.isnan(X_scaled).any()
    assert np.allclose(X_scaled[:, 0], 0.0)


def test_compute_domain_features():
    df = pd.DataFrame({
        "RTU_RA_TEMP": [24.0, 25.0],
        "RTU_SA_TEMP": [14.0, 15.0],
        "RTU_REFG_COND_TEMP": [45.0, 50.0],
        "RTU_OA_TEMP": [30.0, 35.0],
        "RTU_REFG_DISC_PRES": [2000.0, 2200.0],
        "RTU_REFG_SUCT_PRES": [800.0, 800.0],
        "RTU_SA_FLOW": [3000.0, 3200.0],
    })
    df_feat = compute_domain_features(df)
    assert "feat_delta_T_air" in df_feat.columns
    assert np.allclose(df_feat["feat_delta_T_air"], [10.0, 10.0])
    assert "feat_compression_ratio" in df_feat.columns
    assert np.allclose(df_feat["feat_compression_ratio"], [2.5, 2.75])
    assert "feat_air_cooling_load_proxy" in df_feat.columns
    assert np.allclose(df_feat["feat_air_cooling_load_proxy"], [30000.0, 32000.0])


def test_split_time_and_scenario_aware():
    df = pd.DataFrame({
        "Datetime": pd.date_range("2021-01-01", periods=100, freq="1min"),
        "scenario_name": ["ScenA"] * 50 + ["ScenB"] * 50,
        "val": range(100),
    })
    df_train, df_test = split_time_and_scenario_aware(df, test_ratio=0.20)
    assert len(df_train) == 80
    assert len(df_test) == 20
    assert len(df_train[df_train["scenario_name"] == "ScenA"]) == 40
    assert len(df_test[df_test["scenario_name"] == "ScenA"]) == 10
