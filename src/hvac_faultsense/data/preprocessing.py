"""Preprocessing and Leakage-Safe Data Splitting for HVAC Telemetry.

Enforces strict separation of training and test statistics. Imputers,
scalers, and selectors are fitted exclusively on training splits.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class ScalerState:
    """Stores mean and standard deviation fitted strictly on training data."""
    means: Dict[str, float]
    stds: Dict[str, float]
    feature_names: List[str]


class StandardScalerScratch:
    """StandardScaler implemented with explicit NumPy operations.

    Transforms features via z = (x - mu) / sigma.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.state: Optional[ScalerState] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "StandardScalerScratch":
        """Fit mean and std on training data only."""
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            arr = X.to_numpy(dtype=np.float64)
        else:
            feature_names = [f"f_{i}" for i in range(X.shape[1])]
            arr = np.asarray(X, dtype=np.float64)

        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)
        # Avoid division by zero for constant columns
        stds = np.where(stds < self.eps, 1.0, stds)

        self.state = ScalerState(
            means={name: float(m) for name, m in zip(feature_names, means)},
            stds={name: float(s) for name, s in zip(feature_names, stds)},
            feature_names=feature_names,
        )
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply standardization z = (X - mean) / std."""
        if self.state is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            arr = X[self.state.feature_names].to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)

        means_vec = np.array([self.state.means[col] for col in self.state.feature_names])
        stds_vec = np.array([self.state.stds[col] for col in self.state.feature_names])
        return (arr - means_vec) / stds_vec

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit on X and transform X."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Invert scaling back to original physical engineering units."""
        if self.state is None:
            raise ValueError("Scaler has not been fitted.")
        means_vec = np.array([self.state.means[col] for col in self.state.feature_names])
        stds_vec = np.array([self.state.stds[col] for col in self.state.feature_names])
        return X_scaled * stds_vec + means_vec


def split_time_and_scenario_aware(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
    scenario_col: str = "scenario_name",
    time_col: str = "Datetime",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a leakage-safe chronological split within each operating scenario.

    For each distinct equipment scenario (e.g. baseline, fouling severities),
    the initial (1 - test_ratio) chronological fraction forms the training set,
    and the final test_ratio chronological fraction forms the held-out test set.

    This ensures:
    1. Past observations never bleed into the future.
    2. Every operational scenario is evaluated on strictly held-out temporal windows.
    3. No row-shuffling or data leakage across adjacent time-series steps.
    """
    train_dfs = []
    test_dfs = []

    if scenario_col not in df.columns:
        sorted_df = df.sort_values(time_col).reset_index(drop=True)
        split_idx = int(len(sorted_df) * (1 - test_ratio))
        return sorted_df.iloc[:split_idx].copy(), sorted_df.iloc[split_idx:].copy()

    for _, group in df.groupby(scenario_col):
        sorted_group = group.sort_values(time_col).reset_index(drop=True)
        n = len(sorted_group)
        split_idx = int(n * (1 - test_ratio))
        train_dfs.append(sorted_group.iloc[:split_idx])
        test_dfs.append(sorted_group.iloc[split_idx:])

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test = pd.concat(test_dfs, ignore_index=True)

    return df_train, df_test


def split_scenario_holdout(
    df: pd.DataFrame,
    holdout_scenarios: List[str],
    scenario_col: str = "scenario_name",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform scenario-aware holdout splitting to test generalisation to unseen fault severities."""
    train_mask = ~df[scenario_col].isin(holdout_scenarios)
    test_mask = df[scenario_col].isin(holdout_scenarios)
    return df[train_mask].copy().reset_index(drop=True), df[test_mask].copy().reset_index(drop=True)


def prepare_feature_matrices(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "RTU_TOT_WATT",
    label_col: str = "fault_type",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScalerScratch]:
    """Extract and standardize feature matrices without data leakage.

    Returns:
        X_train_scaled, X_test_scaled, y_train_reg, y_test_reg, y_train_cls, y_test_cls, scaler
    """
    available_features = [f for f in feature_cols if f in df_train.columns and f != target_col]

    # Handle missing values: fit medians on train only
    train_medians = df_train[available_features].median()
    X_train_clean = df_train[available_features].fillna(train_medians)
    X_test_clean = df_test[available_features].fillna(train_medians)

    # Fit scaler strictly on training set
    scaler = StandardScalerScratch()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    # Continuous regression target
    y_train_reg = df_train[target_col].to_numpy(dtype=np.float64) if target_col in df_train.columns else np.array([])
    y_test_reg = df_test[target_col].to_numpy(dtype=np.float64) if target_col in df_test.columns else np.array([])

    # Supervised classification labels
    y_train_cls = df_train[label_col].to_numpy() if label_col in df_train.columns else np.array([])
    y_test_cls = df_test[label_col].to_numpy() if label_col in df_test.columns else np.array([])

    return X_train_scaled, X_test_scaled, y_train_reg, y_test_reg, y_train_cls, y_test_cls, scaler
