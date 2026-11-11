"""Regression Evaluation Metrics and Baseline Estimator.

Calculates MAE, MSE, RMSE, and R2 from scratch, plus provides a naive Mean Baseline.
"""

from typing import Dict, Union
import numpy as np


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: (1/n) * sum(|y - y_hat|)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error: (1/n) * sum((y - y_hat)^2)."""
    return float(np.mean((y_true - y_pred) ** 2))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: sqrt(MSE)."""
    return float(np.sqrt(compute_mse(y_true, y_pred)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination R^2: 1 - (SS_res / SS_tot)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all standard regression metrics."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return {
        "MAE": round(compute_mae(y_true, y_pred), 4),
        "MSE": round(compute_mse(y_true, y_pred), 4),
        "RMSE": round(compute_rmse(y_true, y_pred), 4),
        "R2": round(compute_r2(y_true, y_pred), 4),
    }


class MeanBaselineRegressor:
    """Naive baseline predicting the mean of training target values."""

    def __init__(self):
        self.mean_value_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanBaselineRegressor":
        self.mean_value_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_value_)

