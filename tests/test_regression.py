"""Unit tests for Linear Regression and Ridge Regression from Scratch."""

import numpy as np
import pytest

from hvac_faultsense.models.regression_scratch import LinearRegressionScratch
from hvac_faultsense.models.ridge_scratch import RidgeRegressionScratch
from hvac_faultsense.evaluation.regression_metrics import (
    MeanBaselineRegressor,
    compute_mae,
    compute_mse,
    compute_r2,
    compute_rmse,
    evaluate_regression,
)


def test_linear_regression_scratch_synthetic():
    rng = np.random.default_rng(42)
    # y = 3*x1 - 2*x2 + 5
    X = rng.normal(size=(200, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 5.0 + rng.normal(scale=0.01, size=200)

    model = LinearRegressionScratch(learning_rate=0.05, max_iter=2000, solver="gradient_descent")
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = compute_r2(y, y_pred)
    assert r2 > 0.99
    assert np.isclose(model.bias, 5.0, atol=0.2)
    assert np.isclose(model.weights[0], 3.0, atol=0.2)
    assert np.isclose(model.weights[1], -2.0, atol=0.2)


def test_linear_regression_closed_form():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3))
    y = 2.0 * X[:, 0] + 4.0 * X[:, 1] - X[:, 2] + 10.0

    model = LinearRegressionScratch(solver="closed_form").fit(X, y)
    y_pred = model.predict(X)
    assert compute_r2(y, y_pred) > 0.9999
    assert np.isclose(model.bias, 10.0, atol=1e-5)


def test_ridge_regression_scratch_shrinkage():
    rng = np.random.default_rng(42)
    # Multicollinear features
    x1 = rng.normal(size=100)
    x2 = x1 + rng.normal(scale=0.01, size=100)
    X = np.column_stack([x1, x2])
    y = 5.0 * x1 + 3.0 + rng.normal(scale=0.1, size=100)

    # OLS vs Ridge with high lambda
    ols = LinearRegressionScratch(solver="closed_form").fit(X, y)
    ridge = RidgeRegressionScratch(lambda_param=10.0, solver="closed_form").fit(X, y)

    # Ridge weights norm should be smaller than OLS (parameter shrinkage)
    assert np.linalg.norm(ridge.weights) < np.linalg.norm(ols.weights)
    # Intercept must remain unbiased
    assert np.isclose(ridge.bias, 3.0, atol=0.3)


def test_regression_metrics_calculations():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 31.0])

    m = evaluate_regression(y_true, y_pred)
    assert np.isclose(m["MAE"], 5.0 / 3.0, atol=1e-4)
    assert np.isclose(m["MSE"], 9.0 / 3.0, atol=1e-4)
    assert np.isclose(m["RMSE"], np.sqrt(9.0 / 3.0), atol=1e-4)
    assert m["R2"] > 0.9


def test_mean_baseline_regressor():
    y = np.array([10.0, 20.0, 30.0])
    base = MeanBaselineRegressor().fit(np.zeros((3, 1)), y)
    preds = base.predict(np.zeros((2, 1)))
    assert np.allclose(preds, 20.0)
