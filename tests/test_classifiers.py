"""Unit tests for CART Decision Tree, AdaBoost Classifier, and Classification Metrics."""

import numpy as np
import pytest

from hvac_faultsense.models.cart_classifier import DecisionTreeClassifierScratch
from hvac_faultsense.models.adaboost_classifier import AdaBoostClassifierScratch
from hvac_faultsense.evaluation.classification_metrics import (
    MajorityBaselineClassifier,
    evaluate_classification,
)


def test_cart_classifier_separable_data():
    rng = np.random.default_rng(42)
    # Linearly separable classes
    c0 = rng.normal(loc=[0.0, 0.0], scale=0.5, size=(40, 2))
    c1 = rng.normal(loc=[5.0, 5.0], scale=0.5, size=(40, 2))
    X = np.vstack([c0, c1])
    y = np.array(["Class0"] * 40 + ["Class1"] * 40)

    cart = DecisionTreeClassifierScratch(max_depth=4).fit(X, y)
    preds = cart.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.98
    assert cart.feature_importances_ is not None


def test_adaboost_classifier_boosting():
    rng = np.random.default_rng(42)
    # Linearly separated classes with some non-linear boundary
    c0 = rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(50, 2))
    c1 = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(50, 2))
    X = np.vstack([c0, c1])
    y = np.array(["A"] * 50 + ["B"] * 50)

    ada = AdaBoostClassifierScratch(n_estimators=10, weak_learner_max_depth=2, random_state=42).fit(X, y)
    preds = ada.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.90
    assert len(ada.estimators_) > 0


def test_classification_metrics():
    y_true = np.array(["FaultA", "FaultA", "FaultB", "FaultB"])
    y_pred = np.array(["FaultA", "FaultB", "FaultB", "FaultB"])

    res = evaluate_classification(y_true, y_pred)
    assert np.isclose(res["Accuracy"], 0.75)
    assert "Macro-F1" in res
    assert "Balanced_Accuracy" in res
    assert "confusion_matrix" in res


def test_majority_baseline_classifier():
    y = np.array(["FaultA", "FaultB", "FaultA", "FaultA"])
    base = MajorityBaselineClassifier().fit(np.zeros((4, 1)), y)
    preds = base.predict(np.zeros((3, 1)))
    assert np.all(preds == "FaultA")
