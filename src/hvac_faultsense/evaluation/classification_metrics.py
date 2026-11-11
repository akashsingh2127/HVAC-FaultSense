"""Supervised Classification Metrics and Majority-Class Baseline.

Calculates multi-class Accuracy, Balanced Accuracy, Precision, Recall, Macro-F1,
Weighted-F1, and Confusion Matrices with support for class imbalance.
"""

from typing import Any, Dict, List, Tuple
import numpy as np


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Compute square confusion matrix where row i represents true class and column j predicted class."""
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        if yt in class_to_idx and yp in class_to_idx:
            cm[class_to_idx[yt], class_to_idx[yp]] += 1

    return cm


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_samples = len(y_true)

    cm = compute_confusion_matrix(y_true, y_pred, classes)

    # Accuracy
    accuracy = float(np.trace(cm) / max(n_samples, 1))

    # Per-class precision, recall, f1, and support
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = int(np.sum(cm[i, :]))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

        per_class[str(cls)] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    balanced_accuracy = macro_recall

    # Weighted F1
    total_supp = sum(supports)
    weighted_f1 = float(np.sum([f * s for f, s in zip(f1s, supports)]) / total_supp) if total_supp > 0 else 0.0

    return {
        "Accuracy": round(accuracy, 4),
        "Balanced_Accuracy": round(balanced_accuracy, 4),
        "Precision": round(macro_precision, 4),
        "Recall": round(macro_recall, 4),
        "Macro-F1": round(macro_f1, 4),
        "Weighted-F1": round(weighted_f1, 4),
        "classes": [str(c) for c in classes],
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


class MajorityBaselineClassifier:
    """Naive baseline predicting the majority class from training data."""

    def __init__(self):
        self.majority_class_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityBaselineClassifier":
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class_ = unique[np.argmax(counts)]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.majority_class_)

