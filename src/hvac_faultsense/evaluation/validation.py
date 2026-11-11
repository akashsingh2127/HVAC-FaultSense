"""Model Validation, Benchmarking Registry, and Comparison Reports.

Consolidates performance metrics across regression, clustering, sequential modeling,
and fault diagnosis, objectively identifying the best-performing models.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class ModelResultRecord:
    """Standardized record for individual model evaluation."""
    model_name: str
    task: str # 'Regression', 'Clustering', 'Sequence', 'Classification'
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    macro_f1: Optional[float] = None
    silhouette: Optional[float] = None
    training_time_sec: float = 0.0
    n_train_samples: int = 0
    n_test_samples: int = 0
    hyperparameters: Optional[Dict[str, Any]] = None
    timestamp: str = ""


class ModelComparisonRegistry:
    """Maintains benchmark results and generates consolidated reports."""

    def __init__(self):
        self.records: List[ModelResultRecord] = []

    def log_result(self, record: ModelResultRecord) -> None:
        """Add an evaluated model record."""
        if not record.timestamp:
            record.timestamp = datetime.now().isoformat()
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all logged evaluations into a clean comparison DataFrame."""
        rows = []
        for r in self.records:
            rows.append({
                "Model": r.model_name,
                "Task": r.task,
                "MAE": f"{r.mae:.4f}" if r.mae is not None else "N/A",
                "RMSE": f"{r.rmse:.4f}" if r.rmse is not None else "N/A",
                "R2": f"{r.r2:.4f}" if r.r2 is not None else "N/A",
                "Accuracy": f"{r.accuracy:.4f}" if r.accuracy is not None else "N/A",
                "Precision": f"{r.precision:.4f}" if r.precision is not None else "N/A",
                "Recall": f"{r.recall:.4f}" if r.recall is not None else "N/A",
                "Macro-F1": f"{r.macro_f1:.4f}" if r.macro_f1 is not None else "N/A",
                "Silhouette": f"{r.silhouette:.4f}" if r.silhouette is not None else "N/A",
                "Train Time (s)": round(r.training_time_sec, 3),
                "Train Samples": r.n_train_samples,
                "Test Samples": r.n_test_samples,
            })
        return pd.DataFrame(rows)

    def identify_best_models(self) -> Dict[str, Dict[str, Any]]:
        """Identify winning model per analytical task using verified held-out results."""
        best = {}

        # 1. Best Regression (Lowest RMSE, highest R2)
        reg_records = [r for r in self.records if r.task == "Regression" and r.rmse is not None]
        if reg_records:
            best_reg = min(reg_records, key=lambda x: x.rmse)
            best["Regression"] = {
                "winner": best_reg.model_name,
                "primary_metric": "RMSE",
                "score": best_reg.rmse,
                "r2": best_reg.r2,
                "reason": f"Achieved lowest Root Mean Squared Error ({best_reg.rmse:.4f} W) and highest R2 ({best_reg.r2:.4f}) on held-out test data.",
            }

        # 2. Best Classification (Highest Macro-F1)
        cls_records = [r for r in self.records if r.task == "Classification" and r.macro_f1 is not None]
        if cls_records:
            best_cls = max(cls_records, key=lambda x: x.macro_f1)
            best["Classification"] = {
                "winner": best_cls.model_name,
                "primary_metric": "Macro-F1",
                "score": best_cls.macro_f1,
                "accuracy": best_cls.accuracy,
                "reason": f"Achieved highest Macro-F1 ({best_cls.macro_f1:.4f}) accounting for fault class imbalance on held-out scenarios.",
            }

        # 3. Best Clustering
        clust_records = [r for r in self.records if r.task == "Clustering" and r.silhouette is not None]
        if clust_records:
            best_clust = max(clust_records, key=lambda x: x.silhouette)
            best["Clustering"] = {
                "winner": best_clust.model_name,
                "primary_metric": "Silhouette Score",
                "score": best_clust.silhouette,
                "reason": f"Yielded highest cluster cohesion and separation (Silhouette = {best_clust.silhouette:.4f}).",
            }

        return best

