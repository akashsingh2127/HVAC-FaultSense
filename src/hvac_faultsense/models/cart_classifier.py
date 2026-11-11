"""Classification and Regression Tree (CART) Decision Tree from Scratch.

Implements recursive binary splitting, Gini impurity optimization,
minimum leaf constraints, and sample-weight support for boosting.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class TreeNode:
    """Node in the CART decision tree."""
    is_leaf: bool
    predicted_class: Optional[int] = None
    class_probabilities: Optional[np.ndarray] = None
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    gain: float = 0.0


class DecisionTreeClassifierScratch:
    """CART Decision Tree Classifier implemented with NumPy."""

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        criterion: str = "gini", # 'gini' or 'entropy'
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.root: Optional[TreeNode] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    def _compute_impurity(self, y: np.ndarray, sample_weight: np.ndarray) -> float:
        """Calculate Gini impurity or entropy for a subset of weighted samples."""
        total_weight = np.sum(sample_weight)
        if total_weight <= 0:
            return 0.0

        # Calculate class probabilities
        probs = np.zeros(self.n_classes_)
        for c_idx in range(self.n_classes_):
            probs[c_idx] = np.sum(sample_weight[y == c_idx]) / total_weight

        if self.criterion == "entropy":
            eps = 1e-15
            return -float(np.sum([p * np.log2(p + eps) for p in probs if p > 0]))

        # Gini = 1 - sum(p_k^2)
        return float(1.0 - np.sum(probs ** 2))

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find the feature and threshold that maximize impurity reduction."""
        n_samples, n_features = X.shape
        parent_impurity = self._compute_impurity(y, sample_weight)
        total_weight = np.sum(sample_weight)

        best_gain = -1.0
        best_feat = None
        best_thresh = None

        # Feature subsampling if requested
        if self.max_features == "sqrt":
            k_feats = max(1, int(np.sqrt(n_features)))
            feat_indices = rng.choice(n_features, size=k_feats, replace=False)
        else:
            feat_indices = np.arange(n_features)

        for feat in feat_indices:
            x_col = X[:, feat]
            # Use percentiles or unique values for efficient split search
            unique_vals = np.unique(x_col)
            if len(unique_vals) <= 1:
                continue

            # Evaluate split candidates (up to 10 candidate thresholds for speed and precision)
            if len(unique_vals) > 10:
                thresholds = np.percentile(unique_vals, np.linspace(10, 90, 9))
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = x_col <= thresh
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                w_left = np.sum(sample_weight[left_mask])
                w_right = np.sum(sample_weight[right_mask])

                if w_left <= 0 or w_right <= 0:
                    continue

                imp_left = self._compute_impurity(y[left_mask], sample_weight[left_mask])
                imp_right = self._compute_impurity(y[right_mask], sample_weight[right_mask])

                child_impurity = (w_left / total_weight) * imp_left + (w_right / total_weight) * imp_right
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh, max(0.0, best_gain)

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        depth: int,
        rng: np.random.Generator,
    ) -> TreeNode:
        """Recursively construct the CART decision tree."""
        total_weight = np.sum(sample_weight)
        probs = np.zeros(self.n_classes_)
        if total_weight > 0:
            for c_idx in range(self.n_classes_):
                probs[c_idx] = np.sum(sample_weight[y == c_idx]) / total_weight
        pred_class = int(np.argmax(probs))

        # Stopping conditions
        n_samples = len(y)
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.max(probs) >= 0.999 # Pure node
        ):
            return TreeNode(is_leaf=True, predicted_class=pred_class, class_probabilities=probs)

        # Find best split
        feat, thresh, gain = self._best_split(X, y, sample_weight, rng)
        if feat is None or gain <= 1e-7:
            return TreeNode(is_leaf=True, predicted_class=pred_class, class_probabilities=probs)

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1, rng)
        right_child = self._build_tree(X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1, rng)

        # Accumulate feature importance
        if self.feature_importances_ is not None:
            self.feature_importances_[feat] += gain * total_weight

        return TreeNode(
            is_leaf=False,
            predicted_class=pred_class,
            class_probabilities=probs,
            feature_idx=feat,
            threshold=thresh,
            left=left_child,
            right=right_child,
            gain=gain,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "DecisionTreeClassifierScratch":
        """Fit decision tree classifier."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # Map labels to discrete integer indices
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([class_to_idx[val] for val in y], dtype=int)

        n_samples, n_features = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float64) / n_samples
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            sample_weight = sample_weight / np.sum(sample_weight)

        self.feature_importances_ = np.zeros(n_features)
        rng = np.random.default_rng(self.random_state)
        self.root = self._build_tree(X, y_encoded, sample_weight, depth=0, rng=rng)

        # Normalize feature importances
        imp_sum = np.sum(self.feature_importances_)
        if imp_sum > 0:
            self.feature_importances_ /= imp_sum

        return self

    def _predict_row(self, x: np.ndarray, node: TreeNode) -> Tuple[int, np.ndarray]:
        """Traverse tree for a single feature vector."""
        if node.is_leaf or node.feature_idx is None:
            return node.predicted_class, node.class_probabilities

        if x[node.feature_idx] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probability estimates for test samples."""
        if self.root is None:
            raise ValueError("Tree is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        probs = np.zeros((len(X), self.n_classes_))
        for i, row in enumerate(X):
            _, p = self._predict_row(row, self.root)
            probs[i] = p
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for test samples."""
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return self.classes_[pred_indices]

