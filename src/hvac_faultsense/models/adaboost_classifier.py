"""AdaBoost Classifier Implemented from Scratch.

Uses multi-class SAMME ensemble formulation with CART weak learners,
exponential weight updates, and learner weight optimization.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from hvac_faultsense.models.cart_classifier import DecisionTreeClassifierScratch


class AdaBoostClassifierScratch:
    """AdaBoost Classifier with SAMME multi-class boosting and decision stump/tree weak learners."""

    def __init__(
        self,
        n_estimators: int = 25,
        learning_rate: float = 0.5,
        weak_learner_max_depth: int = 2,
        random_state: Optional[int] = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.weak_learner_max_depth = weak_learner_max_depth
        self.random_state = random_state

        self.estimators_: List[DecisionTreeClassifierScratch] = []
        self.estimator_weights_: List[float] = []
        self.estimator_errors_: List[float] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifierScratch":
        """Fit ensemble of weak learners via iterative reweighting."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([class_to_idx[val] for val in y], dtype=int)

        n_samples, n_features = X.shape
        # Initialize uniform sample weights
        sample_weights = np.ones(n_samples, dtype=np.float64) / n_samples

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        rng = np.random.default_rng(self.random_state)
        feature_importances_accum = np.zeros(n_features)

        for m in range(self.n_estimators):
            learner_seed = rng.integers(0, 100000)
            weak_learner = DecisionTreeClassifierScratch(
                max_depth=self.weak_learner_max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=learner_seed,
            )
            weak_learner.fit(X, y_encoded, sample_weight=sample_weights)

            # Predictions on training data
            preds_encoded = weak_learner.predict(X)
            incorrect = preds_encoded != y_encoded

            # Weighted error rate
            err_m = float(np.sum(sample_weights[incorrect]) / np.sum(sample_weights))
            self.estimator_errors_.append(err_m)

            # Stopping check: random guessing bound in multi-class SAMME is (1 - 1/K)
            max_err = 1.0 - (1.0 / max(2, self.n_classes_))
            if err_m >= max_err:
                if len(self.estimators_) == 0:
                    # Keep at least 1 estimator
                    self.estimators_.append(weak_learner)
                    self.estimator_weights_.append(1.0)
                break

            if err_m <= 1e-10:
                # Perfect weak learner
                alpha_m = 2.0
                self.estimators_.append(weak_learner)
                self.estimator_weights_.append(alpha_m)
                break

            # Multi-class SAMME alpha update:
            # alpha = lr * (ln((1 - err) / err) + ln(K - 1))
            alpha_m = self.learning_rate * (
                np.log((1.0 - err_m) / err_m) + np.log(max(1, self.n_classes_ - 1))
            )
            self.estimators_.append(weak_learner)
            self.estimator_weights_.append(float(alpha_m))

            # Sample weight update:
            # w_i <- w_i * exp(alpha * I(y_i != h_m(x_i)))
            sample_weights *= np.exp(alpha_m * incorrect.astype(float))
            # Normalize sample weights
            sample_weights /= np.sum(sample_weights)

            # Accumulate feature importances
            if weak_learner.feature_importances_ is not None:
                feature_importances_accum += alpha_m * weak_learner.feature_importances_

        # Normalize overall feature importances
        if np.sum(feature_importances_accum) > 0:
            self.feature_importances_ = feature_importances_accum / np.sum(feature_importances_accum)
        else:
            self.feature_importances_ = np.zeros(n_features)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute final weighted ensemble prediction."""
        if len(self.estimators_) == 0:
            raise ValueError("AdaBoost is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)

        # Accumulator: votes[sample_i, class_k]
        votes = np.zeros((n_samples, self.n_classes_))

        for learner, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = learner.predict(X) # Returns class indices
            for i, p in enumerate(preds):
                votes[i, p] += alpha

        best_indices = np.argmax(votes, axis=1)
        return self.classes_[best_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute softmax ensemble decision confidence."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        votes = np.zeros((n_samples, self.n_classes_))

        for learner, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = learner.predict(X)
            for i, p in enumerate(preds):
                votes[i, p] += alpha

        # Softmax normalisation
        max_v = np.max(votes, axis=1, keepdims=True)
        exp_v = np.exp(votes - max_v)
        return exp_v / np.sum(exp_v, axis=1, keepdims=True)

