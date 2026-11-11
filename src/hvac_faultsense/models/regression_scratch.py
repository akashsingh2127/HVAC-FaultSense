"""Ordinary Least Squares (OLS) Linear Regression Implemented from Scratch.

Uses NumPy exclusively for forward pass, gradient computation, and parameter optimization.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class LinearRegressionScratch:
    """Ordinary Least Squares (OLS) Linear Regression implemented with NumPy.

    Optimizes MSE loss using Batch Gradient Descent or analytical Normal Equation.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1500,
        tol: float = 1e-6,
        solver: str = "gradient_descent", # 'gradient_descent' or 'closed_form'
        random_state: Optional[int] = 42,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: List[float] = []
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        """Fit linear regression model parameters.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Continuous target values, shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        if self.solver == "closed_form":
            # Normal equation: w_aug = (X_aug^T X_aug)^(-1) X_aug^T y
            X_aug = np.column_stack([np.ones(n_samples), X])
            try:
                theta = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                theta = np.linalg.pinv(X_aug.T @ X_aug) @ (X_aug.T @ y)
            self.bias = float(theta[0])
            self.weights = theta[1:]
            y_pred = X @ self.weights + self.bias
            self.loss_history = [float(np.mean((y - y_pred) ** 2))]
            self.n_iter_ = 1
            return self

        # Gradient Descent initialization
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = float(np.mean(y))
        self.loss_history = []

        prev_loss = float("inf")
        for iteration in range(self.max_iter):
            # Forward pass: y_hat = X w + b
            y_hat = X @ self.weights + self.bias
            residuals = y_hat - y

            # Mean Squared Error: J = (1/n) * sum(residuals^2)
            loss = float(np.mean(residuals ** 2))
            self.loss_history.append(loss)

            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                self.n_iter_ = iteration + 1
                break
            prev_loss = loss

            # Gradients:
            # dJ/dw = (2/n) * X^T (y_hat - y)
            # dJ/db = (2/n) * sum(y_hat - y)
            grad_w = (2.0 / n_samples) * (X.T @ residuals)
            grad_b = (2.0 / n_samples) * np.sum(residuals)

            # Parameter updates
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        else:
            self.n_iter_ = self.max_iter

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute forward linear predictions."""
        if self.weights is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias

    def calculate_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate prediction residuals: e = y - y_hat."""
        y = np.asarray(y, dtype=np.float64).ravel()
        y_hat = self.predict(X)
        return y - y_hat

