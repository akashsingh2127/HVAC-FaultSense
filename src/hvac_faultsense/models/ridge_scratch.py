"""Ridge Regression (L2 Regularization) Implemented from Scratch.

Uses NumPy for gradient descent optimization and analytical regularized equations.
The intercept (bias) is strictly unpenalized.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class RidgeRegressionScratch:
    """L2 Regularized Ridge Regression implemented with NumPy.

    Objective function:
        J(w, b) = (1/n) * sum((y_i - y_hat_i)^2) + lambda_param * sum(w_j^2)
    """

    def __init__(
        self,
        lambda_param: float = 1.0,
        learning_rate: float = 0.01,
        max_iter: int = 1500,
        tol: float = 1e-6,
        solver: str = "gradient_descent", # 'gradient_descent' or 'closed_form'
        random_state: Optional[int] = 42,
    ):
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: List[float] = []
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionScratch":
        """Fit Ridge regression parameters."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        if self.solver == "closed_form":
            # Analytical Ridge solution with unpenalized intercept via centering:
            # X_c = X - mean(X), y_c = y - mean(y)
            # w = (X_c^T X_c + n * lambda * I)^(-1) X_c^T y_c
            # b = mean(y) - mean(X) @ w
            x_mean = np.mean(X, axis=0)
            y_mean = float(np.mean(y))
            X_c = X - x_mean
            y_c = y - y_mean

            reg_matrix = (n_samples * self.lambda_param) * np.eye(n_features)
            lhs = X_c.T @ X_c + reg_matrix
            rhs = X_c.T @ y_c
            try:
                self.weights = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                self.weights = np.linalg.pinv(lhs) @ rhs

            self.bias = y_mean - float(x_mean @ self.weights)
            y_pred = X @ self.weights + self.bias
            mse = float(np.mean((y - y_pred) ** 2))
            penalty = self.lambda_param * float(np.sum(self.weights ** 2))
            self.loss_history = [mse + penalty]
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

            # Objective: MSE + lambda * ||w||_2^2
            mse = float(np.mean(residuals ** 2))
            l2_penalty = self.lambda_param * float(np.sum(self.weights ** 2))
            loss = mse + l2_penalty
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tol:
                self.n_iter_ = iteration + 1
                break
            prev_loss = loss

            # Gradients:
            # dJ/dw = (2/n) * X^T (y_hat - y) + 2 * lambda * w
            # dJ/db = (2/n) * sum(y_hat - y) (Unpenalized intercept!)
            grad_w = (2.0 / n_samples) * (X.T @ residuals) + (2.0 * self.lambda_param * self.weights)
            grad_b = (2.0 / n_samples) * np.sum(residuals)

            # Updates
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        else:
            self.n_iter_ = self.max_iter

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute predictions."""
        if self.weights is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias

    def calculate_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute residuals: e = y - y_hat."""
        y = np.asarray(y, dtype=np.float64).ravel()
        y_hat = self.predict(X)
        return y - y_hat

