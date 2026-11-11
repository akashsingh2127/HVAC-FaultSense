"""Gaussian Mixture Model (GMM) with Expectation-Maximization (EM) from Scratch.

Implements the E-step, M-step, log-likelihood tracking, numerical stability
regularization, and BIC/AIC information criteria using NumPy.
"""

from typing import Optional, Tuple
import numpy as np


class GaussianMixtureScratch:
    """Gaussian Mixture Model fitted via Expectation-Maximization."""

    def __init__(
        self,
        n_components: int = 4,
        max_iter: int = 80,
        tol: float = 1e-4,
        reg_covar: float = 1e-4,
        random_state: Optional[int] = 42,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None # Shape (K,)
        self.means_: Optional[np.ndarray] = None # Shape (K, d)
        self.covariances_: Optional[np.ndarray] = None # Shape (K, d, d)
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = -float("inf")

    def _estimate_log_gaussian_prob(self, X: np.ndarray) -> np.ndarray:
        """Compute log probability density under each Gaussian component.

        Returns:
            log_prob: shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            mu = self.means_[k]
            cov = self.covariances_[k] + self.reg_covar * np.eye(n_features)

            # Use Cholesky decomposition for stable determinant and solve
            try:
                L = np.linalg.cholesky(cov)
                # log det(cov) = 2 * sum(log(diag(L)))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                # (X - mu) @ cov^(-1) @ (X - mu)^T
                diff = X - mu
                # Solve L y = diff.T => y = L^(-1) diff.T
                sol = np.linalg.solve(L, diff.T)
                quad = np.sum(sol ** 2, axis=0)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if Cholesky fails
                diff = X - mu
                inv_cov = np.linalg.pinv(cov)
                sign, log_det = np.linalg.slogdet(cov)
                quad = np.sum((diff @ inv_cov) * diff, axis=1)

            log_prob[:, k] = -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + quad)

        return log_prob

    def fit(self, X: np.ndarray) -> "GaussianMixtureScratch":
        """Fit GMM parameters using the Expectation-Maximization (EM) algorithm."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # 1. Initialize weights, means, and covariances
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)
        random_indices = rng.choice(n_samples, size=self.n_components, replace=False)
        self.means_ = X[random_indices].copy()

        # Initialize covariances to empirical feature covariance
        init_cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
        self.covariances_ = np.array([init_cov.copy() for _ in range(self.n_components)])

        prev_log_likelihood = -float("inf")

        for iteration in range(self.max_iter):
            # ===== E-STEP: Estimate responsibilities gamma_ik =====
            log_prob = self._estimate_log_gaussian_prob(X) # (n, K)
            weighted_log_prob = log_prob + np.log(self.weights_ + 1e-12)

            # Log-sum-exp trick for numerical stability
            max_log = np.max(weighted_log_prob, axis=1, keepdims=True)
            sum_exp = np.sum(np.exp(weighted_log_prob - max_log), axis=1, keepdims=True)
            log_likelihood_per_sample = max_log + np.log(sum_exp)
            log_likelihood = float(np.sum(log_likelihood_per_sample))

            # gamma_ik = exp(weighted_log_prob - log_likelihood_per_sample)
            responsibilities = np.exp(weighted_log_prob - log_likelihood_per_sample)

            # Check convergence
            change = abs(log_likelihood - prev_log_likelihood)
            if change < self.tol and iteration > 0:
                self.converged_ = True
                self.n_iter_ = iteration + 1
                self.lower_bound_ = log_likelihood
                break
            prev_log_likelihood = log_likelihood

            # ===== M-STEP: Update parameters based on responsibilities =====
            Nk = np.sum(responsibilities, axis=0) + 1e-10 # (K,)
            self.weights_ = Nk / n_samples
            self.means_ = (responsibilities.T @ X) / Nk[:, np.newaxis]

            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_diff = responsibilities[:, k : k + 1] * diff
                self.covariances_[k] = (weighted_diff.T @ diff) / Nk[k] + (self.reg_covar * np.eye(n_features))
        else:
            self.n_iter_ = self.max_iter
            self.lower_bound_ = prev_log_likelihood

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior cluster responsibilities."""
        if self.means_ is None:
            raise ValueError("GMM is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        log_prob = self._estimate_log_gaussian_prob(X)
        weighted_log_prob = log_prob + np.log(self.weights_ + 1e-12)
        max_log = np.max(weighted_log_prob, axis=1, keepdims=True)
        exp_vals = np.exp(weighted_log_prob - max_log)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the most likely Gaussian component."""
        return np.argmax(self.predict_proba(X), axis=1)

    def compute_bic(self, X: np.ndarray) -> float:
        """Compute Bayesian Information Criterion (BIC).

        BIC = -2 * ln(L) + p * ln(n), where lower is better.
        """
        n_samples, n_features = X.shape
        # Total free parameters: weights (K - 1) + means (K * d) + covariances (K * d * (d + 1) / 2)
        n_params = (self.n_components - 1) + (self.n_components * n_features) + (self.n_components * n_features * (n_features + 1) // 2)
        return float(-2.0 * self.lower_bound_ + n_params * np.log(n_samples))

    def compute_aic(self, X: np.ndarray) -> float:
        """Compute Akaike Information Criterion (AIC).

        AIC = -2 * ln(L) + 2 * p, where lower is better.
        """
        n_samples, n_features = X.shape
        n_params = (self.n_components - 1) + (self.n_components * n_features) + (self.n_components * n_features * (n_features + 1) // 2)
        return float(-2.0 * self.lower_bound_ + 2.0 * n_params)

