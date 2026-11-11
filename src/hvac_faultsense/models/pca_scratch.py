"""Principal Component Analysis (PCA) Implemented from Scratch.

Uses NumPy for covariance matrix estimation, eigenvalue decomposition,
and projection of multidimensional HVAC sensor states.
"""

from typing import Optional, Tuple
import numpy as np


class PCAScratch:
    """Principal Component Analysis (PCA) implemented via NumPy SVD / Eigendecomposition."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None # Eigenvectors (n_components, n_features)
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCAScratch":
        """Compute principal components on training features."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if self.n_components > min(n_samples, n_features):
            raise ValueError(f"n_components={self.n_components} cannot exceed min(n_samples, n_features)={min(n_samples, n_features)}")

        # 1. Center the features
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Covariance matrix: (1 / (n - 1)) * X_c^T X_c
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # 3. Eigendecomposition of symmetric positive semi-definite matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[idx]
        sorted_eigenvectors = eigenvectors[:, idx]

        # 5. Retain top k components
        self.explained_variance_ = sorted_eigenvalues[: self.n_components]
        total_var = np.sum(sorted_eigenvalues)
        if total_var > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_var
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)

        # Components shape: (n_components, n_features)
        self.components_ = sorted_eigenvectors[:, : self.n_components].T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project features onto the principal components."""
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCA is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and transform X into principal component space."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """Reconstruct original sensor features from principal components."""
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCA is not fitted.")
        return (X_pca @ self.components_) + self.mean_

