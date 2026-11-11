"""K-Means Clustering Implemented from Scratch.

Includes k-means++ initialization, Euclidean distance metric,
inertia computation, convergence criteria, and multiple random restarts.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


class KMeansScratch:
    """K-Means clustering algorithm implemented with NumPy."""

    def __init__(
        self,
        n_clusters: int = 4,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 5,
        init: str = "k-means++", # 'k-means++' or 'random'
        random_state: Optional[int] = 42,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Initialize centroids using random sampling or k-means++."""
        n_samples, n_features = X.shape

        if self.init == "random":
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()

        # k-means++ initialization
        centroids = [X[rng.integers(0, n_samples)]]
        for _ in range(1, self.n_clusters):
            # Compute distance from each sample to closest existing centroid
            dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dist_sq / np.sum(dist_sq) if np.sum(dist_sq) > 0 else np.ones(n_samples) / n_samples
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids, dtype=np.float64)

    def _run_single_kmeans(self, X: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Execute one complete run of K-Means until convergence."""
        n_samples = X.shape[0]
        centroids = self._init_centroids(X, rng)

        labels = np.zeros(n_samples, dtype=int)
        for iteration in range(self.max_iter):
            # 1. Assignment step: find nearest centroid for every sample
            # distances shape: (n_samples, n_clusters)
            distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # 2. Update step: recompute centroids as cluster means
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_pts = X[labels == k]
                if len(cluster_pts) > 0:
                    new_centroids[k] = np.mean(cluster_pts, axis=0)
                else:
                    # Handle empty cluster by reassigning to a random point
                    new_centroids[k] = X[rng.integers(0, n_samples)]

            # 3. Check convergence: shift in centroid positions
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        # Compute final inertia: sum of squared distances to closest centroid
        final_distances = np.min(np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        inertia = float(np.sum(final_distances))

        return centroids, labels, inertia, iteration + 1

    def fit(self, X: np.ndarray) -> "KMeansScratch":
        """Fit K-Means across n_init initializations, retaining the lowest inertia."""
        X = np.asarray(X, dtype=np.float64)
        best_inertia = float("inf")
        best_centroids = None
        best_labels = None
        best_n_iter = 0

        rng_base = np.random.default_rng(self.random_state)
        for i in range(self.n_init):
            run_seed = rng_base.integers(0, 100000)
            run_rng = np.random.default_rng(run_seed)
            centroids, labels, inertia, n_iter = self._run_single_kmeans(X, run_rng)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign cluster indices to new observations."""
        if self.cluster_centers_ is None:
            raise ValueError("KMeans model is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)

