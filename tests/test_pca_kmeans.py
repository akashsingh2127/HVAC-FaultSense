"""Unit tests for PCA, K-Means, and GMM from Scratch."""

import numpy as np
import pytest

from hvac_faultsense.models.pca_scratch import PCAScratch
from hvac_faultsense.models.kmeans_scratch import KMeansScratch
from hvac_faultsense.models.gmm_em_scratch import GaussianMixtureScratch


def test_pca_scratch():
    rng = np.random.default_rng(42)
    # Generate data with high variance along first dimension
    x1 = rng.normal(scale=10.0, size=200)
    x2 = rng.normal(scale=1.0, size=200)
    X = np.column_stack([x1, x2])

    pca = PCAScratch(n_components=2).fit(X)
    X_trans = pca.transform(X)

    assert X_trans.shape == (200, 2)
    # PC1 should capture majority of variance (> 90%)
    assert pca.explained_variance_ratio_[0] > 0.90
    assert np.isclose(np.sum(pca.explained_variance_ratio_), 1.0)

    # Check reconstruction
    X_recon = pca.inverse_transform(X_trans)
    assert np.allclose(X, X_recon, atol=1e-5)

    # Check reconstruction error metric
    recon_err = pca.compute_reconstruction_error(X)
    assert recon_err >= 0.0
    assert np.isclose(recon_err, 0.0, atol=1e-5)


def test_kmeans_scratch_clusters():
    rng = np.random.default_rng(42)
    # Three well-separated Gaussian clusters
    c1 = rng.normal(loc=[-10.0, -10.0], scale=1.0, size=(50, 2))
    c2 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(50, 2))
    c3 = rng.normal(loc=[10.0, 10.0], scale=1.0, size=(50, 2))
    X = np.vstack([c1, c2, c3])

    kmeans = KMeansScratch(n_clusters=3, max_iter=50, n_init=3, random_state=42).fit(X)
    assert len(kmeans.cluster_centers_) == 3
    assert len(np.unique(kmeans.labels_)) == 3
    assert kmeans.inertia_ > 0.0

    preds = kmeans.predict(X)
    assert len(preds) == len(X)


def test_gmm_scratch_responsibilities():
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[-5.0], scale=0.5, size=(40, 1))
    c2 = rng.normal(loc=[5.0], scale=0.5, size=(40, 1))
    X = np.vstack([c1, c2])

    gmm = GaussianMixtureScratch(n_components=2, max_iter=30, random_state=42).fit(X)
    probs = gmm.predict_proba(X)

    # Responsibilities must sum to 1 across components for each sample
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    # Clear separation between the two components
    preds = gmm.predict(X)
    assert len(np.unique(preds)) == 2

