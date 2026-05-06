"""
Feature Extractor

Performs:
- Normalization
- PCA dimensionality reduction
- Quantum-ready feature generation
"""

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class FeatureExtractor:

    def __init__(
        self,
        n_components=4
    ):

        self.n_components = n_components

        self.scaler = MinMaxScaler(
            feature_range=(0, np.pi)
        )

        self.pca = PCA(
            n_components=n_components
        )

        self.fitted = False

    # ─────────────────────────────────────────────────────

    def fit(
        self,
        X
    ):

        X_scaled = self.scaler.fit_transform(X)

        self.pca.fit(X_scaled)

        self.fitted = True

    # ─────────────────────────────────────────────────────

    def transform(
        self,
        X
    ):

        if not self.fitted:

            raise ValueError(
                "FeatureExtractor not fitted."
            )

        X_scaled = self.scaler.transform(X)

        X_pca = self.pca.transform(X_scaled)

        X_pca = np.clip(
            X_pca,
            -np.pi,
            np.pi
        )

        X_pca = (
            X_pca + np.pi
        ) / 2

        return X_pca.astype(np.float32)

    # ─────────────────────────────────────────────────────

    def fit_transform(
        self,
        X
    ):

        self.fit(X)

        return self.transform(X)