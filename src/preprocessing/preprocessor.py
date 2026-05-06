"""
Unified Preprocessor

Supports:
- Simulated datasets
- Real-world datasets
"""

import pandas as pd
import numpy as np

from src.preprocessing.cleaner import (
    DataCleaner
)

from src.preprocessing.feature_extractor import (
    FeatureExtractor
)

from src.preprocessing.encoders import (
    angle_encoding
)


class Preprocessor:

    def __init__(self):

        self.cleaner = DataCleaner()

        self.extractor = FeatureExtractor(
            n_components=4
        )

        self.feature_columns = None

        self.fitted = False

    # ─────────────────────────────────────────────────────

    def detect_feature_columns(
        self,
        df
    ):

        return [

            col

            for col in df.columns

            if col.startswith("feature_")
        ]

    # ─────────────────────────────────────────────────────

    def fit(
        self,
        df
    ):

        self.feature_columns = (
            self.detect_feature_columns(df)
        )

        cleaned = self.cleaner.fit_transform(

            df,

            self.feature_columns
        )

        X = cleaned[
            self.feature_columns
        ].values

        self.extractor.fit(X)

        self.fitted = True

    # ─────────────────────────────────────────────────────

    def transform(
        self,
        df
    ):

        if not self.fitted:

            raise RuntimeError(
                "Preprocessor not fitted."
            )

        cleaned = self.cleaner.transform(df)

        X = cleaned[
            self.feature_columns
        ].values

        X = self.extractor.transform(X)

        X = np.array([

            angle_encoding(x)

            for x in X
        ])

        y = df["label"].values

        meta = df["source"].values

        return X, y, meta

    # ─────────────────────────────────────────────────────

    def fit_transform(
        self,
        df
    ):

        self.fit(df)

        return self.transform(df)

    # ─────────────────────────────────────────────────────

    def transform_single(
        self,
        record
    ):

        df = pd.DataFrame([record])

        cleaned = self.cleaner.transform(df)

        X = cleaned[
            self.feature_columns
        ].values

        X = self.extractor.transform(X)

        X = angle_encoding(X[0])

        return X