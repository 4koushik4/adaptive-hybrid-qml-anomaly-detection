"""
Data Cleaner

Handles:
- Missing values
- Infinite values
- Outlier clipping
"""

import pandas as pd
import numpy as np


class DataCleaner:

    def __init__(self):

        self.medians = {}
        self.means = {}
        self.stds = {}

        self.fitted = False

    # ─────────────────────────────────────────────────────

    def fit(
        self,
        df,
        feature_columns
    ):

        self.feature_columns = feature_columns

        for col in feature_columns:

            self.medians[col] = df[col].median()

            self.means[col] = df[col].mean()

            self.stds[col] = df[col].std()

        self.fitted = True

    # ─────────────────────────────────────────────────────

    def transform(
        self,
        df
    ):

        if not self.fitted:

            raise ValueError(
                "Cleaner not fitted."
            )

        df = df.copy()

        for col in self.feature_columns:

            # Replace inf
            df[col] = df[col].replace(
                [np.inf, -np.inf],
                np.nan
            )

            # Fill NaN
            df[col] = df[col].fillna(
                self.medians[col]
            )

            # Clip outliers
            lower = (
                self.means[col]
                - 4 * self.stds[col]
            )

            upper = (
                self.means[col]
                + 4 * self.stds[col]
            )

            df[col] = df[col].clip(
                lower,
                upper
            )

        return df

    # ─────────────────────────────────────────────────────

    def fit_transform(
        self,
        df,
        feature_columns
    ):

        self.fit(
            df,
            feature_columns
        )

        return self.transform(df)