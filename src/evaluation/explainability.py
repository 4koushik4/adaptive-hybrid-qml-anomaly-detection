"""
Explainability Module

Provides feature importance analysis
for anomaly predictions.
"""

import numpy as np
import pandas as pd


class ExplainabilityEngine:

    def __init__(self):

        pass

    # ─────────────────────────────────────────────────────

    def feature_importance(
        self,
        feature_names,
        values
    ):

        importance = {}

        total = np.sum(np.abs(values))

        for name, value in zip(feature_names, values):

            score = abs(value) / (total + 1e-8)

            importance[name] = round(score, 4)

        return importance

    # ─────────────────────────────────────────────────────

    def explain_prediction(
        self,
        feature_names,
        values,
        anomaly_score
    ):

        importance = self.feature_importance(
            feature_names,
            values
        )

        explanation = {
            "anomaly_score": anomaly_score,
            "important_features": importance
        }

        return explanation

    # ─────────────────────────────────────────────────────

    def print_explanation(
        self,
        explanation
    ):

        print("\n========== EXPLANATION ==========")

        print(f"Anomaly Score: {explanation['anomaly_score']}")

        print("\nFeature Importance:")

        for feature, score in explanation[
            "important_features"
        ].items():

            print(f"{feature}: {score}")

        print("=================================\n")