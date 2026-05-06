"""
Concept Drift Detector

Detects changes in anomaly score distribution
using Page-Hinkley drift detection.
"""

import numpy as np


class DriftDetector:

    def __init__(
        self,
        delta=0.01,
        threshold=50,
        alpha=0.999
    ):

        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        self.reset()

    # ─────────────────────────────────────────────────────

    def reset(self):

        self.mean = 0.0
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0
        self.n_samples = 0

    # ─────────────────────────────────────────────────────

    def update(self, value):

        self.n_samples += 1

        # Running mean
        self.mean = (
            self.alpha * self.mean
            + (1 - self.alpha) * value
        )

        # Page-Hinkley statistic
        self.cumulative_sum += (
            value
            - self.mean
            - self.delta
        )

        self.min_cumulative_sum = min(
            self.min_cumulative_sum,
            self.cumulative_sum
        )

        drift_score = (
            self.cumulative_sum
            - self.min_cumulative_sum
        )

        if drift_score > self.threshold:

            self.reset()

            return True

        return False