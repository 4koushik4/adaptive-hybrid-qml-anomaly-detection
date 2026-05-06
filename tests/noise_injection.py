"""
Noise Injection & Stress Testing

Tests robustness of the QML model under:
- Gaussian noise
- Missing data
- Adversarial perturbations
- Sensor drift
- Bit-flip corruption
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    roc_auc_score
)

from src.qml.quantum_circuit import (
    batch_predict_proba
)


class NoiseInjector:

    # ─────────────────────────────────────────────────────

    @staticmethod
    def gaussian_noise(
        X,
        std=0.1
    ):

        noise = np.random.normal(
            0,
            std,
            X.shape
        )

        return np.clip(
            X + noise,
            0,
            np.pi
        )

    # ─────────────────────────────────────────────────────

    @staticmethod
    def missing_data(
        X,
        missing_rate=0.2
    ):

        X_noisy = X.copy()

        mask = np.random.rand(
            *X.shape
        ) < missing_rate

        col_means = np.mean(
            X,
            axis=0
        )

        for i in range(X.shape[1]):

            X_noisy[
                mask[:, i],
                i
            ] = col_means[i]

        return X_noisy

    # ─────────────────────────────────────────────────────

    @staticmethod
    def adversarial_noise(
        X,
        epsilon=0.2
    ):

        signs = np.random.choice(
            [-1, 1],
            size=X.shape
        )

        return np.clip(
            X + epsilon * signs,
            0,
            np.pi
        )

    # ─────────────────────────────────────────────────────

    @staticmethod
    def sensor_drift(
        X,
        drift_rate=0.05
    ):

        drift = np.linspace(

            0,

            drift_rate * np.pi,

            X.shape[0]

        ).reshape(-1, 1)

        return np.clip(
            X + drift,
            0,
            np.pi
        )

    # ─────────────────────────────────────────────────────

    @staticmethod
    def bit_flip(
        X,
        flip_rate=0.05
    ):

        X_flip = X.copy()

        mask = np.random.rand(
            *X.shape
        ) < flip_rate

        X_flip[mask] = np.pi - X_flip[mask]

        return X_flip


# ─────────────────────────────────────────────────────────────
# Stress Tester
# ─────────────────────────────────────────────────────────────

class StressTester:

    def __init__(
        self,
        detector
    ):

        self.detector = detector

    # ─────────────────────────────────────────────────────

    def evaluate(
        self,
        X,
        y,
        label
    ):

        scores = batch_predict_proba(
            X,
            self.detector.weights
        )

        preds = (
            scores >= self.detector.threshold
        ).astype(int)

        f1 = f1_score(
            y,
            preds
        )

        auc = roc_auc_score(
            y,
            scores
        )

        return {

            "condition": label,

            "f1_score": round(f1, 4),

            "roc_auc": round(auc, 4)
        }

    # ─────────────────────────────────────────────────────

    def run_all(
        self,
        X,
        y
    ):

        injector = NoiseInjector()

        results = []

        # Clean baseline
        results.append(
            self.evaluate(
                X,
                y,
                "Clean"
            )
        )

        # Gaussian
        X_g = injector.gaussian_noise(
            X,
            std=0.15
        )

        results.append(
            self.evaluate(
                X_g,
                y,
                "Gaussian Noise"
            )
        )

        # Missing data
        X_m = injector.missing_data(
            X,
            missing_rate=0.25
        )

        results.append(
            self.evaluate(
                X_m,
                y,
                "Missing Data"
            )
        )

        # Adversarial
        X_a = injector.adversarial_noise(
            X,
            epsilon=0.3
        )

        results.append(
            self.evaluate(
                X_a,
                y,
                "Adversarial Noise"
            )
        )

        # Drift
        X_d = injector.sensor_drift(
            X,
            drift_rate=0.05
        )

        results.append(
            self.evaluate(
                X_d,
                y,
                "Sensor Drift"
            )
        )

        # Bit flip
        X_b = injector.bit_flip(
            X,
            flip_rate=0.1
        )

        results.append(
            self.evaluate(
                X_b,
                y,
                "Bit Flip"
            )
        )

        return pd.DataFrame(results)