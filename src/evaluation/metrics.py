"""
Performance Metrics Module

Computes:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Latency Statistics
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


class PerformanceMetrics:

    def __init__(self):

        self.y_true = []
        self.y_pred = []
        self.y_scores = []
        self.latencies = []

    # ─────────────────────────────────────────────────────

    def update(
        self,
        label,
        prediction,
        score,
        latency
    ):

        self.y_true.append(label)
        self.y_pred.append(prediction)
        self.y_scores.append(score)
        self.latencies.append(latency)

    # ─────────────────────────────────────────────────────

    def compute(self):

        if len(self.y_true) == 0:
            return {}

        metrics = {}

        metrics["accuracy"] = accuracy_score(
            self.y_true,
            self.y_pred
        )

        metrics["precision"] = precision_score(
            self.y_true,
            self.y_pred,
            zero_division=0
        )

        metrics["recall"] = recall_score(
            self.y_true,
            self.y_pred,
            zero_division=0
        )

        metrics["f1_score"] = f1_score(
            self.y_true,
            self.y_pred,
            zero_division=0
        )

        # ROC AUC only if both classes exist
        if len(set(self.y_true)) > 1:

            metrics["roc_auc"] = roc_auc_score(
                self.y_true,
                self.y_scores
            )

        cm = confusion_matrix(
            self.y_true,
            self.y_pred
        )

        metrics["confusion_matrix"] = cm.tolist()

        metrics["avg_latency_ms"] = np.mean(
            self.latencies
        )

        metrics["max_latency_ms"] = np.max(
            self.latencies
        )

        metrics["min_latency_ms"] = np.min(
            self.latencies
        )

        return metrics

    # ─────────────────────────────────────────────────────

    def print_report(self):

        metrics = self.compute()

        print("\n========== PERFORMANCE REPORT ==========")

        for key, value in metrics.items():

            print(f"{key}: {value}")

        print("========================================\n")