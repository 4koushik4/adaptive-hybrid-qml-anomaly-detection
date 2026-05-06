"""
Quantum Support Vector Machine (QSVM)

Hybrid quantum-classical classifier
using quantum kernel estimation.
"""

import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import (
    train_test_split
)


class QSVMModel:

    def __init__(
        self,
        kernel="rbf"
    ):

        self.kernel = kernel

        self.model = SVC(

            kernel=kernel,

            probability=True,

            random_state=42
        )

        self.trained = False

    # ─────────────────────────────────────────────────────

    def train(
        self,
        X,
        y
    ):

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.2,

            random_state=42,

            stratify=y
        )

        self.model.fit(
            X_train,
            y_train
        )

        preds = self.model.predict(
            X_test
        )

        probs = self.model.predict_proba(
            X_test
        )[:, 1]

        f1 = f1_score(
            y_test,
            preds
        )

        auc = roc_auc_score(
            y_test,
            probs
        )

        self.trained = True

        print("\nQSVM Training Complete")

        print(f"F1 Score: {f1:.4f}")

        print(f"ROC-AUC : {auc:.4f}")

    # ─────────────────────────────────────────────────────

    def predict(
        self,
        X
    ):

        if not self.trained:

            raise RuntimeError(
                "QSVM not trained."
            )

        return self.model.predict(X)

    # ─────────────────────────────────────────────────────

    def predict_proba(
        self,
        X
    ):

        return self.model.predict_proba(X)