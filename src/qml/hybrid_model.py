"""
Hybrid Quantum-Classical Model

Architecture:
Classical Features
        ↓
Quantum Feature Extraction
        ↓
Classical Neural Network
        ↓
Anomaly Prediction
"""

import numpy as np

from sklearn.neural_network import (
    MLPClassifier
)

from sklearn.metrics import (
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import (
    train_test_split
)

from src.qml.quantum_circuit import (
    batch_predict_proba,
    initialize_weights
)


class HybridQMLModel:

    def __init__(self):

        self.weights = initialize_weights()

        self.classifier = MLPClassifier(

            hidden_layer_sizes=(64, 32),

            activation="relu",

            max_iter=200,

            random_state=42
        )

        self.trained = False

    # ─────────────────────────────────────────────────────

    def generate_quantum_features(
        self,
        X
    ):

        quantum_scores = batch_predict_proba(
            X,
            self.weights
        )

        quantum_scores = quantum_scores.reshape(
            -1,
            1
        )

        combined = np.hstack([
            X,
            quantum_scores
        ])

        return combined

    # ─────────────────────────────────────────────────────

    def train(
        self,
        X,
        y
    ):

        X_quantum = self.generate_quantum_features(
            X
        )

        X_train, X_test, y_train, y_test = train_test_split(

            X_quantum,
            y,

            test_size=0.2,

            random_state=42,

            stratify=y
        )

        self.classifier.fit(
            X_train,
            y_train
        )

        preds = self.classifier.predict(
            X_test
        )

        probs = self.classifier.predict_proba(
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

        print("\nHybrid QML Training Complete")

        print(f"F1 Score: {f1:.4f}")

        print(f"ROC-AUC : {auc:.4f}")

    # ─────────────────────────────────────────────────────

    def predict(
        self,
        X
    ):

        if not self.trained:

            raise RuntimeError(
                "Hybrid model not trained."
            )

        X_quantum = self.generate_quantum_features(
            X
        )

        return self.classifier.predict(
            X_quantum
        )

    # ─────────────────────────────────────────────────────

    def predict_proba(
        self,
        X
    ):

        X_quantum = self.generate_quantum_features(
            X
        )

        return self.classifier.predict_proba(
            X_quantum
        )