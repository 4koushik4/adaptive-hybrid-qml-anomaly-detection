"""
Benchmark Suite

Compares:
- Isolation Forest
- Random Forest
- Hybrid QML
- QSVM
- QNN

Research-Oriented Benchmarking
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier
)

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score
)

from sklearn.model_selection import (
    train_test_split
)

from src.qml.hybrid_model import (
    HybridQMLModel
)

from src.qml.qsvm import (
    QSVMModel
)

from src.qml.qnn import (
    QNNTrainer
)

# ─────────────────────────────────────────────────────────────
# BENCHMARK SUITE
# ─────────────────────────────────────────────────────────────

class BenchmarkSuite:

    def __init__(self):

        self.results = []

    # ─────────────────────────────────────────────────────
    # STORE RESULTS
    # ─────────────────────────────────────────────────────

    def add_result(
        self,
        model_name,
        y_true,
        y_pred,
        y_prob=None
    ):

        result = {

            "model": model_name,

            "accuracy": round(

                accuracy_score(
                    y_true,
                    y_pred
                ),
                4
            ),

            "precision": round(

                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),
                4
            ),

            "recall": round(

                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),
                4
            ),

            "f1_score": round(

                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),
                4
            )
        }

        if y_prob is not None:

            try:

                result["roc_auc"] = round(

                    roc_auc_score(
                        y_true,
                        y_prob
                    ),
                    4
                )

            except:

                result["roc_auc"] = 0.0

        self.results.append(result)

    # ─────────────────────────────────────────────────────
    # CLASSICAL BASELINES
    # ─────────────────────────────────────────────────────

    def evaluate_isolation_forest(
        self,
        X_train,
        X_test,
        y_test
    ):

        model = IsolationForest(

            contamination=0.15,

            random_state=42
        )

        model.fit(X_train)

        preds = model.predict(X_test)

        preds = np.where(
            preds == -1,
            1,
            0
        )

        self.add_result(

            "Isolation Forest",

            y_test,

            preds
        )

    # ─────────────────────────────────────────────────────

    def evaluate_random_forest(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        model = RandomForestClassifier(

            n_estimators=100,

            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        preds = model.predict(
            X_test
        )

        probs = model.predict_proba(
            X_test
        )[:, 1]

        self.add_result(

            "Random Forest",

            y_test,

            preds,

            probs
        )

    # ─────────────────────────────────────────────────────
    # HYBRID QML
    # ─────────────────────────────────────────────────────

    def evaluate_hybrid_qml(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        model = HybridQMLModel()

        model.train(
            X_train,
            y_train
        )

        preds = model.predict(
            X_test
        )

        probs = model.predict_proba(
            X_test
        )[:, 1]

        self.add_result(

            "Hybrid QML",

            y_test,

            preds,

            probs
        )

    # ─────────────────────────────────────────────────────
    # QSVM
    # ─────────────────────────────────────────────────────

    def evaluate_qsvm(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        model = QSVMModel()

        model.train(
            X_train,
            y_train
        )

        preds = model.predict(
            X_test
        )

        probs = model.predict_proba(
            X_test
        )[:, 1]

        self.add_result(

            "QSVM",

            y_test,

            preds,

            probs
        )

    # ─────────────────────────────────────────────────────
    # QNN
    # ─────────────────────────────────────────────────────

    def evaluate_qnn(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        model = QNNTrainer(
            lr=0.01,
            epochs=3
        )

        # Smaller subset for speed
        model.train(
            X_train[:1000],
            y_train[:1000]
        )

        probs = model.predict(
            X_test
        ).flatten()

        preds = (
            probs >= 0.5
        ).astype(int)

        self.add_result(

            "QNN",

            y_test,

            preds,

            probs
        )

    # ─────────────────────────────────────────────────────
    # FULL BENCHMARK
    # ─────────────────────────────────────────────────────

    def run_all(
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

        print(
            "\nRunning Isolation Forest..."
        )

        self.evaluate_isolation_forest(
            X_train,
            X_test,
            y_test
        )

        print(
            "\nRunning Random Forest..."
        )

        self.evaluate_random_forest(
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(
            "\nRunning Hybrid QML..."
        )

        self.evaluate_hybrid_qml(
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(
            "\nRunning QSVM..."
        )

        self.evaluate_qsvm(
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(
            "\nRunning QNN..."
        )

        self.evaluate_qnn(
            X_train,
            X_test,
            y_train,
            y_test
        )

    # ─────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────

    def get_results(self):

        return pd.DataFrame(
            self.results
        )

    # ─────────────────────────────────────────────────────

    def print_results(self):

        df = self.get_results()

        print(
            "\n========== BENCHMARK RESULTS ==========\n"
        )

        print(df)

        print(
            "\n=======================================\n"
        )