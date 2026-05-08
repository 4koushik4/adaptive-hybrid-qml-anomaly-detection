"""
Adaptive Hybrid Quantum-Classical
Drift-Aware Anomaly Detection System

FINAL RESEARCH PIPELINE
"""

import time
import numpy as np
import pandas as pd

from src.ingestion.real_data_loader import (
    RealDataLoader
)

from src.ingestion.stream_simulator import (
    MultiSourceSimulator
)

from src.preprocessing.preprocessor import (
    Preprocessor
)

from src.qml.hybrid_model import (
    HybridQMLModel
)





from src.evaluation.metrics import (
    PerformanceMetrics
)

from src.evaluation.drift_detector import (
    DriftDetector
)

from src.evaluation.benchmark import (
    BenchmarkSuite
)

# ─────────────────────────────────────────────────────────────
# LOAD DATASETS
# ─────────────────────────────────────────────────────────────

print("\nLoading datasets...\n")

loader = RealDataLoader()

df = loader.load_all()


# Use full combined datasets

df = df.reset_index(drop=True)

print(df.head())

print(f"\nDataset Shape: {df.shape}")

# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────

print("\nPreprocessing datasets...\n")

preprocessor = Preprocessor()

X, y, meta = preprocessor.fit_transform(df)

print(f"Processed Shape: {X.shape}")

# ─────────────────────────────────────────────────────────────
# TRAIN HYBRID MODEL
# ─────────────────────────────────────────────────────────────

print("\nTraining Hybrid QML Model...\n")

hybrid_model = HybridQMLModel()

hybrid_model.train(X, y)






# ─────────────────────────────────────────────────────────────
# SAVE TRAINED MODELS
# ─────────────────────────────────────────────────────────────

import os
import joblib

print("\nSaving trained models...\n")

os.makedirs(
    "models",
    exist_ok=True
)

# Save Hybrid QML
joblib.dump(
    hybrid_model,
    "models/hybrid_qml.pkl"
)

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)



print("Model saved successfully.")
# ─────────────────────────────────────────────────────────────
# BENCHMARK SUITE
# ─────────────────────────────────────────────────────────────

print("\nRunning Benchmark Suite...\n")

benchmark = BenchmarkSuite()

benchmark.run_all(X, y)

benchmark.print_results()

results_df = benchmark.get_results()

results_df.to_csv(
    "benchmark_results.csv",
    index=False
)

print(
    "\nBenchmark results saved "
    "to benchmark_results.csv"
)

# ─────────────────────────────────────────────────────────────
# REAL-TIME STREAMING
# ─────────────────────────────────────────────────────────────

print("\nStarting Real-Time Stream...\n")

simulator = MultiSourceSimulator(
    anomaly_rate=0.15
)

stream = simulator.stream(
    n_points=100,
    delay=0.05
)

metrics = PerformanceMetrics()

drift_detector = DriftDetector()

score_window = []

stream_counts = {

    "network": 0,

    "iot": 0,

    "industrial": 0
}

# ─────────────────────────────────────────────────────────────
# STREAM LOOP
# ─────────────────────────────────────────────────────────────

for record in stream:

    stream_counts[
        record["source"]
    ] += 1

    # Convert stream schema
    transformed = {

        "source": "stream",

        "label": record["label"]
    }

    values = [

        v for k, v in record.items()

        if isinstance(v, (int, float))
    ]

    for i in range(10):

        transformed[f"feature_{i}"] = (

            values[i]

            if i < len(values)

            else 0
        )

    # Preprocess
    x = preprocessor.transform_single(
        transformed
    )

    # ─────────────────────────────────────────────────────
    # LATENCY START
    # ─────────────────────────────────────────────────────

    start_time = time.time()

    # ─────────────────────────────────────────────────────
    # HYBRID PREDICTION
    # ─────────────────────────────────────────────────────

    hybrid_prob = hybrid_model.predict_proba(
        [x]
    )[0][1]

    # ─────────────────────────────────────────────────────
    # LATENCY END
    # ─────────────────────────────────────────────────────

    latency = (
        time.time() - start_time
    ) * 1000

    # ─────────────────────────────────────────────────────
    # ADAPTIVE THRESHOLDING
    # ─────────────────────────────────────────────────────

    score_window.append(hybrid_prob)

    recent_scores = score_window[-50:]

    dynamic_threshold = np.percentile(
        recent_scores,
        85
    )

    # Better anomaly logic
    is_anomaly = int(

        hybrid_prob >= dynamic_threshold

        and

        hybrid_prob > 0.7
    )

    # ─────────────────────────────────────────────────────
    # DRIFT DETECTION
    # ─────────────────────────────────────────────────────

    drift_detected = drift_detector.update(
        hybrid_prob
    )

    # ─────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────

    metrics.update(

        label=record["label"],

        prediction=is_anomaly,

        score=hybrid_prob,

        latency=latency
    )

    # ─────────────────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────────────────

    print(
        f"\n[{record['source']}]"
    )

    print(
        f"Score: {hybrid_prob:.4f}"
    )

    print(
        f"Dynamic Threshold: "
        f"{dynamic_threshold:.4f}"
    )

    print(
        f"Anomaly: {is_anomaly}"
    )

    print(
        f"Latency: {latency:.2f} ms"
    )

    if drift_detected:

        print(
            "Concept Drift Detected!"
        )

# ─────────────────────────────────────────────────────────────
# STREAM STATS
# ─────────────────────────────────────────────────────────────

print("\nSTREAM DISTRIBUTION\n")

print(stream_counts)

# ─────────────────────────────────────────────────────────────
# FINAL METRICS
# ─────────────────────────────────────────────────────────────

print("\nFINAL PERFORMANCE\n")

metrics.print_report()

print(
    "\nAdaptive Hybrid QML "
    "Pipeline Complete.\n"
)