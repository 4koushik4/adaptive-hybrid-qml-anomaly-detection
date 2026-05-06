"""
Latency Benchmark Test

Measures:
- Prediction latency
- Average latency
- Peak latency
"""

import time
import numpy as np

from src.ingestion.stream_simulator import (
    MultiSourceSimulator
)

from src.preprocessing.preprocessor import (
    Preprocessor
)

from src.qml.detector import (
    AnomalyDetector
)


class LatencyBenchmark:

    def __init__(
        self,
        detector,
        preprocessor
    ):

        self.detector = detector

        self.preprocessor = preprocessor

    # ─────────────────────────────────────────────────────

    def run(
        self,
        n_points=500
    ):

        simulator = MultiSourceSimulator(
            anomaly_rate=0.15
        )

        stream = simulator.stream(
            n_points=n_points,
            delay=0
        )

        latencies = []

        for record in stream:

            x = self.preprocessor.transform_single(
                record
            )

            start = time.perf_counter()

            self.detector.predict_one(
                x
            )

            end = time.perf_counter()

            latency_ms = (
                end - start
            ) * 1000

            latencies.append(
                latency_ms
            )

        print("\n========== LATENCY TEST ==========")

        print(
            f"Average Latency : {np.mean(latencies):.2f} ms"
        )

        print(
            f"Minimum Latency : {np.min(latencies):.2f} ms"
        )

        print(
            f"Maximum Latency : {np.max(latencies):.2f} ms"
        )

        print(
            f"P95 Latency     : {np.percentile(latencies, 95):.2f} ms"
        )

        print("==================================\n")