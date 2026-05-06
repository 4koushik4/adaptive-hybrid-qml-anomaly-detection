"""
System Stress Test

Tests:
- High-volume streaming
- Detector stability
- Throughput performance
"""

import time

from src.ingestion.stream_simulator import (
    MultiSourceSimulator
)

from src.preprocessing.preprocessor import (
    Preprocessor
)

from src.qml.detector import (
    AnomalyDetector
)


class SystemStressTest:

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
        n_points=1000
    ):

        simulator = MultiSourceSimulator(
            anomaly_rate=0.15
        )

        stream = simulator.stream(
            n_points=n_points,
            delay=0
        )

        start_time = time.time()

        anomalies = 0

        processed = 0

        for record in stream:

            x = self.preprocessor.transform_single(
                record
            )

            result = self.detector.predict_one(
                x
            )

            processed += 1

            anomalies += result[
                "is_anomaly"
            ]

        end_time = time.time()

        total_time = (
            end_time - start_time
        )

        throughput = (
            processed / total_time
        )

        print("\n========== STRESS TEST ==========")

        print(f"Processed Records : {processed}")

        print(f"Detected Anomalies: {anomalies}")

        print(f"Execution Time    : {total_time:.2f} sec")

        print(f"Throughput        : {throughput:.2f} records/sec")

        print("=================================\n")