"""
Stream Simulator

Simulates real-time streaming data from:
- Network traffic
- IoT sensors
- Industrial systems
"""

import numpy as np
import pandas as pd
import time
import random
from datetime import datetime


class StreamSimulator:

    def __init__(
        self,
        source_type="network",
        anomaly_rate=0.05,
        seed=42
    ):

        self.source_type = source_type
        self.anomaly_rate = anomaly_rate

        self.rng = np.random.default_rng(seed)

        self.configs = {

            "network": {
                "features": [
                    "packet_size",
                    "connection_rate",
                    "port_number",
                    "protocol_type",
                    "byte_count"
                ],

                "normal_mean": [
                    500,
                    10,
                    443,
                    1,
                    1500
                ],

                "normal_std": [
                    100,
                    3,
                    50,
                    0.2,
                    300
                ],

                "anomaly_mean": [
                    2500,
                    100,
                    8080,
                    3,
                    10000
                ],

                "anomaly_std": [
                    500,
                    20,
                    500,
                    1,
                    3000
                ]
            },

            "iot": {

                "features": [
                    "temperature",
                    "humidity",
                    "vibration",
                    "battery_voltage",
                    "signal_strength"
                ],

                "normal_mean": [
                    25,
                    60,
                    0.02,
                    3.7,
                    -65
                ],

                "normal_std": [
                    2,
                    5,
                    0.01,
                    0.1,
                    5
                ],

                "anomaly_mean": [
                    80,
                    95,
                    0.8,
                    2.0,
                    -95
                ],

                "anomaly_std": [
                    10,
                    5,
                    0.2,
                    0.3,
                    10
                ]
            },

            "industrial": {

                "features": [
                    "pressure",
                    "rpm",
                    "voltage",
                    "temperature",
                    "vibration_freq"
                ],

                "normal_mean": [
                    100,
                    3000,
                    220,
                    60,
                    50
                ],

                "normal_std": [
                    5,
                    100,
                    5,
                    5,
                    5
                ],

                "anomaly_mean": [
                    180,
                    6000,
                    300,
                    120,
                    200
                ],

                "anomaly_std": [
                    20,
                    500,
                    20,
                    20,
                    50
                ]
            }
        }

    # ─────────────────────────────────────────────────────

    def generate_point(self):

        config = self.configs[self.source_type]

        is_anomaly = (
            self.rng.random()
            < self.anomaly_rate
        )

        if is_anomaly:

            values = self.rng.normal(
                config["anomaly_mean"],
                config["anomaly_std"]
            )

        else:

            values = self.rng.normal(
                config["normal_mean"],
                config["normal_std"]
            )

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": self.source_type,
            "label": int(is_anomaly)
        }

        for i, feature in enumerate(config["features"]):

            record[feature] = round(
                float(values[i]),
                4
            )

        return record

    # ─────────────────────────────────────────────────────

    def stream(
        self,
        n_points=None,
        delay=0.1
    ):

        count = 0

        while True:

            yield self.generate_point()

            count += 1

            if n_points and count >= n_points:
                break

            time.sleep(delay)

    # ─────────────────────────────────────────────────────

    def generate_batch(
        self,
        n=1000
    ):

        records = [
            self.generate_point()
            for _ in range(n)
        ]

        return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# Multi-Source Simulator
# ─────────────────────────────────────────────────────────────

class MultiSourceSimulator:

    def __init__(
        self,
        anomaly_rate=0.05,
        seed=42
    ):

        self.simulators = {

            "network": StreamSimulator(
                "network",
                anomaly_rate,
                seed
            ),

            "iot": StreamSimulator(
                "iot",
                anomaly_rate,
                seed + 1
            ),

            "industrial": StreamSimulator(
                "industrial",
                anomaly_rate,
                seed + 2
            )
        }

        self.sources = list(
            self.simulators.keys()
        )

    # ─────────────────────────────────────────────────────

    def stream(
        self,
        n_points=None,
        delay=0.1
    ):

        count = 0

        while True:

            source = random.choice(
                self.sources
            )

            yield self.simulators[
                source
            ].generate_point()

            count += 1

            if n_points and count >= n_points:
                break

            time.sleep(delay)

    # ─────────────────────────────────────────────────────

    def generate_training_data(
        self,
        n_per_source=500
    ):

        dfs = []

        for source in self.sources:

            df = self.simulators[
                source
            ].generate_batch(n_per_source)

            dfs.append(df)

        return pd.concat(
            dfs,
            ignore_index=True
        )