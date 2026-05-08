"""
Automatic Real Dataset Loader

Features:
- Auto-download datasets
- Local caching
- Unified schema generation
- Automatic preprocessing support

Datasets:
- NSL-KDD
- UNSW-NB15
- MQTT-IoT
"""

import os
import urllib.request
import pandas as pd
import numpy as np


class RealDataLoader:

    def __init__(
        self,
        data_dir="data/raw"
    ):

        self.data_dir = data_dir

        os.makedirs(
            self.data_dir,
            exist_ok=True
        )

    # ─────────────────────────────────────────────────────
    # Download Utility
    # ─────────────────────────────────────────────────────

    def download_file(
        self,
        url,
        save_path
    ):

        if os.path.exists(save_path):

            print(
                f"[CACHE] {save_path}"
            )

            return

        print(
            f"[DOWNLOAD] {url}"
        )

        try:

            urllib.request.urlretrieve(
                url,
                save_path
            )

            print(
                f"[SAVED] {save_path}"
            )

        except Exception as e:

            raise RuntimeError(
                f"Failed download: {e}"
            )

    # ─────────────────────────────────────────────────────
    # Standardization
    # ─────────────────────────────────────────────────────

    def standardize(
        self,
        df,
        source_name,
        feature_columns,
        label_column
    ):

        df = df.copy()

        # Label handling
        if df[label_column].dtype == object:

            df["label"] = (

                df[label_column]
                .astype(str)
                .str.lower()
                .apply(

                    lambda x:
                    0 if x in [
                        "normal",
                        "benign",
                        "0"
                    ] else 1
                )
            )

        else:

            df["label"] = (
                pd.to_numeric(
                    df[label_column],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )

        # Numeric features
        features = df[
            feature_columns
        ].copy()

        for col in feature_columns:

            features[col] = pd.to_numeric(

                features[col],

                errors="coerce"

            ).fillna(0)

        # Rename features
        features.columns = [

            f"feature_{i}"

            for i in range(
                len(feature_columns)
            )
        ]

        features["label"] = df["label"]

        features["source"] = source_name

        return features

    # ─────────────────────────────────────────────────────
    # NSL-KDD
    # ─────────────────────────────────────────────────────

    def load_nsl_kdd(self):

        folder = os.path.join(
            self.data_dir,
            "nsl_kdd"
        )

        os.makedirs(
            folder,
            exist_ok=True
        )

        path = os.path.join(
            folder,
            "nsl_kdd.csv"
        )

        url = (
            "https://raw.githubusercontent.com/"
            "jmnwong/NSL-KDD-Dataset/master/"
            "KDDTrain+.txt"
        )

        self.download_file(
            url,
            path
        )

        columns = [

            "duration","protocol_type","service","flag",

            "src_bytes","dst_bytes","land",

            "wrong_fragment","urgent","hot",

            "num_failed_logins","logged_in",

            "num_compromised","root_shell",

            "su_attempted","num_root",

            "num_file_creations","num_shells",

            "num_access_files","num_outbound_cmds",

            "is_host_login","is_guest_login",

            "count","srv_count","serror_rate",

            "srv_serror_rate","rerror_rate",

            "srv_rerror_rate","same_srv_rate",

            "diff_srv_rate","srv_diff_host_rate",

            "dst_host_count","dst_host_srv_count",

            "dst_host_same_srv_rate",

            "dst_host_diff_srv_rate",

            "dst_host_same_src_port_rate",

            "dst_host_srv_diff_host_rate",

            "dst_host_serror_rate",

            "dst_host_srv_serror_rate",

            "dst_host_rerror_rate",

            "dst_host_srv_rerror_rate",

            "label","difficulty"
        ]

        df = pd.read_csv(
            path,
            names=columns
        )

        return self.standardize(

            df,

            "nsl_kdd",

            [
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count"
],
            "label"
        )

   
    # ─────────────────────────────────────────────────────
    # Load All
    # ─────────────────────────────────────────────────────

    def load_all(self):

        datasets = []

        loaders = [

            self.load_nsl_kdd
            
        ]

        for loader in loaders:

            try:

                df = loader()

                datasets.append(df)

                print(
                    f"Loaded: {df['source'].iloc[0]}"
                )

            except Exception as e:

                print(
                    f"Skipping dataset: {e}"
                )

        if len(datasets) == 0:

            raise RuntimeError(
                "No datasets loaded."
            )

        combined = pd.concat(
            datasets,
            ignore_index=True
        )

        combined = combined.sample(
            frac=1,
            random_state=42
        )

        return combined.reset_index(
            drop=True
        )


# ─────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    loader = RealDataLoader()

    df = loader.load_all()

    print(df.head())

    print(df.shape)