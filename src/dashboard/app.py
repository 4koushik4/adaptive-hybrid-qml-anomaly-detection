"""
Adaptive Hybrid Quantum-Classical
Drift-Aware Dashboard

Run:
streamlit run src/dashboard/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import time

from src.ingestion.stream_simulator import (
    MultiSourceSimulator
)

from src.ingestion.real_data_loader import (
    RealDataLoader
)

from src.preprocessing.preprocessor import (
    Preprocessor
)

from src.qml.hybrid_model import (
    HybridQMLModel
)





from src.evaluation.drift_detector import (
    DriftDetector
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(

    page_title="Adaptive Hybrid QML",

    page_icon="⚛️",

    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────

if "trained" not in st.session_state:

    st.session_state.trained = False

if "running" not in st.session_state:

    st.session_state.running = False

if "scores" not in st.session_state:

    st.session_state.scores = []

if "thresholds" not in st.session_state:

    st.session_state.thresholds = []

if "timestamps" not in st.session_state:

    st.session_state.timestamps = []

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

st.sidebar.title(
    "Adaptive QML Controls"
)

sample_size = st.sidebar.slider(

    "Training Sample Size",

    500,

    5000,

    2000
)

epochs = st.sidebar.slider(

    "QNN Epochs",

    1,

    10,

    3
)

anomaly_rate = st.sidebar.slider(

    "Streaming Anomaly Rate",

    0.01,

    0.5,

    0.15
)

# ─────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────

st.title(
    "⚛️ Adaptive Hybrid Quantum-Classical Anomaly Detection"
)

st.markdown("""

Real-Time Drift-Aware Streaming Intelligence System

Features:
- Hybrid Quantum-Classical Learning
- Adaptive Thresholding
- Concept Drift Detection
- Real-Time Streaming Analytics
- Multi-Domain Monitoring

""")

# ─────────────────────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────────────────────

if st.sidebar.button(
    "Train Models"
):

    with st.spinner(
        "Training Hybrid QML Models..."
    ):

        # Load datasets
        loader = RealDataLoader()

        df = loader.load_all()

        df = df.sample(

            n=min(sample_size, len(df)),

            random_state=42
        )

        # Preprocessing
        preprocessor = Preprocessor()

        X, y, meta = preprocessor.fit_transform(
            df
        )

        # Hybrid QML
        hybrid_model = HybridQMLModel()

        hybrid_model.train(X, y)

        

        # Save to session
        st.session_state.preprocessor = (
            preprocessor
        )

        st.session_state.hybrid_model = (
            hybrid_model
        )

        
        st.session_state.trained = True

    st.success(
        "Adaptive Hybrid QML System Trained!"
    )

# ─────────────────────────────────────────────────────────────
# STREAMING CONTROLS
# ─────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "Start Stream",
        disabled=not st.session_state.trained
    ):

        st.session_state.running = True

with col2:

    if st.button(
        "Stop Stream"
    ):

        st.session_state.running = False

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)

processed = len(
    st.session_state.scores
)

anomalies = sum([

    1

    for s, t in zip(

        st.session_state.scores,

        st.session_state.thresholds
    )

    if s > t
])

avg_score = (

    np.mean(st.session_state.scores)

    if processed > 0

    else 0
)

avg_threshold = (

    np.mean(st.session_state.thresholds)

    if processed > 0

    else 0
)

m1.metric(
    "Processed",
    processed
)

m2.metric(
    "Anomalies",
    anomalies
)

m3.metric(
    "Avg Score",
    round(avg_score, 4)
)

m4.metric(
    "Adaptive Threshold",
    round(avg_threshold, 4)
)

# ─────────────────────────────────────────────────────────────
# STREAMING ENGINE
# ─────────────────────────────────────────────────────────────

chart_placeholder = st.empty()

log_placeholder = st.empty()

if (

    st.session_state.running

    and

    st.session_state.trained
):

    simulator = MultiSourceSimulator(
        anomaly_rate=anomaly_rate
    )

    drift_detector = DriftDetector()

    stream = simulator.stream(
        n_points=200,
        delay=0.05
    )

    for record in stream:

        if not st.session_state.running:
            break

        # Convert schema
        transformed = {

            "source": "stream",

            "label": record["label"]
        }

        values = [

            v for k, v in record.items()

            if isinstance(v, (int, float))
        ]

        for i in range(5):

            transformed[f"feature_{i}"] = (

                values[i]

                if i < len(values)

                else 0
            )

        # Preprocess
        x = st.session_state.preprocessor.transform_single(
            transformed
        )

        # Hybrid prediction
        score = st.session_state.hybrid_model.predict_proba(
            [x]
        )[0][1]

        # Adaptive threshold
        recent_scores = (

            st.session_state.scores[-50:]

            if len(st.session_state.scores) > 10

            else [score]
        )

        threshold = (

            np.mean(recent_scores)

            + 1.5 * np.std(recent_scores)
        )

        is_anomaly = int(
            score >= threshold
        )

        # Drift detection
        drift = drift_detector.update(
            score
        )

        # Store
        st.session_state.scores.append(
            score
        )

        st.session_state.thresholds.append(
            threshold
        )

        st.session_state.timestamps.append(
            len(st.session_state.timestamps)
        )

        # Keep window
        MAX_POINTS = 100

        if len(st.session_state.scores) > MAX_POINTS:

            st.session_state.scores.pop(0)

            st.session_state.thresholds.pop(0)

            st.session_state.timestamps.pop(0)

        # ─────────────────────────────────────────────────
        # LIVE PLOT
        # ─────────────────────────────────────────────────

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=st.session_state.timestamps,

                y=st.session_state.scores,

                mode="lines+markers",

                name="Anomaly Score"
            )
        )

        fig.add_trace(

            go.Scatter(

                x=st.session_state.timestamps,

                y=st.session_state.thresholds,

                mode="lines",

                name="Adaptive Threshold"
            )
        )

        fig.update_layout(

            title="Real-Time Adaptive Anomaly Detection",

            xaxis_title="Time",

            yaxis_title="Score",

            height=500
        )

        chart_placeholder.plotly_chart(
            fig,
            use_container_width=True
        )

        # ─────────────────────────────────────────────────
        # ALERTS
        # ─────────────────────────────────────────────────

        if drift:

            log_placeholder.warning(
                "Concept Drift Detected!"
            )

        elif is_anomaly:

            log_placeholder.error(

                f"""
                ANOMALY DETECTED

                Source: {record['source']}

                Score: {score:.4f}

                Threshold: {threshold:.4f}
                """
            )

        else:

            log_placeholder.success(

                f"""
                NORMAL

                Source: {record['source']}

                Score: {score:.4f}
                """
            )

        time.sleep(0.05)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────

st.markdown("---")

st.markdown("""

Adaptive Hybrid Quantum-Classical Drift-Aware
Anomaly Detection System

Research-Oriented Real-Time Streaming Intelligence

""")