"""
Adaptive Hybrid Quantum-Classical
Inference API

Run:
uvicorn src.api.inference_api:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd

from src.ingestion.real_data_loader import (
    RealDataLoader
)

from src.preprocessing.preprocessor import (
    Preprocessor
)

from src.qml.hybrid_model import (
    HybridQMLModel
)

from src.qml.qsvm import (
    QSVMModel
)

from src.evaluation.drift_detector import (
    DriftDetector
)

# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(

    title="Adaptive Hybrid QML API",

    version="2.0.0",

    description="""
    Drift-Aware Hybrid Quantum-Classical
    Real-Time Anomaly Detection API
    """
)

# ─────────────────────────────────────────────────────────────
# GLOBAL OBJECTS
# ─────────────────────────────────────────────────────────────

preprocessor = None

hybrid_model = None

qsvm_model = None

drift_detector = DriftDetector()

score_window = []

# ─────────────────────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────────────────────

class StreamRecord(BaseModel):

    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float

# ─────────────────────────────────────────────────────────────
# STARTUP TRAINING
# ─────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():

    global preprocessor
    global hybrid_model
    global qsvm_model

    print("\nLoading datasets...\n")

    loader = RealDataLoader()

    df = loader.load_all()

    df = df.sample(

        n=min(2000, len(df)),

        random_state=42
    )

    print("\nPreprocessing...\n")

    preprocessor = Preprocessor()

    X, y, meta = preprocessor.fit_transform(
        df
    )

    print("\nTraining Hybrid Model...\n")

    hybrid_model = HybridQMLModel()

    hybrid_model.train(X, y)

    print("\nTraining QSVM...\n")

    qsvm_model = QSVMModel()

    qsvm_model.train(X, y)

    print("\nAPI Ready.\n")

# ─────────────────────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():

    return {

        "project": (
            "Adaptive Hybrid "
            "Quantum-Classical "
            "Anomaly Detection"
        ),

        "status": "running"
    }

# ─────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():

    return {

        "status": "healthy"
    }

# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────

@app.post("/predict")
def predict(record: StreamRecord):

    global score_window

    # Create dataframe
    data = {

        "source": "api",

        "label": 0,

        "feature_0": record.feature_0,

        "feature_1": record.feature_1,

        "feature_2": record.feature_2,

        "feature_3": record.feature_3,

        "feature_4": record.feature_4
    }

    # Preprocess
    x = preprocessor.transform_single(
        data
    )

    # Hybrid prediction
    hybrid_prob = hybrid_model.predict_proba(
        [x]
    )[0][1]

    # QSVM prediction
    qsvm_prob = qsvm_model.predict_proba(
        [x]
    )[0][1]

    # Ensemble score
    final_score = (
        hybrid_prob
        + qsvm_prob
    ) / 2

    # ─────────────────────────────────────────────────────
    # ADAPTIVE THRESHOLD
    # ─────────────────────────────────────────────────────

    score_window.append(final_score)

    recent_scores = score_window[-50:]

    adaptive_threshold = (

        np.mean(recent_scores)

        + 1.5 * np.std(recent_scores)
    )

    is_anomaly = int(
        final_score >= adaptive_threshold
    )

    # ─────────────────────────────────────────────────────
    # DRIFT DETECTION
    # ─────────────────────────────────────────────────────

    drift_detected = drift_detector.update(
        final_score
    )

    # ─────────────────────────────────────────────────────
    # RESPONSE
    # ─────────────────────────────────────────────────────

    return {

        "hybrid_score": round(
            float(hybrid_prob),
            4
        ),

        "qsvm_score": round(
            float(qsvm_prob),
            4
        ),

        "final_score": round(
            float(final_score),
            4
        ),

        "adaptive_threshold": round(
            float(adaptive_threshold),
            4
        ),

        "is_anomaly": is_anomaly,

        "drift_detected": drift_detected
    }

# ─────────────────────────────────────────────────────────────
# MODEL INFO
# ─────────────────────────────────────────────────────────────

@app.get("/model-info")
def model_info():

    return {

        "models": [

            "HybridQMLModel",

            "QSVMModel",

            "AdaptiveThresholding",

            "ConceptDriftDetection"
        ],

        "features": 5,

        "streaming": True,

        "adaptive": True
    }