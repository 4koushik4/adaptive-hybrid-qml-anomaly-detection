"""
Adaptive Hybrid Quantum-Classical
API Routes
"""

from fastapi import APIRouter
import numpy as np

from src.evaluation.drift_detector import (
    DriftDetector
)

# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────

router = APIRouter()

# ─────────────────────────────────────────────────────────────
# GLOBAL STATS
# ─────────────────────────────────────────────────────────────

system_stats = {

    "total_requests": 0,

    "total_anomalies": 0,

    "avg_score": 0.0,

    "adaptive_threshold": 0.0,

    "drift_events": 0
}

score_window = []

drift_detector = DriftDetector()

# ─────────────────────────────────────────────────────────────
# UPDATE STATS
# ─────────────────────────────────────────────────────────────

def update_stats(
    score,
    is_anomaly,
    drift
):

    global score_window

    system_stats["total_requests"] += 1

    system_stats["total_anomalies"] += (
        int(is_anomaly)
    )

    if drift:

        system_stats["drift_events"] += 1

    score_window.append(score)

    recent_scores = score_window[-100:]

    system_stats["avg_score"] = round(

        float(np.mean(recent_scores)),

        4
    )

    adaptive_threshold = (

        np.mean(recent_scores)

        + 1.5 * np.std(recent_scores)
    )

    system_stats["adaptive_threshold"] = round(

        float(adaptive_threshold),

        4
    )

# ─────────────────────────────────────────────────────────────
# SYSTEM STATS
# ─────────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats():

    return {

        "system_statistics": system_stats
    }

# ─────────────────────────────────────────────────────────────
# RESET STATS
# ─────────────────────────────────────────────────────────────

@router.post("/reset")
def reset_stats():

    global score_window
    global drift_detector

    score_window = []

    drift_detector = DriftDetector()

    system_stats["total_requests"] = 0

    system_stats["total_anomalies"] = 0

    system_stats["avg_score"] = 0.0

    system_stats["adaptive_threshold"] = 0.0

    system_stats["drift_events"] = 0

    return {

        "message": (
            "Adaptive system statistics reset."
        )
    }

# ─────────────────────────────────────────────────────────────
# CURRENT THRESHOLD
# ─────────────────────────────────────────────────────────────

@router.get("/threshold")
def get_threshold():

    return {

        "adaptive_threshold": system_stats[
            "adaptive_threshold"
        ]
    }

# ─────────────────────────────────────────────────────────────
# SYSTEM STATUS
# ─────────────────────────────────────────────────────────────

@router.get("/status")
def system_status():

    return {

        "system": (
            "Adaptive Hybrid "
            "Quantum-Classical "
            "Anomaly Detection"
        ),

        "streaming": True,

        "adaptive_learning": True,

        "drift_detection": True,

        "models": [

            "HybridQMLModel",

            "QSVMModel",

            "QNN"
        ]
    }