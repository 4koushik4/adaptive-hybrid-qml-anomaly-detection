"""
Quantum Feature Encoders

Supports:
- Angle Encoding
- Amplitude Encoding
- Basis Encoding
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# Angle Encoding
# ─────────────────────────────────────────────────────────────

def angle_encoding(x):

    x = np.array(x)

    x = np.clip(
        x,
        0,
        np.pi
    )

    return x


# ─────────────────────────────────────────────────────────────
# Amplitude Encoding
# ─────────────────────────────────────────────────────────────

def amplitude_encoding(x):

    x = np.array(x)

    norm = np.linalg.norm(x)

    if norm == 0:
        return x

    return x / norm


# ─────────────────────────────────────────────────────────────
# Basis Encoding
# ─────────────────────────────────────────────────────────────

def basis_encoding(x):

    x = np.array(x)

    return np.where(
        x > np.mean(x),
        1,
        0
    )