"""
Quantum Circuit Module

Implements:
- Variational Quantum Circuit (VQC)
- Angle Embedding
- Strongly Entangling Layers
- Quantum Probability Prediction
"""

import pennylane as qml
import numpy as np

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

N_QUBITS = 4
N_LAYERS = 2

# ─────────────────────────────────────────────────────────────
# Quantum Device
# ─────────────────────────────────────────────────────────────

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)

# ─────────────────────────────────────────────────────────────
# Quantum Circuit
# ─────────────────────────────────────────────────────────────

@qml.qnode(dev, interface="numpy")
def quantum_circuit(
    inputs,
    weights
):

    # Angle Embedding
    qml.AngleEmbedding(
        inputs,
        wires=range(N_QUBITS),
        rotation="Y"
    )

    # Variational Layers
    qml.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS)
    )

    # Measurement
    return qml.expval(
        qml.PauliZ(0)
    )

# ─────────────────────────────────────────────────────────────
# Convert Output → Probability
# ─────────────────────────────────────────────────────────────

def circuit_to_probability(
    inputs,
    weights
):

    raw_output = quantum_circuit(
        inputs,
        weights
    )

    probability = (
        1 - raw_output
    ) / 2

    return float(probability)

# ─────────────────────────────────────────────────────────────
# Weight Initialization
# ─────────────────────────────────────────────────────────────

def initialize_weights(
    seed=42
):

    rng = np.random.default_rng(seed)

    weights = rng.uniform(

        low=0,

        high=2 * np.pi,

        size=(
            N_LAYERS,
            N_QUBITS,
            3
        )
    )

    return weights

# ─────────────────────────────────────────────────────────────
# Batch Prediction
# ─────────────────────────────────────────────────────────────

def batch_predict_proba(
    X,
    weights
):

    probs = []

    for x in X:

        p = circuit_to_probability(
            x,
            weights
        )

        probs.append(p)

    return np.array(probs)

# ─────────────────────────────────────────────────────────────
# Gradient Calculation
# ─────────────────────────────────────────────────────────────

def compute_gradient(
    inputs,
    weights
):

    grad_fn = qml.grad(
        quantum_circuit,
        argnum=1
    )

    return grad_fn(
        inputs,
        weights
    )

# ─────────────────────────────────────────────────────────────
# Circuit Drawer
# ─────────────────────────────────────────────────────────────

def draw_circuit():

    inputs = np.zeros(N_QUBITS)

    weights = initialize_weights()

    print(
        qml.draw(quantum_circuit)(
            inputs,
            weights
        )
    )

# ─────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Testing Quantum Circuit")

    weights = initialize_weights()

    x = np.array([
        0.2,
        1.1,
        2.0,
        0.7
    ])

    prob = circuit_to_probability(
        x,
        weights
    )

    print(f"Probability: {prob}")

    draw_circuit()