"""
Quantum Neural Network (QNN)

Implements a hybrid QNN using
PennyLane + PyTorch.
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import (
    f1_score
)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

N_QUBITS = 4

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)

# ─────────────────────────────────────────────────────────────
# Quantum Layer
# ─────────────────────────────────────────────────────────────

@qml.qnode(dev, interface="torch")
def quantum_layer(
    inputs,
    weights
):

    qml.AngleEmbedding(
        inputs,
        wires=range(N_QUBITS)
    )

    qml.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS)
    )

    return qml.expval(
        qml.PauliZ(0)
    )

# ─────────────────────────────────────────────────────────────
# QNN Model
# ─────────────────────────────────────────────────────────────

class QNN(nn.Module):

    def __init__(self):

        super().__init__()

        weight_shapes = {
            "weights": (2, N_QUBITS, 3)
        }

        self.q_layer = qml.qnn.TorchLayer(
            quantum_layer,
            weight_shapes
        )

        self.fc = nn.Linear(
            1,
            1
        )

        self.sigmoid = nn.Sigmoid()

    # ─────────────────────────────────────────────────────

    def forward(
        self,
        x
    ):

        q_out = self.q_layer(x)

        q_out = q_out.unsqueeze(1)

        out = self.fc(q_out)

        out = self.sigmoid(out)

        return out

# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class QNNTrainer:

    def __init__(
        self,
        lr=0.01,
        epochs=20
    ):

        self.model = QNN()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        self.loss_fn = nn.BCELoss()

        self.epochs = epochs

    # ─────────────────────────────────────────────────────

    def train(
        self,
        X,
        y
    ):

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )

        y_tensor = torch.tensor(
            y,
            dtype=torch.float32
        ).unsqueeze(1)

        for epoch in range(self.epochs):

            self.optimizer.zero_grad()

            outputs = self.model(
                X_tensor
            )

            loss = self.loss_fn(
                outputs,
                y_tensor
            )

            loss.backward()

            self.optimizer.step()

            preds = (
                outputs.detach().numpy()
                > 0.5
            ).astype(int)

            f1 = f1_score(
                y,
                preds
            )

            print(
                f"Epoch {epoch+1}/{self.epochs}"
            )

            print(
                f"Loss: {loss.item():.4f}"
            )

            print(
                f"F1 Score: {f1:.4f}\n"
            )

    # ─────────────────────────────────────────────────────

    def predict(
        self,
        X
    ):

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )

        outputs = self.model(
            X_tensor
        )

        return outputs.detach().numpy()