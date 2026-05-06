# QML Real-Time Anomaly Detection System Architecture

## Overview

The project implements a Real-Time Anomaly Detection System using Quantum Machine Learning (QML). The system is designed to process streaming data from multiple domains such as cybersecurity, IoT, and industrial monitoring systems.

The architecture combines classical preprocessing techniques with quantum-enhanced anomaly detection to improve detection accuracy, robustness, and response latency.

---

# System Architecture

```text
                ┌─────────────────────────┐
                │     Data Sources        │
                │─────────────────────────│
                │ • NSL-KDD               │
                │ • UNSW-NB15             │
                │ • MQTT-IoT              │
                │ • Simulated Streams     │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │   Data Ingestion Layer  │
                │─────────────────────────│
                │ • Stream Simulator      │
                │ • Real Dataset Loader   │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │  Preprocessing Layer    │
                │─────────────────────────│
                │ • Data Cleaning         │
                │ • Missing Value Handling│
                │ • Normalization         │
                │ • PCA Feature Reduction │
                │ • Quantum Encoding      │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │    QML Detection Core   │
                │─────────────────────────│
                │ • Variational Circuit   │
                │ • Angle Embedding       │
                │ • Entangling Layers     │
                │ • Anomaly Probability   │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Evaluation & Monitoring │
                │─────────────────────────│
                │ • F1 Score              │
                │ • ROC-AUC               │
                │ • Latency Tracking      │
                │ • Drift Detection       │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │    Dashboard Layer      │
                │─────────────────────────│
                │ • Live Stream Charts    │
                │ • Anomaly Alerts        │
                │ • Performance Metrics   │
                └─────────────────────────┘