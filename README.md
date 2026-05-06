# Adaptive Hybrid Quantum-Classical Drift-Aware Anomaly Detection

## Overview

Adaptive Hybrid Quantum-Classical Drift-Aware Anomaly Detection is a real-time intelligent anomaly detection framework designed for cybersecurity, IoT, and industrial monitoring environments. The system combines Quantum Machine Learning (QML) with classical machine learning techniques to detect anomalous behavior in streaming data while dynamically adapting to changing data distributions through adaptive thresholding and concept drift detection.

This project integrates:

* Hybrid Quantum-Classical Learning
* Quantum Support Vector Machines (QSVM)
* Quantum Neural Networks (QNN)
* Real-Time Streaming Analytics
* Adaptive Thresholding
* Concept Drift Detection
* Benchmarking against Classical ML Models
* Interactive Dashboard Visualization

The framework demonstrates the feasibility of adaptive quantum-classical anomaly detection in real-time streaming systems.

---

# Research Motivation

Modern streaming environments such as:

* Cybersecurity systems
* IoT infrastructures
* Smart industrial systems
* Sensor networks

generate large-scale continuous data streams where anomalies evolve over time. Traditional static machine learning systems struggle to adapt to dynamic environments and concept drift.

This project proposes an adaptive hybrid quantum-classical framework capable of:

* Real-time anomaly detection
* Dynamic threshold adaptation
* Streaming intelligence
* Drift-aware analytics

---

# Key Features

* Hybrid Quantum-Classical Anomaly Detection
* Real-Time Streaming Simulation
* Adaptive Dynamic Thresholding
* Concept Drift Detection
* Multi-Domain Stream Support
* Benchmarking Suite
* Latency Monitoring
* Streamlit Dashboard
* Model Persistence
* Research-Oriented Modular Architecture

---

# System Architecture

```text
Streaming Data Sources
        ↓
Data Ingestion Layer
        ↓
Preprocessing & Feature Extraction
        ↓
Hybrid QML / QSVM / QNN Models
        ↓
Adaptive Thresholding Engine
        ↓
Concept Drift Detection
        ↓
Real-Time Dashboard & Analytics
```

---

# Technologies Used

## Quantum Machine Learning

* PennyLane

## Classical Machine Learning

* Scikit-learn
* Random Forest
* Isolation Forest

## Deep Learning

* PyTorch

## Dashboard & Visualization

* Streamlit
* Plotly

## Backend & APIs

* FastAPI
* Uvicorn

## Data Processing

* NumPy
* Pandas

---

# Datasets Used

The project supports real-world cybersecurity and streaming datasets including:

* NSL-KDD
* UNSW-NB15
* MQTT-IoT
* Simulated Multi-Domain Streaming Data

Domains simulated:

* Network Traffic
* IoT Systems
* Industrial Monitoring

---

# Project Structure

```text
adaptive-hybrid-qml-anomaly-detection/
│
├── data/
├── models/
├── results/
├── logs/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── qml/
│   ├── evaluation/
│   ├── dashboard/
│   └── api/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/adaptive-hybrid-qml-anomaly-detection.git
```

```bash
cd adaptive-hybrid-qml-anomaly-detection
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Main Pipeline

```bash
python main.py
```

The pipeline performs:

* Dataset loading
* Preprocessing
* Hybrid QML training
* QSVM training
* QNN training
* Benchmark evaluation
* Streaming simulation
* Drift detection
* Result generation

---

# Run Dashboard

```bash
streamlit run src/dashboard/app.py
```

---

# Output Files

Generated outputs are saved in:

```text
results/
```

Including:

* benchmark_results.csv
* stream_logs.csv
* final_summary.csv

Saved models are stored in:

```text
models/
```

---

# Benchmark Models

The framework compares:

* Hybrid QML
* QSVM
* QNN
* Random Forest
* Isolation Forest

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Latency

---

# Research Contributions

* Adaptive Hybrid Quantum-Classical Framework
* Real-Time Streaming Anomaly Detection
* Drift-Aware Streaming Intelligence
* Dynamic Threshold Adaptation
* Multi-Model Benchmarking
* Latency-Aware Real-Time Analytics

---

# Dashboard Features

The Streamlit dashboard provides:

* Live anomaly monitoring
* Adaptive threshold visualization
* Real-time analytics
* Streaming statistics
* Drift detection indicators

---

# Future Enhancements

* Deployment on real quantum hardware
* Federated quantum anomaly detection
* Reinforcement learning-based adaptation
* Edge-device deployment
* Advanced QNN optimization
* Real-time Kafka integration

---

# Research & Publication

This project is designed as a research-oriented prototype suitable for:

* IEEE student conferences
* Research internships
* Quantum AI workshops
* Undergraduate research publications
* Cybersecurity and IoT research demonstrations

---

# Disclaimer

This project is a research prototype developed for experimental and academic purposes. It demonstrates the feasibility of adaptive hybrid quantum-classical anomaly detection in streaming environments and is not intended as a production-grade cybersecurity solution.

---

# Author

Koushik Juluri

Computer Science Engineering
Quantum Machine Learning | Cybersecurity | AI Systems | Real-Time Analytics

---

# License

MIT License
