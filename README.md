# 🏭 Factory Sensor Anomaly Detection System

AI-powered predictive maintenance system for manufacturing equipment using machine learning to detect potential failures before they occur.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This project implements a machine learning solution to predict equipment failures in manufacturing facilities by analyzing real-time sensor data. The system uses a Balanced Random Forest Classifier to handle imbalanced data and provides actionable insights through an interactive web interface.

## ✨ Features

- **Real-time Predictions**: Instant fault detection from sensor readings
- **Batch Processing**: Upload CSV files for multiple equipment analysis
- **Interactive Dashboard**: Beautiful visualizations with Plotly
- **High Accuracy**: Optimized for imbalanced industrial data
- **User-Friendly Interface**: No technical expertise required
- **Downloadable Reports**: Export predictions for record-keeping

## 📊 Model Performance

- **Algorithm**: Balanced Random Forest Classifier
- **Handling Imbalance**: 90-10 split (normal vs faulty)
- **Optimized Threshold**: 0.35 for better recall
- **Features**: Temperature, Vibration, Pressure, Humidity, Equipment Type

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/factory-sensor-anomaly-detection.git
cd factory-sensor-anomaly-detection
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open in browser**