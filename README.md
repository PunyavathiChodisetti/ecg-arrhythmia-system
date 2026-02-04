# ECG Arrhythmia Detection System 🫀

An end-to-end Machine Learning application for detecting cardiac arrhythmias from ECG signals using a **1D Convolutional Neural Network (CNN)** trained on the **PTB-XL dataset**.

This project includes:
- Data preprocessing & model training
- CNN-based ECG classification
- FastAPI backend for inference
- React + TypeScript frontend with ECG waveform visualization

---

## 🚀 Features

- Upload **ECG .dat and .hea files**
- Automatic ECG signal preprocessing
- CNN-based arrhythmia classification
- Confidence-calibrated predictions
- ECG waveform visualization in UI
- REST API built with FastAPI
- Modular and production-ready code structure

---

## 🧠 Machine Learning Details

- **Model**: 1D Convolutional Neural Network (CNN)
- **Dataset**: PTB-XL (21,000+ ECG recordings)
- **Classes**:
  - NORM (Normal)
  - MI (Myocardial Infarction)
  - STTC (ST/T Changes)
  - HYP (Hypertrophy)
  - CD (Conduction Disturbance)
- **Input Shape**: `(1000 timesteps × 12 leads)`
- **Calibration**: Temperature Scaling
- **Training Samples Used**: 2,000 (balanced)
- **Frameworks**: TensorFlow, NumPy, WFDB

---

## 🛠 Tech Stack

### Backend
- Python 3.11
- FastAPI
- TensorFlow / Keras
- NumPy, Pandas
- WFDB
- Scikit-learn

### Frontend
- React.js
- TypeScript
- Tailwind CSS
- Fetch API

---

## 📂 Project Structure

ecg-arrhythmia-system/
├── backend/
│ ├── app/ # FastAPI routes
│ ├── ml/ # ML training & inference
│ ├── uploads/ # Uploaded ECG files
│ └── requirements.txt
├── frontend/
│ └── ecg-ui/ # React + TS frontend
├── data/
│ └── ptb-xl/ # Dataset (not pushed to Git)
├── README.md
└── .gitignore