import os
import numpy as np
import wfdb
import tensorflow as tf

# -------------------------------
# PATHS (SAFE)
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "ecg_cnn_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.npy")

# -------------------------------
# LAZY LOAD MODEL
# -------------------------------
_model = None
_classes = None

def load_model_and_classes():
    global _model, _classes

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _classes is None:
        if not os.path.exists(CLASSES_PATH):
            raise FileNotFoundError(f"Classes not found at {CLASSES_PATH}")
        _classes = np.load(CLASSES_PATH, allow_pickle=True)

    return _model, _classes

# -------------------------------
# PREPROCESS ECG
# -------------------------------
def preprocess_ecg(record_path):
    signal, _ = wfdb.rdsamp(record_path)

    signal = signal[:1000, :12]
    signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-8)

    return np.expand_dims(signal, axis=0)

# -------------------------------
# PREDICT
# -------------------------------
def predict_ecg(record_path):
    model, classes = load_model_and_classes()

    X = preprocess_ecg(record_path)
    probs = model.predict(X, verbose=0)[0]

    idx = int(np.argmax(probs))

    return {
        "prediction": str(classes[idx]),
        "confidence": float(probs[idx])
    }
