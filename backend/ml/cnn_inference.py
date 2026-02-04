import os
import numpy as np
import wfdb
import tensorflow as tf

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "ecg_saved_model")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.npy")

_model = None
_classes = None

def load_model_and_classes():
    global _model, _classes

    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _classes is None:
        _classes = np.load(CLASSES_PATH, allow_pickle=True)

    return _model, _classes


def preprocess_ecg(record_path):
    signal, _ = wfdb.rdsamp(record_path)
    signal = signal[:1000, :12]
    signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-8)
    return np.expand_dims(signal, axis=0)


def predict_ecg(record_path):
    model, classes = load_model_and_classes()
    X = preprocess_ecg(record_path)
    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))

    return {
        "prediction": str(classes[idx]),
        "confidence": float(probs[idx])
    }
