import numpy as np
import wfdb

# -------------------------------
# PREPROCESS ECG
# -------------------------------
def preprocess_ecg(record_path: str) -> np.ndarray:
    signal, _ = wfdb.rdsamp(record_path)

    signal = signal[:1000, :12]
    signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-8)

    return np.expand_dims(signal, axis=0)


# -------------------------------
# PREDICT (MODEL INJECTED)
# -------------------------------
def predict_ecg(record_path: str, model, classes) -> dict:
    if model is None or classes is None:
        raise RuntimeError("Model or classes not loaded")

    X = preprocess_ecg(record_path)

    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))

    return {
        "prediction": str(classes[idx]),
        "confidence": float(probs[idx])
    }
