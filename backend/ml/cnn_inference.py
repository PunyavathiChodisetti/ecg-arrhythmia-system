import numpy as np
import wfdb

# -------------------------------
# PREPROCESS ECG
# -------------------------------
def preprocess_ecg(record_path: str) -> np.ndarray:
    """
    Read ECG record and preprocess to model input shape (1, 1000, 12)
    """
    signal, _ = wfdb.rdsamp(record_path)

    # Ensure correct shape
    signal = signal[:1000, :12]

    # Normalize per channel
    signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-8)

    # Add batch dimension
    return np.expand_dims(signal, axis=0)


# -------------------------------
# PREDICT ECG (MODEL INJECTED)
# -------------------------------
def predict_ecg(record_path: str, model, classes) -> dict:
    """
    Run ECG inference using preloaded model and classes
    """
    if model is None or classes is None:
        raise RuntimeError("Model or classes not loaded")

    X = preprocess_ecg(record_path)

    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))

    return {
        "prediction": str(classes[idx]),
        "confidence": float(probs[idx])
    }
