import numpy as np
import wfdb
import tensorflow as tf

model = tf.keras.models.load_model("backend/ml/ecg_cnn_model")
classes = np.load("backend/ml/classes.npy", allow_pickle=True)

def preprocess_ecg(record_path):
    signal, _ = wfdb.rdsamp(record_path)

    signal = signal[:1000, :12]
    signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-8)

    return np.expand_dims(signal, axis=0)

def predict_ecg(record_path):
    X = preprocess_ecg(record_path)
    probs = model.predict(X)[0]

    idx = np.argmax(probs)

    return {
        "prediction": classes[idx],
        "confidence": float(probs[idx])
    }
