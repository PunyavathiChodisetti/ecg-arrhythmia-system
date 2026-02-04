from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import tensorflow as tf

from backend.app.routes import predict

# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description="AI-powered ECG Arrhythmia Detection System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# GLOBAL MODEL STORAGE
# -------------------------------
MODEL = None
CLASSES = None

# -------------------------------
# LOAD MODEL ON STARTUP
# -------------------------------
@app.on_event("startup")
def load_model():
    global MODEL, CLASSES

    try:
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "ml", "ecg_cnn_model.keras")
        CLASSES_PATH = os.path.join(BASE_DIR, "ml", "classes.npy")

        MODEL = tf.keras.models.load_model(MODEL_PATH)
        CLASSES = np.load(CLASSES_PATH, allow_pickle=True)

        print("✅ Model and classes loaded successfully")

    except Exception as e:
        print("❌ Failed to load model:", e)

# -------------------------------
# ROUTES
# -------------------------------
predict.router.model = lambda: MODEL
predict.router.classes = lambda: CLASSES

app.include_router(predict.router)

@app.get("/")
def root():
    return {"message": "ECG Arrhythmia Detection API is running"}
