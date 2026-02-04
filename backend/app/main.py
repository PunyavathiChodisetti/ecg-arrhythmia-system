from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from keras.models import load_model
import numpy as np

from backend.app.routes import predict

# -------------------------------
# APP INITIALIZATION
# -------------------------------
app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description="AI-powered ECG Arrhythmia Detection System",
    version="1.0.0"
)

# -------------------------------
# CORS CONFIG
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# LOAD MODEL & CLASSES (ONCE)
# -------------------------------
MODEL_PATH = "backend/ml/ecg_cnn_model.keras"
CLASSES_PATH = "backend/ml/classes.npy"

try:
    model = load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    # Make them available to predict router
    predict.model = model
    predict.classes = classes

    print("✅ Model and classes loaded successfully")

except Exception as e:
    print("❌ Failed to load model:", e)
    raise e

# -------------------------------
# ROUTES
# -------------------------------
app.include_router(predict.router)

@app.get("/")
def root():
    return {"message": "ECG Arrhythmia Detection API is running"}
