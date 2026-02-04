from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil

# Import inference function ONLY (no model loading here)
from backend.ml.cnn_inference import predict_ecg

router = APIRouter()

# These will be injected from main.py
model = None
classes = None

# -------------------------------
# UPLOAD CONFIG
# -------------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------
# PREDICT ENDPOINT
# -------------------------------
@router.post("/predict")
async def predict_ecg_api(
    dat_file: UploadFile = File(...),
    hea_file: UploadFile = File(...)
):
    # -------------------------------
    # BASIC VALIDATION
    # -------------------------------
    if not dat_file.filename.lower().endswith(".dat"):
        raise HTTPException(status_code=400, detail="Invalid .dat file")

    if not hea_file.filename.lower().endswith(".hea"):
        raise HTTPException(status_code=400, detail="Invalid .hea file")

    if model is None or classes is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded on server"
        )

    base_name = dat_file.filename.replace(".dat", "")

    dat_path = os.path.join(UPLOAD_DIR, f"{base_name}.dat")
    hea_path = os.path.join(UPLOAD_DIR, f"{base_name}.hea")

    try:
        # -------------------------------
        # SAVE FILES
        # -------------------------------
        with open(dat_path, "wb") as f:
            shutil.copyfileobj(dat_file.file, f)

        with open(hea_path, "wb") as f:
            shutil.copyfileobj(hea_file.file, f)

        # -------------------------------
        # RUN INFERENCE
        # -------------------------------
        result = predict_ecg(
            record_path=os.path.join(UPLOAD_DIR, base_name),
            model=model,
            classes=classes
        )

        return {
            "filename": base_name,
            "prediction": result["prediction"],
            "confidence": round(float(result["confidence"]), 4)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:
        # -------------------------------
        # CLEANUP FILES (OPTIONAL BUT GOOD)
        # -------------------------------
        if os.path.exists(dat_path):
            os.remove(dat_path)
        if os.path.exists(hea_path):
            os.remove(hea_path)
