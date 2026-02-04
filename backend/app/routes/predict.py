from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
from backend.ml.cnn_inference import predict_ecg

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/predict")
async def predict_ecg_api(
    dat_file: UploadFile = File(...),
    hea_file: UploadFile = File(...)
):
    # Validate extensions
    if not dat_file.filename.endswith(".dat"):
        raise HTTPException(status_code=400, detail="Invalid .dat file")

    if not hea_file.filename.endswith(".hea"):
        raise HTTPException(status_code=400, detail="Invalid .hea file")

    base_name = dat_file.filename.replace(".dat", "")

    dat_path = os.path.join(UPLOAD_DIR, f"{base_name}.dat")
    hea_path = os.path.join(UPLOAD_DIR, f"{base_name}.hea")

    # Save files
    with open(dat_path, "wb") as f:
        shutil.copyfileobj(dat_file.file, f)

    with open(hea_path, "wb") as f:
        shutil.copyfileobj(hea_file.file, f)

    # Predict
    result = predict_ecg(os.path.join(UPLOAD_DIR, base_name))

    return {
        "filename": base_name,
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 4)
    }
