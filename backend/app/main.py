from fastapi import FastAPI
from backend.app.routes import predict
from fastapi.middleware.cors import CORSMiddleware

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

app.include_router(predict.router)

@app.get("/")
def root():
    return {"message": "ECG Arrhythmia Detection API is running"}
