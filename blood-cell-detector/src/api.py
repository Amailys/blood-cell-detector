from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.model import predict

app = FastAPI(
    title="Blood Cell Detector API",
    description="Blood cell detection and classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Blood Cell Detector API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# "def" (et non "async def") : FastAPI exécute alors l'inférence bloquante
# dans un thread, sans geler le serveur.
@app.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = file.file.read()
    try:
        return predict(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
