import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.model import model_manager, CLASS_NAMES
from app.schemas import PredictResponse, HealthResponse

logger = logging.getLogger("aerovision")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    try:
        model_manager.load()
        logger.info(
            "Model loaded: mode=%s, device=%s",
            model_manager.inference_mode,
            model_manager.device,
        )
    except Exception as e:
        logger.error("Failed to load model: %s", e)
    yield


app = FastAPI(
    title="AeroVision LK API",
    description="YOLOv8 + SAHI aerial vehicle detection for Sri Lankan roads",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    status = "ok" if model_manager.detection_model is not None else "model_not_loaded"
    return HealthResponse(
        status=status,
        model_version=model_manager.model_version,
        inference_mode=model_manager.inference_mode,
        device=model_manager.device,
    )


@app.get("/classes")
async def get_classes():
    return {"classes": CLASS_NAMES, "n_classes": len(CLASS_NAMES)}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if model_manager.detection_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected image file, got {file.content_type}",
        )

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        detections, elapsed_ms = model_manager.predict(image_bytes)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return PredictResponse(
        detections=detections,
        n_detections=len(detections),
        inference_time_ms=round(elapsed_ms, 1),
        model_version=model_manager.model_version,
        inference_mode=model_manager.inference_mode,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
