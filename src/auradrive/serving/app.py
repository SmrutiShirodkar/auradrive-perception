"""
FastAPI serving layer for the AuraDrive CAM_FRONT perception model.

This operationalizes the report's "over-the-air deployment" concept
(section 2.2.4: "trained models are packaged and deployed back to the
fleet") as a real, runnable inference service rather than a diagram.

Run:
    uvicorn auradrive.serving.app:app --reload

    curl -F "file=@some_image.jpg" http://localhost:8000/predict
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision.transforms import functional as F

from auradrive.serving.model_registry import DEFAULT_CHECKPOINT, LoadedModel, load_model
from auradrive.serving.schemas import Detection, HealthResponse, InferenceResponse
from auradrive.training.labels import CLASS_NAMES

SCORE_THRESHOLD = 0.5

_state: dict[str, LoadedModel | None] = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["model"] = load_model(DEFAULT_CHECKPOINT)
    except FileNotFoundError:
        _state["model"] = None  # /health reports this; /predict rejects cleanly
    yield
    _state["model"] = None


app = FastAPI(title="AuraDrive Perception Serving", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _state["model"]
    return HealthResponse(
        status="ok",
        model_loaded=loaded is not None,
        model_version=loaded.version if loaded else None,
    )


@app.post("/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)) -> InferenceResponse:
    loaded = _state["model"]
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a checkpoint first (see auradrive.training.train).",
        )

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    tensor = F.to_tensor(image).to(loaded.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = loaded.model([tensor])[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    keep = output["scores"] > SCORE_THRESHOLD
    detections = [
        Detection(
            x_min=float(box[0]),
            y_min=float(box[1]),
            x_max=float(box[2]),
            y_max=float(box[3]),
            label=CLASS_NAMES[int(label)],
            score=float(score),
        )
        for box, label, score in zip(
            output["boxes"][keep], output["labels"][keep], output["scores"][keep]
        )
    ]

    return InferenceResponse(
        detections=detections,
        inference_ms=elapsed_ms,
        model_version=loaded.version,
    )
