"""Pydantic response models for the inference API."""

from __future__ import annotations

from pydantic import BaseModel


class Detection(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str
    score: float


class InferenceResponse(BaseModel):
    detections: list[Detection]
    inference_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
