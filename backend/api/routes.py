"""HTTP routes. Application logic lives in backend.services and backend.ml."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from backend.ml.preprocessing import DrawingInputError, preprocess_png
from backend.schemas import (
    GameMetaResponse, HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse,
)


router = APIRouter()
log = logging.getLogger("glitch.api")


def _service(request: Request):
    service = request.app.state.prediction_service
    if service is None:
        raise HTTPException(status_code=503, detail=request.app.state.model_error or "Model is unavailable.")
    return service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = request.app.state.prediction_service
    return HealthResponse(status="ok", model_ready=service is not None,
                          model_version=service.model_version if service else None,
                          detail=request.app.state.model_error)


@router.get("/health/ready", response_model=HealthResponse)
def ready(request: Request) -> HealthResponse:
    service = _service(request)
    return HealthResponse(status="ready", model_ready=True, model_version=service.model_version)


@router.get("/api/model", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    service = request.app.state.prediction_service
    if service is None:
        return ModelInfoResponse(model_ready=False, model_version=None, architecture=None, parameters=None,
                                 classes=[], difficulty={}, temperature=1.0, uncertain_below=0.5,
                                 test_metrics={}, requests_served=0, mean_inference_ms=None)
    bundle = service.bundle
    return ModelInfoResponse(
        model_ready=True,
        model_version=bundle.version,
        architecture=bundle.engine.name,
        parameters=bundle.engine.parameter_count,
        classes=bundle.class_names,
        difficulty=bundle.difficulty,
        temperature=bundle.temperature,
        uncertain_below=bundle.uncertain_below,
        test_metrics=bundle.metrics,
        requests_served=service.requests,
        mean_inference_ms=round(service.total_ms / service.requests, 3) if service.requests else None,
    )


@router.get("/classes")
def classes(request: Request) -> dict:
    service = request.app.state.prediction_service
    names = service.bundle.class_names if service else []
    return {"model_ready": service is not None, "classes": names, "count": len(names)}


@router.get("/game/meta", response_model=GameMetaResponse)
def game_meta(request: Request) -> GameMetaResponse:
    service = request.app.state.prediction_service
    return GameMetaResponse(
        model_ready=service is not None,
        model_version=service.model_version if service else None,
        temperature=service.bundle.temperature if service else 1.0,
        difficulty=service.bundle.difficulty if service else {},
    )


async def _predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    service = _service(request)
    if payload.target is not None and payload.target not in service.bundle.class_names:
        raise HTTPException(status_code=422, detail=f"Unknown target class '{payload.target}'.")
    try:
        bitmap = await run_in_threadpool(preprocess_png, payload.image_base64)
    except DrawingInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        result = await run_in_threadpool(service.predict_bitmap, bitmap, payload.target)
    except Exception as exc:  # never invent a prediction
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc
    return PredictionResponse(**result, request_id=payload.request_id)


@router.post("/api/predict", response_model=PredictionResponse)
async def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    return await _predict(request, payload)


@router.post("/predict", response_model=PredictionResponse, include_in_schema=False)
async def predict_legacy(request: Request, payload: PredictionRequest) -> PredictionResponse:
    return await _predict(request, payload)
