"""Request and response types for the GLITCH inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    image_base64: str = Field(
        min_length=1,
        max_length=3_000_000,
        description="The canvas as a PNG, base64-encoded; a data:image/png;base64, prefix is optional.",
    )
    target: str | None = Field(default=None, max_length=40,
                               description="Optional target class; the response then includes its score and rank.")
    request_id: str | None = Field(default=None, max_length=64,
                                   description="Echoed back so the client can discard stale responses.")


class ClassScore(BaseModel):
    class_name: str
    score: float = Field(ge=0, le=1, description="Temperature-scaled softmax score (not a guaranteed probability).")


class PredictionResponse(BaseModel):
    predictions: list[ClassScore]
    uncertain: bool = Field(description="Top-1 score is below the validation-fitted uncertainty threshold.")
    uncertain_below: float
    entropy: float = Field(description="Normalised entropy of the scores, 0 (certain) to 1 (uniform).")
    margin: float = Field(description="Top-1 minus top-2 score.")
    target: str | None = None
    target_score: float | None = None
    target_rank: int | None = None
    attention: list[list[float]] | None = Field(default=None, description="Grad-CAM map for the top-1 class, 0-1, at feature resolution.")
    attention_focus: str | None = None
    model_input: str = Field(description="The exact 28x28 bitmap the CNN received, as a PNG data URL.")
    model_version: str
    temperature: float
    confidence_type: str
    inference_ms: float
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_version: str | None = None
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_ready: bool
    model_version: str | None
    architecture: str | None
    parameters: int | None
    classes: list[str]
    difficulty: dict[str, str]
    temperature: float
    uncertain_below: float
    test_metrics: dict
    requests_served: int
    mean_inference_ms: float | None
    note: str = ("The CNN was trained offline on Quick, Draw! bitmaps. Drawing in the game runs inference only; "
                 "it never updates the model's weights.")


class GameMetaResponse(BaseModel):
    """Kept for the older Doodle Panic client and collect.html."""
    model_ready: bool
    model_version: str | None
    temperature: float
    difficulty: dict[str, str]
