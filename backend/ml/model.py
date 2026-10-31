"""Load the active CNN, its calibration, uncertainty policy and difficulty metadata.

Lookup order (first match wins):
1. backend/model/registry.json -> versions/<active>/model.weights.npz + model.arch.json
2. backend/model/doodle_classifier.weights.npz + doodle_classifier.arch.json
3. backend/model/doodle_classifier.keras, exported on the fly by ml/export_weights.py

Inference always runs through the NumPy engine (backend/ml/engine.py), so the
API does not need TensorFlow. The class mapping is always class_names.json;
it is never inferred or re-ordered.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.ml.engine import NumpyCNN


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
DIFFICULTY_LEVELS = ("easy", "medium", "hard")
DEFAULT_UNCERTAIN_BELOW = 0.5
log = logging.getLogger("glitch.model")


@dataclass
class ModelBundle:
    engine: NumpyCNN
    class_names: list[str]
    version: str
    temperature: float = 1.0
    uncertain_below: float = DEFAULT_UNCERTAIN_BELOW
    difficulty: dict[str, str] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    source: str = ""


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def load_class_names(model_dir: Path) -> list[str]:
    names = _read_json(model_dir / "class_names.json")
    if not isinstance(names, list) or len(names) < 2 or len(set(names)) != len(names) \
            or not all(isinstance(n, str) and n for n in names):
        raise ValueError("class_names.json must list at least two unique labels.")
    return names


def load_temperature(model_dir: Path = DEFAULT_MODEL_DIR) -> float:
    """Calibration temperature fitted on the validation split; 1.0 if absent."""
    data = _read_json(model_dir / "calibration.json", {}) or {}
    try:
        value = float(data["temperature"])
        return value if 0.05 <= value <= 20 else 1.0
    except (KeyError, TypeError, ValueError):
        return 1.0


def load_difficulty(class_names: list[str], model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, str]:
    """Per-class prompt difficulty written by the evaluation scripts; 'medium' if unknown."""
    tiers = (_read_json(model_dir / "class_difficulty.json", {}) or {}).get("tiers", {})
    return {name: tiers.get(name) if tiers.get(name) in DIFFICULTY_LEVELS else "medium" for name in class_names}


def _resolve_files(model_dir: Path) -> tuple[Path, Path, Path, str]:
    """Return (weights, arch, metadata_dir, version_label)."""
    registry = _read_json(model_dir / "registry.json")
    if isinstance(registry, dict) and registry.get("active"):
        version_dir = model_dir / "versions" / str(registry["active"])
        weights, arch = version_dir / "model.weights.npz", version_dir / "model.arch.json"
        if weights.is_file() and arch.is_file():
            return weights, arch, version_dir, str(registry["active"])
        log.warning("Registry points at %s but its files are missing; falling back.", version_dir)
    weights = model_dir / "doodle_classifier.weights.npz"
    arch = model_dir / "doodle_classifier.arch.json"
    if not (weights.is_file() and arch.is_file()):
        keras_path = model_dir / "doodle_classifier.keras"
        if not keras_path.is_file():
            raise FileNotFoundError(
                "No model found. Expected backend/model/registry.json, doodle_classifier.weights.npz, "
                "or doodle_classifier.keras (run ml/train.py, then ml/export_weights.py).")
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
        from export_weights import export  # noqa: E402

        log.info("Exporting %s to the NumPy format", keras_path)
        export(keras_path, model_dir / "doodle_classifier")
    return weights, arch, model_dir, f"local-{_digest(weights)}"


def load_bundle(model_dir: Path = DEFAULT_MODEL_DIR) -> ModelBundle:
    class_names = load_class_names(model_dir)
    weights, arch, meta_dir, version = _resolve_files(model_dir)
    engine = NumpyCNN.load(weights, arch)
    if engine.output_classes != len(class_names):
        raise ValueError(f"Model outputs {engine.output_classes} classes but class_names.json lists {len(class_names)}.")
    policy = _read_json(meta_dir / "confidence_policy.json", {}) or {}
    metrics = _read_json(meta_dir / "test_metrics.json", {}) or {}
    summary = {key: metrics.get(key) for key in ("top_1_accuracy", "top_3_accuracy", "macro_f1", "examples") if key in metrics}
    if "ece_temperature_scaled" in metrics:
        summary["ece_temperature_scaled"] = metrics["ece_temperature_scaled"]
    elif isinstance(metrics.get("calibration"), dict):
        summary["ece_temperature_scaled"] = metrics["calibration"].get("ece_temperature_scaled")
    uncertain = policy.get("uncertain_below", DEFAULT_UNCERTAIN_BELOW)
    try:
        uncertain = float(uncertain)
    except (TypeError, ValueError):
        uncertain = DEFAULT_UNCERTAIN_BELOW
    return ModelBundle(
        engine=engine,
        class_names=class_names,
        version=version,
        temperature=load_temperature(meta_dir) if (meta_dir / "calibration.json").exists() else load_temperature(model_dir),
        uncertain_below=min(max(uncertain, 0.0), 0.99),
        difficulty=load_difficulty(class_names, meta_dir if (meta_dir / "class_difficulty.json").exists() else model_dir),
        metrics=summary,
        source=str(weights.relative_to(model_dir.parent)) if model_dir.parent in weights.parents else str(weights),
    )


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(log(p) / T): identical to dividing the logits by T."""
    if temperature == 1.0:
        return np.asarray(probabilities, dtype=np.float64)
    logits = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=-1, keepdims=True)


def top_predictions(probabilities: np.ndarray, class_names: list[str], limit: int = 3) -> list[dict]:
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(scores) != len(class_names) or len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("Model scores do not match the configured class labels.")
    if np.any(scores < 0) or not np.isclose(float(scores.sum()), 1.0, atol=1e-3):
        raise ValueError("Model scores are not a valid softmax probability vector.")
    count = min(max(1, limit), len(class_names))
    indices = np.argsort(scores)[-count:][::-1]
    return [{"class_name": class_names[i], "score": float(scores[i])} for i in indices]
