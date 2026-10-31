"""One drawing in, one honest prediction out.

The service owns the loaded model bundle and turns a preprocessed 28x28
bitmap into the game's response: calibrated top-k scores, an uncertainty flag
from the validation-fitted policy, an analytic Grad-CAM map ("where GLITCH
looked"), and the exact bitmap the CNN received. Nothing here is invented: if
inference fails, the route returns an error and the game shows a retry.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time

import numpy as np
from PIL import Image

from backend.ml.explain import focus_region, uncertainty
from backend.ml.model import ModelBundle, temperature_scale, top_predictions


log = logging.getLogger("glitch.inference")


def bitmap_png(bitmap: np.ndarray) -> str:
    """The 28x28 model input as a PNG data URL (white strokes on black, as the CNN sees it)."""
    image = Image.fromarray(np.asarray(bitmap, dtype=np.uint8).reshape(28, 28), "L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class PredictionService:
    def __init__(self, bundle: ModelBundle, top_k: int = 5):
        self.bundle = bundle
        self.top_k = top_k
        self._lock = threading.Lock()  # NumPy is thread-safe, but this keeps latency numbers clean
        self.requests = 0
        self.total_ms = 0.0

    @property
    def model_version(self) -> str:
        return self.bundle.version

    def predict_bitmap(self, bitmap: np.ndarray, target: str | None = None) -> dict:
        start = time.perf_counter()
        with self._lock:
            names = self.bundle.class_names
            # One forward pass: probabilities plus Grad-CAM for the top-1 class.
            # Temperature scaling never changes the argmax, so explaining the
            # raw top-1 explains the calibrated top-1 too.
            explained = self.bundle.engine.gradcam(bitmap)
            if explained is None:
                raw, cam = self.bundle.engine.run(bitmap)["output"][0], None
            else:
                cam, raw = explained
            calibrated = temperature_scale(raw, self.bundle.temperature)
            target_score = None
            if target is not None and target in names:
                target_score = float(calibrated[names.index(target)])
        elapsed = (time.perf_counter() - start) * 1000
        self.requests += 1
        self.total_ms += elapsed
        ranked = top_predictions(calibrated, names, limit=self.top_k)
        stats = uncertainty(calibrated, names)
        top_score = ranked[0]["score"]
        response = {
            "predictions": ranked,
            "uncertain": bool(top_score < self.bundle.uncertain_below),
            "uncertain_below": self.bundle.uncertain_below,
            "entropy": stats["entropy"],
            "margin": stats["margin"],
            "target": target if target in names else None,
            "target_score": target_score,
            "target_rank": (next((i + 1 for i, row in enumerate(top_predictions(calibrated, names, len(names)))
                                  if row["class_name"] == target), None) if target in names else None),
            "attention": cam.round(3).tolist() if cam is not None else None,
            "attention_focus": focus_region(cam) if cam is not None else None,
            "model_input": bitmap_png(bitmap),
            "model_version": self.bundle.version,
            "temperature": self.bundle.temperature,
            "confidence_type": "temperature_scaled_softmax" if self.bundle.temperature != 1.0 else "softmax",
            "inference_ms": round(elapsed, 3),
        }
        log.info("predict top=%s score=%.3f target=%s ms=%.2f", ranked[0]["class_name"], top_score, target, elapsed)
        return response
