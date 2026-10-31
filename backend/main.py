"""FastAPI application: loads the CNN once, serves the inference API and the game.

    uvicorn backend.main:app --host 127.0.0.1 --port 8001
    open http://127.0.0.1:8001
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.ml.model import load_bundle
from backend.services.prediction import PredictionService


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("GLITCH_MODEL_DIR", os.environ.get("DOODLE_MODEL_DIR", ROOT / "backend" / "model")))
FRONTEND_DIST = Path(os.environ.get("GLITCH_FRONTEND_DIR", ROOT / "frontend" / "dist"))
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5500,http://127.0.0.1:5500"
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("GLITCH_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()]
MAX_BODY_BYTES = 3_500_000

logging.basicConfig(level=os.environ.get("GLITCH_LOG_LEVEL", "INFO"), format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("glitch")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_service = None
    app.state.model_error = None
    try:
        bundle = load_bundle(Path(MODEL_DIR))
        app.state.prediction_service = PredictionService(bundle)
        log.info("Model %s ready: %s params, %d classes, T=%.3f, uncertain<%.2f (%s)", bundle.version,
                 f"{bundle.engine.parameter_count:,}", len(bundle.class_names), bundle.temperature,
                 bundle.uncertain_below, bundle.source)
    except Exception as exc:  # the API still starts and reports the problem on /health
        app.state.model_error = f"Model could not be loaded: {exc}"
        log.error(app.state.model_error)
    yield
    app.state.prediction_service = None


app = FastAPI(title="GLITCH: Memory Recovery Protocol - inference API", version="1.0.0", lifespan=lifespan)
app.state.prediction_service = None
app.state.model_error = "Model has not loaded yet."
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    length = request.headers.get("content-length")
    if length is not None:
        try:
            too_big = int(length) > MAX_BODY_BYTES
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)
        if too_big:
            return JSONResponse({"detail": "Request body too large (max 3.5 MB)."}, status_code=413)
    return await call_next(request)


app.include_router(router)


@app.get("/collect", include_in_schema=False)
def collect_page() -> FileResponse:
    """Drawing-collection page for ml/eval_canvas.py (real-canvas evaluation)."""
    return FileResponse(ROOT / "collect.html")

if (FRONTEND_DIST / "index.html").is_file():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets", check_dir=False), name="assets")

    @app.get("/", include_in_schema=False)
    def game() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html", headers={"Cache-Control": "no-cache"})
else:
    @app.get("/", include_in_schema=False)
    def game_missing() -> JSONResponse:
        return JSONResponse({"detail": "Frontend build not found. Run: cd frontend && npm run build"}, status_code=404)
