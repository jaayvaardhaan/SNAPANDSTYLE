import os
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CV_DIR = PROJECT_ROOT / "fashion-cv"
sys.path.insert(0, str(CV_DIR))

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from appearance_pipeline import analyze_array
from face_shape_analysis import MODEL_PATH as LANDMARK_MODEL_PATH, landmarker_is_loaded
from face_segmentation import MODEL_PATH as PARSING_MODEL_PATH, parser_is_loaded
from saved_looks_api import router as saved_looks_router

BASE_DIR = CV_DIR
FRONTEND_DIR = BASE_DIR.parent / "frontend"
PROJECT_ROOT = BASE_DIR.parent
IMAGE_DIR = PROJECT_ROOT / "datasets" / "kaggle_fashion" / "images"
CATALOG_PATH = PROJECT_ROOT / "datasets" / "snapstyle_items_tagged.csv"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MAX_LONG_SIDE = 1920

app = FastAPI(
    title="Snap&Style backend API",
    description="Local frontend, appearance analysis, recommendations, catalog images and saved looks.",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
_cv_slot = threading.BoundedSemaphore(value=1)
app.include_router(saved_looks_router)


def _model_status():
    return {
        "face_landmarker": {
            "present": LANDMARK_MODEL_PATH.is_file(),
            "loaded": landmarker_is_loaded(),
        },
        "face_parser": {
            "present": PARSING_MODEL_PATH.is_file(),
            "loaded": parser_is_loaded(),
        },
    }


def decode_jpeg(image_bytes):
    """Validate, decode and safely resize one JPEG byte string."""
    if len(image_bytes) < 3 or image_bytes[:3] != b"\xff\xd8\xff":
        raise ValueError("Body is not a JPEG image")
    try:
        picture = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
    except cv2.error as exc:
        raise ValueError("Invalid JPEG data") from exc
    if picture is None:
        raise ValueError("JPEG could not be decoded")

    height, width = picture.shape[:2]
    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(413, detail="Image dimensions are too large")
    if min(width, height) < 256:
        raise ValueError("Face photo is too small (minimum 256 pixels per side)")
    if max(width, height) > MAX_LONG_SIDE:
        factor = MAX_LONG_SIDE / max(width, height)
        picture = cv2.resize(
            picture,
            (round(width * factor), round(height * factor)),
            interpolation=cv2.INTER_AREA,
        )
    return picture


@app.middleware("http")
async def prevent_api_caching(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response

@app.get("/api/health")
def health(response: Response):
    models = _model_status()
    catalog = {"metadata_present": CATALOG_PATH.is_file(), "images_present": IMAGE_DIR.is_dir()}
    ready = all(item["present"] for item in models.values()) and all(catalog.values())
    if not ready:
        response.status_code = 503
    return {
        "status": "ready" if ready else "unavailable",
        "storage": "in_memory",
        "models": models,
        "catalog": catalog,
    }


@app.post("/api/analyze")
async def analyze(request: Request):
    if request.headers.get("content-type", "").split(";", 1)[0].lower().strip() != "image/jpeg":
        raise HTTPException(415, detail="Send the camera JPEG with Content-Type: image/jpeg")

    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, detail="JPEG is too large (max 8 MB)")
        chunks.append(chunk)
    if not total:
        raise HTTPException(400, detail="Empty photo")

    image_bytes = b"".join(chunks)
    del chunks

    if not _cv_slot.acquire(blocking=False):
        raise HTTPException(429, detail="CV is processing another photo; please retry in a moment")
    try:
        try:
            picture = await run_in_threadpool(decode_jpeg, image_bytes)
            del image_bytes
            profile = await run_in_threadpool(analyze_array, picture)
        except HTTPException:
            raise
        except FileNotFoundError as exc:
            print(f"CV model file missing: {exc}", flush=True)
            raise HTTPException(503, detail="CV model files are unavailable") from exc
        except RuntimeError as exc:
            print(f"CV model initialization error: {exc}", flush=True)
            raise HTTPException(503, detail="CV models could not be initialized") from exc
        except (ValueError, cv2.error) as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        except Exception as exc:
            print(f"CV processing error: {type(exc).__name__}: {exc}", flush=True)
            raise HTTPException(500, detail="CV analysis could not complete. Check server logs and model files.") from exc
    finally:
        _cv_slot.release()
    keys = ("face_shape", "skin_tone", "skin_undertone", "hair_color",
            "confidence", "quality_warnings")
    return {"profile": {key: profile[key] for key in keys}}


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from recommendation_engine.api.recommendation_api import app as recommendation_app

app.mount("/api/recommendations", recommendation_app, name="recommendations")
if IMAGE_DIR.is_dir():
    app.mount("/catalog-images", StaticFiles(directory=str(IMAGE_DIR)), name="catalog-images")

@app.get("/frontend.html", include_in_schema=False)
def legacy_frontend():
    return RedirectResponse(url="/index.html")


# Mount LAST: API and image routes must be checked before static file routing.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("SNAPSTYLE_HOST", "127.0.0.1"),
                port=int(os.getenv("SNAPSTYLE_PORT", "8000")), log_level="info")
