"""
main.py  –  AyurGenX FastAPI backend
Serves both the REST API and the static frontend.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add backend directory to sys.path so we can import cv_model directly.
# This fixes both FastAPI imports and PyTorch's torch.load() unpickling path.
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
MODEL_PATH     = BASE_DIR / "models" / "efficientnet_best.pt"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"
TREE_PATH      = BASE_DIR / "question_tree.json"
FRONTEND_DIR   = BASE_DIR.parent / "frontend"

# ── lazy-load heavy dependencies ──────────────────────────────────────────────
_face_model    = None
_question_eng  = None

def get_face_model():
    global _face_model
    if _face_model is None:
        from cv_model import FaceScanModel
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model weights not found. "
                    "Please place efficientnet_best.pt inside backend/models/."
                ),
            )
        _face_model = FaceScanModel(str(MODEL_PATH), str(LABEL_MAP_PATH))
    return _face_model

def get_question_engine():
    global _question_eng
    if _question_eng is None:
        from question_engine import QuestionEngine
        _question_eng = QuestionEngine(str(TREE_PATH))
    return _question_eng

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AyurGenX API",
    description="AI-powered Ayurvedic health assessment with face scanning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── pydantic schemas ──────────────────────────────────────────────────────────
class NextQuestionRequest(BaseModel):
    cv_conditions:   List[str]
    primary_concerns: List[str]
    answers:         Dict[str, Any]
    question_count:  int

class SummaryRequest(BaseModel):
    cv_results:       List[Dict]
    cv_extra:         Dict
    answers:          Dict[str, Any]
    primary_concerns: List[str]
    age_group:        str

# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/api/analyze")
async def analyze_image(image: UploadFile = File(...)):
    """
    Accept a face image, run the EfficientNet CV model,
    and return detected skin conditions + extra biomarkers.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await image.read()
    if len(image_bytes) > 10 * 1024 * 1024:   # 10 MB guard
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB).")

    try:
        model      = get_face_model()
        conditions = model.predict(image_bytes)
        extra_cv   = model.analyse_extra_cv(image_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error during face analysis")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "conditions": conditions,
        "extra_cv":   extra_cv,
        "condition_names": [c["condition"] for c in conditions],
    }


@app.get("/api/questions/initial")
def initial_questions():
    """Return the two initial onboarding questions (age + primary concern)."""
    engine = get_question_engine()
    return {"questions": engine.get_initial_questions()}


@app.post("/api/questions/next")
def next_question(body: NextQuestionRequest):
    """
    Given current state, return the next question or null if done.
    """
    engine = get_question_engine()
    q = engine.get_next_question(
        cv_conditions    = body.cv_conditions,
        primary_concerns = body.primary_concerns,
        answers          = body.answers,
        question_count   = body.question_count,
    )
    return {"question": q}


@app.post("/api/summary")
def generate_summary(body: SummaryRequest):
    """Synthesise all data into a final Ayurvedic health summary."""
    engine = get_question_engine()
    summary = engine.generate_summary(
        cv_results       = body.cv_results,
        cv_extra         = body.cv_extra,
        answers          = body.answers,
        primary_concerns = body.primary_concerns,
        age_group        = body.age_group,
    )
    return summary


# ── static frontend ───────────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR / "static")),
        name="static",
    )

    @app.get("/")
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/scan")
    def serve_scan():
        return FileResponse(str(FRONTEND_DIR / "scan.html"))

    @app.get("/chat")
    def serve_chat():
        return FileResponse(str(FRONTEND_DIR / "chat.html"))

    @app.get("/results")
    def serve_results():
        return FileResponse(str(FRONTEND_DIR / "results.html"))

# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
