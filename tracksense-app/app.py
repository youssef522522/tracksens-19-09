import io
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "tracksense.db"
MODEL_PATH = APP_DIR / "best.pt"

app = FastAPI(title="TrackSense API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(get_db()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                mode TEXT,
                section TEXT,
                startKP TEXT,
                endKP TEXT,
                distance REAL,
                date TEXT,
                status TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id TEXT,
                type TEXT,
                severity TEXT,
                chainage TEXT,
                confidence INTEGER,
                FOREIGN KEY (inspection_id) REFERENCES inspections(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS defects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id TEXT,
                defect_type TEXT,
                count INTEGER,
                FOREIGN KEY (inspection_id) REFERENCES inspections(id)
            )
            """
        )


init_db()

# ---------------------------------------------------------------------------
# Model (loaded once at startup)
# ---------------------------------------------------------------------------

_model = None
_model_error = None


def get_model():
    """Lazily load the ultralytics YOLO classification model."""
    global _model, _model_error
    if _model is None and _model_error is None:
        try:
            # Older ultralytics .pt checkpoints pickle full model objects, not just
            # state dicts. PyTorch >=2.6 defaults torch.load(weights_only=True),
            # which blocks unpickling those objects and makes every /api/detect
            # call fail. Force weights_only=False for this trusted, local file.
            import torch

            _original_torch_load = torch.load

            def _patched_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _original_torch_load(*args, **kwargs)

            torch.load = _patched_load

            from ultralytics import YOLO

            _model = YOLO(str(MODEL_PATH))
        except Exception as exc:  # noqa: BLE001
            _model_error = f"{type(exc).__name__}: {exc}"
    return _model, _model_error


# ---------------------------------------------------------------------------
# Pydantic bodies
# ---------------------------------------------------------------------------


class InspectionCreate(BaseModel):
    id: str
    mode: str
    section: str
    startKP: str
    targetDistance: float


class InspectionComplete(BaseModel):
    endKP: str
    defects: dict


class DetectionCreate(BaseModel):
    type: str
    severity: str
    chainage: str
    confidence: int


# ---------------------------------------------------------------------------
# Routes: inspections (ported 1:1 from server.js)
# ---------------------------------------------------------------------------


@app.get("/api/stats")
def get_stats():
    with closing(get_db()) as conn:
        total = conn.execute("SELECT COUNT(*) AS count FROM inspections").fetchone()["count"]
        total_distance = conn.execute(
            "SELECT COALESCE(SUM(distance), 0) AS total FROM inspections"
        ).fetchone()["total"]
        recent = conn.execute(
            "SELECT * FROM inspections ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
    return {
        "totalInspections": total,
        "totalDistance": total_distance,
        "recentCount": len(recent),
        "recent": [dict(r) for r in recent],
    }


@app.get("/api/inspections")
def list_inspections():
    with closing(get_db()) as conn:
        rows = conn.execute("SELECT * FROM inspections ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/inspections/{inspection_id}")
def get_inspection(inspection_id: str):
    with closing(get_db()) as conn:
        inspection = conn.execute(
            "SELECT * FROM inspections WHERE id = ?", (inspection_id,)
        ).fetchone()
        if not inspection:
            raise HTTPException(status_code=404, detail="Inspection not found")
        detections = conn.execute(
            "SELECT * FROM detections WHERE inspection_id = ?", (inspection_id,)
        ).fetchall()
        defects = conn.execute(
            "SELECT defect_type, count FROM defects WHERE inspection_id = ?", (inspection_id,)
        ).fetchall()

    result = dict(inspection)
    result["detections"] = [dict(d) for d in detections]
    result["defectsMap"] = {d["defect_type"]: d["count"] for d in defects}
    return result


@app.post("/api/inspections")
def create_inspection(body: InspectionCreate):
    with closing(get_db()) as conn, conn:
        conn.execute(
            "INSERT INTO inspections (id, mode, section, startKP, distance, date, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                body.id,
                body.mode,
                body.section,
                body.startKP,
                body.targetDistance,
                time.strftime("%Y-%m-%d"),
                "active",
            ),
        )
    return {"success": True, "id": body.id}


@app.post("/api/inspections/{inspection_id}/detections")
def add_detection(inspection_id: str, body: DetectionCreate):
    with closing(get_db()) as conn, conn:
        conn.execute(
            "INSERT INTO detections (inspection_id, type, severity, chainage, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (inspection_id, body.type, body.severity, body.chainage, body.confidence),
        )
    return {"success": True}


@app.post("/api/inspections/{inspection_id}/complete")
def complete_inspection(inspection_id: str, body: InspectionComplete):
    with closing(get_db()) as conn, conn:
        conn.execute(
            "UPDATE inspections SET endKP = ?, status = ? WHERE id = ?",
            (body.endKP, "completed", inspection_id),
        )
        for defect_type, count in body.defects.items():
            conn.execute(
                "INSERT INTO defects (inspection_id, defect_type, count) VALUES (?, ?, ?)",
                (inspection_id, defect_type, count),
            )
    return {"success": True}


# ---------------------------------------------------------------------------
# Route: AI defect detection
# ---------------------------------------------------------------------------


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    model, error = get_model()
    if error:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {error}")
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image")

    try:
        results = model.predict(image, verbose=False)
        r = results[0]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}: {exc}")

    if getattr(r, "probs", None) is None:
        raise HTTPException(
            status_code=500,
            detail="Model did not return classification probabilities (unexpected model type)",
        )

    probs = r.probs
    names = r.names
    top1_idx = int(probs.top1)
    top1_conf = float(probs.top1conf)
    all_probs = {names[i]: round(float(probs.data[i]) * 100, 2) for i in range(len(names))}
    all_probs_sorted = dict(sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True))

    return JSONResponse(
        {
            "predicted_class": names[top1_idx],
            "confidence": round(top1_conf * 100, 2),
            "all_class_probs": all_probs_sorted,
        }
    )


@app.get("/api/health")
def health():
    _, error = get_model()
    return {"status": "ok", "model_loaded": error is None}


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

STATIC_DIR = APP_DIR / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
