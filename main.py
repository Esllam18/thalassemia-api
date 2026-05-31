
import io
import logging
import os
import re
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import cv2
import joblib
import numpy as np
import pandas as pd
import pytesseract
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

# ---- Logging ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("thalassemia_api")

# ---- Config -----------------------------------------------------------------
MODEL_PATH    = os.getenv("MODEL_PATH",   "thalassemia_expert_model.pkl")
ENCODER_PATH  = os.getenv("ENCODER_PATH", "label_encoder.pkl")
MAX_BYTES     = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MCV_THRESHOLD = float(os.getenv("MCV_NORMAL_THRESHOLD", "81.0"))

# Max image dimensions to prevent memory attacks (e.g. 20000x20000 PNG)
MAX_IMAGE_DIM = 8000

raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8088")
ALLOWED_ORIGINS: List[str] = [o.strip() for o in raw_origins.split(",") if o.strip()]

VALID_RANGES: Dict[str, tuple] = {
    "HGB": (3.0, 25.0),
    "MCV": (40.0, 130.0),
    "MCH": (10.0, 50.0),
    "RBC": (1.0, 8.0),
}

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
ALLOWED_DATA_EXT  = {".csv", ".xlsx", ".xls"}

# ---- Model globals ----------------------------------------------------------
_model         = None
_label_encoder = None

# ---- Rate limiter -----------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# ---- Lifespan ---------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _label_encoder
    logger.info("Starting up - loading ML artefacts...")
    try:
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded OK: %s", MODEL_PATH)
    except FileNotFoundError:
        logger.error("Model file not found: %s", MODEL_PATH)
    except Exception as exc:
        logger.error("Model load error: %s", exc)

    try:
        _label_encoder = joblib.load(ENCODER_PATH)
        logger.info("Encoder loaded OK: %s", ENCODER_PATH)
    except FileNotFoundError:
        logger.error("Encoder file not found: %s", ENCODER_PATH)
    except Exception as exc:
        logger.error("Encoder load error: %s", exc)

    if _model and _label_encoder:
        logger.info("API ready - all systems nominal")
    else:
        logger.warning("API started but model/encoder missing - predict endpoints will return 503")
    yield
    logger.info("Shutdown complete.")


# ---- App --------------------------------------------------------------------
app = FastAPI(
    title="Thalassemia Diagnosis API",
    description="Predicts thalassemia type from CBC indices.",
    version="3.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---- Middleware: attach request ID to every response -----------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    Attaches a unique X-Request-ID to every response.
    The frontend can log this ID so you can correlate a user-reported error
    with a specific server log line.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---- Global exception handler: no raw tracebacks in production -------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exception and returns a clean JSON error.
    This prevents Python tracebacks from leaking into API responses,
    which is both a security and UX concern.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled exception [request_id=%s]: %s", request_id, exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Please try again.",
            "request_id": request_id,
        },
    )


# ---- Helpers ----------------------------------------------------------------

def require_model():
    if _model is None or _label_encoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not available. Check server logs.",
        )


def run_prediction(hgb: float, mcv: float, mch: float, rbc: float) -> Dict[str, Any]:
    if rbc == 0:
        raise ValueError("RBC cannot be zero.")

    mentzer = mcv / rbc
    shine   = (mcv ** 2 * mch) / 100.0

    features = pd.DataFrame(
        [[mcv, mch, hgb, rbc, mentzer, shine]],
        columns=["MCV", "MCH", "HGB", "RBC", "Mentzer_Index", "Shine_Lal"],
    )

    pred_idx = int(_model.predict(features)[0])

    try:
        proba      = _model.predict_proba(features)[0]
        confidence = float(max(proba))
    except AttributeError:
        confidence = 0.75

    diagnosis: str = _label_encoder.inverse_transform([pred_idx])[0]

    if mcv >= MCV_THRESHOLD and diagnosis != "Normal":
        logger.info("MCV=%.1f overrides '%s' to 'Normal'", mcv, diagnosis)
        diagnosis = "Normal"

    score_map = {
        # Normal / low risk
        "Normal":                   1,
        "Iron Deficiency Anemia":   2,
        # Borderline
        "Borderline":               3,
        # Traits / possible (score 5-6)
        "Alpha Thalassemia Trait":  6,
        "Alpha-Thalassemia Minor":  6,
        "Alpha Thalassemia Minor":  6,
        "Beta Thalassemia Trait":   6,
        "Beta-Thalassemia Minor":   6,
        "Beta Thalassemia Minor":   6,
        "Thalassemia Trait":        6,
        "Possible Thalassemia":     5,
        # Severe (score 7-10)
        "Beta Thalassemia":         8,
        "Beta-Thalassemia Major":   9,
        "Beta Thalassemia Major":   9,
        "Thalassemia Major":        10,
        "Thalassemia Intermedia":   7,
    }
    thalassemia_score = score_map.get(diagnosis, 5)

    rec_map = {
        "Normal":                   "No signs of thalassemia. Routine annual CBC recommended.",
        "Iron Deficiency Anemia":   "Iron supplementation recommended. Recheck CBC in 3 months.",
        "Borderline":               "Borderline results. Repeat CBC in 1 month.",
        "Alpha Thalassemia Trait":  "Alpha thalassemia trait detected. Genetic counselling advised.",
        "Alpha-Thalassemia Minor":  "Alpha thalassemia minor detected. Hemoglobin electrophoresis and genetic counselling recommended.",
        "Alpha Thalassemia Minor":  "Alpha thalassemia minor detected. Hemoglobin electrophoresis and genetic counselling recommended.",
        "Beta Thalassemia Trait":   "Beta thalassemia trait detected. Genetic counselling advised.",
        "Beta-Thalassemia Minor":   "Beta thalassemia minor (trait) detected. Hemoglobin electrophoresis and genetic counselling recommended.",
        "Beta Thalassemia Minor":   "Beta thalassemia minor (trait) detected. Hemoglobin electrophoresis and genetic counselling recommended.",
        "Thalassemia Trait":        "Thalassemia trait detected. Hemoglobin electrophoresis recommended.",
        "Possible Thalassemia":     "Possible thalassemia - consult a hematologist.",
        "Beta Thalassemia":         "Beta thalassemia - specialist evaluation required urgently.",
        "Beta-Thalassemia Major":   "Beta thalassemia major - urgent specialist evaluation and transfusion therapy planning required.",
        "Beta Thalassemia Major":   "Beta thalassemia major - urgent specialist evaluation and transfusion therapy planning required.",
        "Thalassemia Major":        "Thalassemia Major - urgent clinical evaluation required.",
        "Thalassemia Intermedia":   "Thalassemia Intermedia - regular specialist follow-up required.",
    }
    recommendation = rec_map.get(diagnosis, "Abnormal CBC indices detected. Please consult a hematologist for further evaluation.")

    explanation = (
        f"CBC values: HGB={hgb} g/dL, MCV={mcv} fL, MCH={mch} pg, RBC={rbc} x10^12/L. "
        f"Mentzer Index: {mentzer:.2f} "
        f"({'supports thalassemia' if mentzer < 13 else 'supports IDA or normal'}). "
        f"Shine-Lal Index: {shine:.2f}. "
        f"Model confidence: {confidence*100:.0f}%."
    )

    return {
        "prediction":        diagnosis,
        "thalassemia_score": thalassemia_score,
        "confidence":        round(confidence, 3),
        "recommendation":    recommendation,
        "explanation":       explanation,
        "indicators": {
            "mentzer_index": round(mentzer, 2),
            "shine_lal":     round(shine, 2),
        },
    }


def extract_value(name: str, text: str) -> Optional[float]:
    patterns = {
        "HGB": r"H[GgB60b]{2}.*?(\d+\.?\d*)",
        "MCV": r"MC[Vv0uU7].*?(\d+\.?\d*)",
        "MCH": r"MCH.*?(\d+\.?\d*)",
        "RBC": r"RBC.*?(\d+\.?\d*)",
    }
    m = re.search(patterns[name], text, re.IGNORECASE | re.DOTALL)
    return float(m.group(1)) if m else None


def fix_decimal(value: Optional[float], kind: str) -> Optional[float]:
    if value is None:
        return None
    if kind == "HGB" and value > 25:
        return value / 10
    if kind == "RBC" and value > 10:
        return value / 10
    if kind == "MCH" and value > 60:
        return value / 10
    return value


def check_range(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    lo, hi = VALID_RANGES[name]
    if not (lo <= value <= hi):
        logger.warning("OCR %s=%.2f outside valid range, discarding.", name, value)
        return None
    return value


def run_ocr(image_bytes: bytes) -> Dict[str, Optional[float]]:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid PNG or JPEG.")

    # FIX: Guard against memory exhaustion from enormous images.
    # A 20000x20000 image resized 2.5x = 50000x50000 = 2.5 GB RAM — instant OOM crash.
    h, w = img.shape[:2]
    if h > MAX_IMAGE_DIM or w > MAX_IMAGE_DIM:
        raise ValueError(
            f"Image dimensions {w}x{h} exceed the {MAX_IMAGE_DIM}px limit. "
            "Please resize the image and try again."
        )

    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray   = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    _, th  = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    cfg    = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    text   = pytesseract.image_to_string(th, config=cfg)
    logger.debug("OCR text (first 300): %s", text[:300])

    hgb = fix_decimal(extract_value("HGB", text), "HGB")
    mcv = extract_value("MCV", text)
    mch = fix_decimal(extract_value("MCH", text), "MCH")
    rbc = fix_decimal(extract_value("RBC", text), "RBC")

    if hgb is None:
        m2 = re.search(r"Hemoglobin.*?(\d+\.?\d*)", text, re.IGNORECASE | re.DOTALL)
        if m2:
            hgb = fix_decimal(float(m2.group(1)), "HGB")

    if mcv is None:
        cands = [float(n) for n in re.findall(r"(\d{2,3}\.?\d?)", text)]
        mcv = next((n for n in cands if 50 <= n <= 115), None)

    return {
        "HGB": check_range(hgb, "HGB"),
        "MCV": check_range(mcv, "MCV"),
        "MCH": check_range(mch, "MCH"),
        "RBC": check_range(rbc, "RBC"),
    }


# ---- Schemas ----------------------------------------------------------------

class PatientData(BaseModel):
    hgb: float = Field(..., gt=0, le=25,  description="Hemoglobin g/dL",                examples=[13.5])
    mcv: float = Field(..., gt=0, le=130, description="Mean Corpuscular Volume fL",      examples=[72.0])
    mch: float = Field(..., gt=0, le=50,  description="Mean Corpuscular Hemoglobin pg",  examples=[24.0])
    rbc: float = Field(..., gt=0, le=8,   description="Red Blood Cell count x10^12/L",   examples=[5.1])

    @field_validator("rbc")
    @classmethod
    def rbc_nonzero(cls, v: float) -> float:
        if v == 0:
            raise ValueError("RBC must be greater than 0")
        return v


class PredictionResponse(BaseModel):
    prediction:        str
    thalassemia_score: int
    confidence:        float
    recommendation:    str
    explanation:       str
    indicators:        Dict[str, float]


class HealthResponse(BaseModel):
    status:         str
    model_loaded:   bool
    encoder_loaded: bool
    version:        str


# ---- Routes -----------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check — used by Railway, Docker, and monitoring tools."""
    return HealthResponse(
        status="ok" if (_model and _label_encoder) else "degraded",
        model_loaded=_model is not None,
        encoder_loaded=_label_encoder is not None,
        version="3.1.0",
    )


@app.post("/predict/manual", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("10/minute")
async def predict_manual(request: Request, data: PatientData):
    """Predict thalassemia from manually entered CBC values."""
    require_model()
    try:
        return run_prediction(data.hgb, data.mcv, data.mch, data.rbc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Error in predict_manual")
        raise HTTPException(status_code=500, detail="Internal prediction error.")


@app.post("/predict/file", response_model=None, tags=["Prediction"])
@limiter.limit("5/minute")
async def predict_file(request: Request, file: UploadFile = File(...)):
    """Batch predict from a CSV or Excel file (columns: HGB, MCV, MCH, RBC)."""
    require_model()

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_DATA_EXT:
        raise HTTPException(status_code=415, detail=f"Unsupported type '{ext}'. Use: {sorted(ALLOWED_DATA_EXT)}")

    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    try:
        df = pd.read_csv(io.BytesIO(contents)) if ext == ".csv" else pd.read_excel(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    df.columns = df.columns.str.upper().str.strip()
    missing = {"HGB", "MCV", "MCH", "RBC"} - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {sorted(missing)}")

    results = []
    for i, row in df.iterrows():
        name = str(row.get("NAME", f"Patient {i+1}"))
        try:
            res = run_prediction(float(row["HGB"]), float(row["MCV"]), float(row["MCH"]), float(row["RBC"]))
            results.append({"patient": name, "result": res})
        except Exception as exc:
            results.append({"patient": name, "result": {"prediction": f"ERROR: {exc}"}})

    logger.info("Batch prediction: %d rows processed", len(results))
    return results


@app.post("/predict/image", response_model=None, tags=["Prediction"])
@limiter.limit("3/minute")
async def predict_image(request: Request, file: UploadFile = File(...)):
    """Extract CBC values from a lab report image via OCR, then predict."""
    require_model()

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise HTTPException(status_code=415, detail=f"Unsupported type '{ext}'. Use: {sorted(ALLOWED_IMAGE_EXT)}")

    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    try:
        extracted = run_ocr(contents)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("OCR failed")
        raise HTTPException(status_code=500, detail="Image processing failed.")

    hgb, mcv, mch, rbc = extracted["HGB"], extracted["MCV"], extracted["MCH"], extracted["RBC"]
    missing_fields = [k for k, v in {"HGB": hgb, "MCV": mcv, "MCH": mch, "RBC": rbc}.items() if v is None]

    if missing_fields:
        return {
            "status":         "partial",
            "message":        f"Could not extract: {missing_fields}. Try manual entry instead.",
            "detected":       {k: v for k, v in extracted.items() if v is not None},
            "missing_fields": missing_fields,
        }

    try:
        result = run_prediction(hgb, mcv, mch, rbc)
    except Exception:
        logger.exception("Prediction after OCR failed")
        raise HTTPException(status_code=500, detail="Prediction failed after image extraction.")

    logger.info("Image prediction: %s (confidence %.2f)", result["prediction"], result["confidence"])
    return {
        "status":           "success",
        "diagnosis":        result["prediction"],
        "result":           result,
        "extracted_values": {"HGB": hgb, "MCV": mcv, "MCH": mch, "RBC": rbc},
    }


# ---- Entry point (local dev only) ------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level="info",
    )
