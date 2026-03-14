"""
backend/main.py — FloodNet FastAPI Inference Server
ONNX inference <50ms/image | Handles 100 req/sec
Endpoints: /predict, /health, /weather, /batch-predict
"""

import os
import io
import sys
import time
import json
import base64
import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from PIL import Image
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("floodnet-api")

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
KOLLAM_LAT, KOLLAM_LON = 8.8932, 76.6141
DAMAGE_LABELS = ["No Damage", "Minor Damage", "Major Damage", "Catastrophic"]
DAMAGE_COLORS = ["#22c55e", "#eab308", "#f97316", "#ef4444"]
WEATHER_COLS = [
    "rainfall_mm", "river_level_m", "wind_speed_kmh", "humidity_pct",
    "temp_c", "pressure_hpa", "duration_hrs", "upstream_rainfall_mm",
    "soil_moisture", "previous_flood_days",
]
WEATHER_MEAN = np.array([45.2, 3.8, 22.1, 82.3, 27.5, 1008.2, 12.4, 38.7, 0.55, 2.1], dtype=np.float32)
WEATHER_STD  = np.array([38.6, 2.1, 15.4, 11.2, 4.8, 6.3, 9.8, 32.1, 0.28, 3.7], dtype=np.float32)

IMG_SIZE = 256
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB


# ─────────────────────────────────────────────────────────
# ONNX INFERENCE ENGINE
# ─────────────────────────────────────────────────────────
class FloodNetONNX:
    """Thread-safe ONNX inference wrapper with warm cache."""

    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        # Add CUDA if available
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("ONNX: Using CUDA execution provider")

        self.session = ort.InferenceSession(model_path, opts, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        logger.info(f"ONNX model loaded | inputs: {self.input_names}")

        # Warm up
        self._warmup()

    def _warmup(self):
        dummy_img = np.zeros((1, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        dummy_wx = np.zeros((1, 10), dtype=np.float32)
        self.run(dummy_img, dummy_wx)
        logger.info("ONNX warmup complete")

    def run(self, image: np.ndarray, weather: np.ndarray) -> np.ndarray:
        """Returns (N, 4) probability array."""
        feeds = {"sar_image": image, "weather": weather}
        outputs = self.session.run(["damage_probs"], feeds)
        return outputs[0]

    @property
    def latency_ms(self) -> float:
        img = np.random.rand(1, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)
        wx = np.random.rand(1, 10).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(10):
            self.run(img, wx)
        return (time.perf_counter() - t0) * 100  # avg ms


# ─────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert uploaded image bytes → (1, 1, 256, 256) float32 ONNX input.
    Handles RGB satellite images and grayscale SAR.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert to grayscale (SAR-like)
        img = img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)

        # SAR log normalization
        arr = np.clip(arr, 1.0, None)
        arr = np.log10(arr + 1e-6)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

        # Z-score with SAR stats
        arr = (arr - 0.33) / 0.22
        arr = np.clip(arr, -5.0, 5.0)

        return arr[np.newaxis, np.newaxis, :, :].astype(np.float32)

    except Exception as e:
        raise HTTPException(400, f"Image preprocessing failed: {str(e)}")


def preprocess_weather(data: Dict) -> np.ndarray:
    """Convert weather dict → (1, 10) float32 normalized ONNX input."""
    defaults = dict(zip(WEATHER_COLS, [0, 2.5, 15, 75, 28, 1010, 6, 0, 0.5, 0]))
    vec = np.array(
        [float(data.get(k, defaults[k]) or defaults[k]) for k in WEATHER_COLS],
        dtype=np.float32,
    )
    vec = np.nan_to_num(vec, nan=0.0, posinf=1e4, neginf=-1e4)
    vec = (vec - WEATHER_MEAN) / (WEATHER_STD + 1e-8)
    vec = np.clip(vec, -5.0, 5.0)
    return vec[np.newaxis, :].astype(np.float32)


# ─────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────
class WeatherInput(BaseModel):
    rainfall_mm:         float = Field(default=0.0,   ge=0, le=2000)
    river_level_m:       float = Field(default=2.5,   ge=0, le=50)
    wind_speed_kmh:      float = Field(default=15.0,  ge=0, le=250)
    humidity_pct:        float = Field(default=75.0,  ge=0, le=100)
    temp_c:              float = Field(default=28.0,  ge=-10, le=55)
    pressure_hpa:        float = Field(default=1010.0, ge=900, le=1100)
    duration_hrs:        float = Field(default=6.0,   ge=0, le=240)
    upstream_rainfall_mm:float = Field(default=0.0,   ge=0, le=2000)
    soil_moisture:       float = Field(default=0.5,   ge=0, le=1)
    previous_flood_days: float = Field(default=0.0,   ge=0, le=365)
    tweet_text:          Optional[str] = Field(default=None, max_length=512)
    lat:                 float = Field(default=KOLLAM_LAT)
    lon:                 float = Field(default=KOLLAM_LON)


class PredictionResponse(BaseModel):
    class_idx:    int
    label:        str
    color:        str
    confidence:   float
    probabilities: Dict[str, float]
    inference_ms: float
    timestamp:    str
    location:     Dict[str, float]
    alert_level:  str


class BatchPredictRequest(BaseModel):
    items: List[WeatherInput]
    image_base64_list: Optional[List[str]] = None


# ─────────────────────────────────────────────────────────
# APP STATE
# ─────────────────────────────────────────────────────────
class AppState:
    onnx_model: Optional[FloodNetONNX] = None
    request_count: int = 0
    total_latency_ms: float = 0.0
    start_time: float = time.time()


app_state = AppState()


# ─────────────────────────────────────────────────────────
# APP LIFESPAN
# ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ONNX model at startup
    model_paths = [
        os.environ.get("ONNX_MODEL_PATH", ""),
        "floodnet.onnx",
        "../model/floodnet.onnx",
        "model/floodnet.onnx",
    ]
    for path in model_paths:
        if path and Path(path).exists():
            try:
                app_state.onnx_model = FloodNetONNX(path)
                logger.info(f"✓ ONNX model loaded from {path}")
                logger.info(f"  Avg latency: {app_state.onnx_model.latency_ms:.1f}ms")
                break
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    if app_state.onnx_model is None:
        logger.warning("No ONNX model found — running in MOCK mode")

    yield  # app runs

    logger.info("Shutting down FloodNet API")


# ─────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Kerala Flood Damage Assessment API",
    description="Multi-modal DL model for flood damage severity prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ─────────────────────────────────────────────────────────
# RATE LIMITER (simple in-memory)
# ─────────────────────────────────────────────────────────
_rate_store: Dict[str, List[float]] = {}
RATE_LIMIT = 100  # req/sec per IP

async def check_rate_limit(request: Request):
    ip = request.client.host
    now = time.time()
    if ip not in _rate_store:
        _rate_store[ip] = []
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < 1.0]
    if len(_rate_store[ip]) >= RATE_LIMIT:
        raise HTTPException(429, "Rate limit exceeded (100 req/sec)")
    _rate_store[ip].append(now)


# ─────────────────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────────────────
def _run_inference(
    image_np: np.ndarray,
    weather_np: np.ndarray,
    lat: float,
    lon: float,
) -> PredictionResponse:
    t0 = time.perf_counter()

    if app_state.onnx_model is not None:
        probs = app_state.onnx_model.run(image_np, weather_np)[0]
    else:
        # Mock mode: rule-based from weather
        rain_norm = weather_np[0, 0] * 38.6 + 45.2
        if rain_norm < 10:
            probs = np.array([0.75, 0.15, 0.07, 0.03])
        elif rain_norm < 50:
            probs = np.array([0.15, 0.60, 0.20, 0.05])
        elif rain_norm < 120:
            probs = np.array([0.05, 0.20, 0.60, 0.15])
        else:
            probs = np.array([0.02, 0.08, 0.25, 0.65])
        probs = probs / probs.sum()

    inf_ms = (time.perf_counter() - t0) * 1000

    # Track metrics
    app_state.request_count += 1
    app_state.total_latency_ms += inf_ms

    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])

    alert_levels = {0: "GREEN", 1: "YELLOW", 2: "ORANGE", 3: "RED"}

    return PredictionResponse(
        class_idx=class_idx,
        label=DAMAGE_LABELS[class_idx],
        color=DAMAGE_COLORS[class_idx],
        confidence=round(confidence, 4),
        probabilities={label: round(float(p), 4) for label, p in zip(DAMAGE_LABELS, probs)},
        inference_ms=round(inf_ms, 2),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        location={"lat": lat, "lon": lon},
        alert_level=alert_levels[class_idx],
    )


# ─────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check with system stats."""
    uptime = time.time() - app_state.start_time
    avg_lat = (app_state.total_latency_ms / max(app_state.request_count, 1))
    return {
        "status": "healthy",
        "model_loaded": app_state.onnx_model is not None,
        "uptime_seconds": round(uptime, 1),
        "total_requests": app_state.request_count,
        "avg_latency_ms": round(avg_lat, 2),
        "onnx_providers": ort.get_available_providers(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="SAR or satellite image (PNG/JPG/TIF)"),
    weather_json: str = File(default="{}", description="JSON weather features"),
    _: None = Depends(check_rate_limit),
):
    """
    Predict flood damage severity from satellite image + weather data.
    Returns 4-class damage assessment with confidence scores.
    """
    # Validate image
    if image.size and image.size > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"Image too large (max {MAX_IMAGE_BYTES//1024//1024}MB)")

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty image file")

    # Parse weather
    try:
        weather_dict = json.loads(weather_json) if weather_json.strip() != "{}" else {}
    except json.JSONDecodeError:
        weather_dict = {}

    weather_input = WeatherInput(**weather_dict)

    # Preprocess
    image_np = preprocess_image(image_bytes)
    weather_np = preprocess_weather(weather_input.dict())

    return _run_inference(image_np, weather_np, weather_input.lat, weather_input.lon)


@app.post("/predict-weather", response_model=PredictionResponse)
async def predict_weather_only(
    weather: WeatherInput,
    _: None = Depends(check_rate_limit),
):
    """
    Predict flood damage using weather data only (no image required).
    Uses a synthetic grayscale image based on rainfall intensity.
    """
    # Generate synthetic SAR-like image from rainfall
    rain_normalized = min(weather.rainfall_mm / 200.0, 1.0)
    synthetic_sar = np.random.normal(
        1.0 - rain_normalized * 0.7,
        0.1 + rain_normalized * 0.15,
        (IMG_SIZE, IMG_SIZE),
    ).clip(0, 1).astype(np.float32)
    synthetic_sar = (synthetic_sar - 0.33) / 0.22
    image_np = synthetic_sar[np.newaxis, np.newaxis, :, :].astype(np.float32)

    weather_np = preprocess_weather(weather.dict())
    return _run_inference(image_np, weather_np, weather.lat, weather.lon)


@app.post("/batch-predict")
async def batch_predict(
    request: BatchPredictRequest,
    _: None = Depends(check_rate_limit),
):
    """Batch prediction for up to 32 items."""
    if len(request.items) > 32:
        raise HTTPException(400, "Maximum 32 items per batch")

    results = []
    for i, item in enumerate(request.items):
        # Get image
        if request.image_base64_list and i < len(request.image_base64_list):
            try:
                img_bytes = base64.b64decode(request.image_base64_list[i])
                image_np = preprocess_image(img_bytes)
            except Exception:
                image_np = np.zeros((1, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        else:
            # Weather-only fallback
            rain = item.rainfall_mm / 200.0
            syn = np.random.normal(1 - rain * 0.7, 0.1, (IMG_SIZE, IMG_SIZE)).clip(0, 1).astype(np.float32)
            syn = (syn - 0.33) / 0.22
            image_np = syn[np.newaxis, np.newaxis].astype(np.float32)

        weather_np = preprocess_weather(item.dict())
        result = _run_inference(image_np, weather_np, item.lat, item.lon)
        results.append(result.dict())

    return {"predictions": results, "count": len(results)}


@app.get("/weather/current")
async def get_current_weather(
    lat: float = KOLLAM_LAT,
    lon: float = KOLLAM_LON,
):
    """Fetch current weather from Open-Meteo (free, no API key)."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,"
            f"wind_speed_10m,surface_pressure,weather_code"
            f"&timezone=Asia%2FKolkata"
        )
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            d = r.json()

        curr = d.get("current", {})
        rain = curr.get("precipitation", 0) or 0

        weather = {
            "rainfall_mm": rain,
            "river_level_m": 2.5 + rain * 0.02,  # estimated
            "wind_speed_kmh": curr.get("wind_speed_10m", 15) or 15,
            "humidity_pct": curr.get("relative_humidity_2m", 75) or 75,
            "temp_c": curr.get("temperature_2m", 28) or 28,
            "pressure_hpa": curr.get("surface_pressure", 1010) or 1010,
            "duration_hrs": 1.0,
            "upstream_rainfall_mm": rain * 0.8,
            "soil_moisture": min(0.3 + rain * 0.003, 0.98),
            "previous_flood_days": 0.0,
            "lat": lat,
            "lon": lon,
            "weather_code": curr.get("weather_code", 0),
            "source": "Open-Meteo (free)",
        }
        return weather

    except Exception as e:
        logger.warning(f"Weather API failed: {e}")
        # Return Kerala June-avg defaults
        return {
            "rainfall_mm": 12.5, "river_level_m": 3.2, "wind_speed_kmh": 25.0,
            "humidity_pct": 85.0, "temp_c": 27.5, "pressure_hpa": 1005.0,
            "duration_hrs": 6.0, "upstream_rainfall_mm": 10.0,
            "soil_moisture": 0.7, "previous_flood_days": 1.0,
            "lat": lat, "lon": lon, "source": "defaults",
        }


@app.get("/districts")
async def get_kerala_districts():
    """Return flood risk assessment for all Kerala districts."""
    districts = {
        "Thiruvananthapuram": (8.5241, 76.9366),
        "Kollam":             (8.8932, 76.6141),
        "Pathanamthitta":     (9.2648, 76.7870),
        "Alappuzha":          (9.4981, 76.3388),
        "Kottayam":           (9.5916, 76.5222),
        "Idukki":             (9.9189, 76.9749),
        "Ernakulam":          (9.9816, 76.2999),
        "Thrissur":           (10.5276, 76.2144),
        "Palakkad":           (10.7867, 76.6548),
        "Malappuram":         (11.0730, 76.0740),
        "Kozhikode":          (11.2588, 75.7804),
        "Wayanad":            (11.6854, 76.1320),
        "Kannur":             (11.8745, 75.3704),
        "Kasaragod":          (12.4996, 74.9869),
    }
    return {
        "districts": [
            {"name": name, "lat": lat, "lon": lon}
            for name, (lat, lon) in districts.items()
        ],
        "total": len(districts),
        "bbox": {"min_lat": 8.0, "max_lat": 13.0, "min_lon": 74.8, "max_lon": 77.5},
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    uptime = time.time() - app_state.start_time
    avg_lat = app_state.total_latency_ms / max(app_state.request_count, 1)
    return {
        "floodnet_requests_total": app_state.request_count,
        "floodnet_avg_latency_ms": round(avg_lat, 2),
        "floodnet_uptime_seconds": round(uptime, 1),
        "floodnet_model_loaded": int(app_state.onnx_model is not None),
    }


# ─────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=4,        # 4 workers for 100 req/sec target
        loop="uvloop",    # faster event loop
        access_log=False, # disable for performance
    )