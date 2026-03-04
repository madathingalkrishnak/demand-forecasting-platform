import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import xgboost as xgb
import redis
import json
import time
import os
from datetime import datetime, timedelta

app = FastAPI(
    title="Demand Forecasting API",
    description="Real-time order demand predictions for delivery zones",
    version="1.0.0"
)

# ── Load model at startup ─────────────────────────────────────────────────────
MODEL_PATH = "models/xgb_global.json"
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

# ── Redis cache ───────────────────────────────────────────────────────────────
try:
    cache = redis.Redis(host='localhost', port=6380, db=0, decode_responses=True)
    cache.ping()
    CACHE_AVAILABLE = True
    print("✅ Redis cache connected")
except:
    CACHE_AVAILABLE = False
    print("⚠️  Redis unavailable, running without cache")

CACHE_TTL = 300  # 5 minutes

# ── Zone metadata (in production this would come from a DB) ───────────────────
ZONE_TYPES = {f"zone_{i:03d}": t for i, t in enumerate(
    ['suburb','residential','downtown','suburb','university',
     'university','downtown','suburb','university','residential',
     'residential','downtown','residential','residential','university',
     'residential','downtown','residential','downtown','downtown']
)}

ZONE_DEMAND_PERCENTILES = {
    f"zone_{i:03d}": round(0.1 + (i / 20) * 0.85, 2) for i in range(20)
}

# ── Request/Response schemas ──────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    zone_id: str
    timestamp: Optional[str] = None  # ISO format, defaults to now
    temperature: Optional[float] = 65.0
    precipitation: Optional[float] = 0.0
    is_raining: Optional[float] = 0.0
    wind_speed: Optional[float] = 8.0
    # Lag features — in production these come from feature store
    lag_1h: Optional[float] = None
    lag_24h: Optional[float] = None
    lag_168h: Optional[float] = None
    same_hour_last_week: Optional[float] = None

class PredictionResponse(BaseModel):
    zone_id: str
    timestamp: str
    predicted_orders: float
    confidence_low: float
    confidence_high: float
    zone_type: str
    cached: bool
    latency_ms: float

class BatchRequest(BaseModel):
    zones: List[str]
    timestamp: Optional[str] = None
    temperature: Optional[float] = 65.0
    is_raining: Optional[float] = 0.0

# ── Feature builder ───────────────────────────────────────────────────────────
def build_feature_vector(zone_id: str, ts: datetime, req) -> dict:
    """Build feature vector for a single zone+timestamp"""
    zone_type = ZONE_TYPES.get(zone_id, 'residential')

    features = {
        'hour':               ts.hour,
        'dayofweek':          ts.weekday(),
        'is_weekend':         int(ts.weekday() >= 5),
        'month':              ts.month,
        'is_lunch_rush':      int(11 <= ts.hour <= 14),
        'is_dinner_rush':     int(17 <= ts.hour <= 21),
        'is_late_night':      int(ts.hour in [22, 23, 0, 1]),
        'hour_sin':           np.sin(2 * np.pi * ts.hour / 24),
        'hour_cos':           np.cos(2 * np.pi * ts.hour / 24),
        'dow_sin':            np.sin(2 * np.pi * ts.weekday() / 7),
        'dow_cos':            np.cos(2 * np.pi * ts.weekday() / 7),
        # Lag features — use provided or reasonable defaults
        'lag_1h':             req.lag_1h or 8.0,
        'lag_2h':             8.0,
        'lag_3h':             8.0,
        'lag_6h':             8.0,
        'lag_12h':            8.0,
        'lag_24h':            req.lag_24h or 8.0,
        'lag_48h':            8.0,
        'lag_168h':           req.lag_168h or 8.0,
        'rolling_mean_3h':    req.lag_1h or 8.0,
        'rolling_mean_6h':    8.0,
        'rolling_mean_24h':   8.0,
        'same_hour_last_week': req.same_hour_last_week or 8.0,
        # Weather
        'temperature':        req.temperature,
        'precipitation':      req.precipitation,
        'is_raining':         req.is_raining,
        'wind_speed':         req.wind_speed,
        # Zone features
        'zone_demand_percentile': ZONE_DEMAND_PERCENTILES.get(zone_id, 0.5),
        'zone_type_downtown':    int(zone_type == 'downtown'),
        'zone_type_residential': int(zone_type == 'residential'),
        'zone_type_suburb':      int(zone_type == 'suburb'),
        'zone_type_university':  int(zone_type == 'university'),
    }
    return features

def predict_single(zone_id: str, ts: datetime, req) -> tuple[float, float, float]:
    """Returns (prediction, low, high)"""
    features = build_feature_vector(zone_id, ts, req)
    X = pd.DataFrame([features])
    pred = float(max(0, model.predict(X)[0]))
    # Simple confidence interval (±20% — in production use quantile regression)
    return pred, pred * 0.80, pred * 1.20

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "cache": CACHE_AVAILABLE
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    start = time.time()

    # Validate zone
    if req.zone_id not in ZONE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown zone_id: {req.zone_id}")

    # Parse timestamp
    ts = datetime.fromisoformat(req.timestamp) if req.timestamp else datetime.now()

    # Check cache
    cache_key = f"pred:{req.zone_id}:{ts.strftime('%Y%m%d%H')}"
    if CACHE_AVAILABLE:
        cached = cache.get(cache_key)
        if cached:
            result = json.loads(cached)
            result['cached'] = True
            result['latency_ms'] = round((time.time() - start) * 1000, 2)
            return result

    # Predict
    pred, low, high = predict_single(req.zone_id, ts, req)

    result = {
        "zone_id":          req.zone_id,
        "timestamp":        ts.isoformat(),
        "predicted_orders": round(pred, 2),
        "confidence_low":   round(low, 2),
        "confidence_high":  round(high, 2),
        "zone_type":        ZONE_TYPES[req.zone_id],
        "cached":           False,
        "latency_ms":       round((time.time() - start) * 1000, 2)
    }

    # Store in cache
    if CACHE_AVAILABLE:
        cache.setex(cache_key, CACHE_TTL, json.dumps(result))

    return result

@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    start = time.time()

    invalid = [z for z in req.zones if z not in ZONE_TYPES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown zones: {invalid}")

    ts = datetime.fromisoformat(req.timestamp) if req.timestamp else datetime.now()

    class MockReq:
        def __init__(self):
            self.temperature = req.temperature
            self.precipitation = 0.0
            self.is_raining = req.is_raining
            self.wind_speed = 8.0
            self.lag_1h = None
            self.lag_24h = None
            self.lag_168h = None
            self.same_hour_last_week = None

    mock_req = MockReq()
    results = []
    for zone_id in req.zones:
        pred, low, high = predict_single(zone_id, ts, mock_req)
        results.append({
            "zone_id":          zone_id,
            "zone_type":        ZONE_TYPES[zone_id],
            "predicted_orders": round(pred, 2),
            "confidence_low":   round(low, 2),
            "confidence_high":  round(high, 2),
        })

    results.sort(key=lambda x: x['predicted_orders'], reverse=True)

    return {
        "timestamp":              ts.isoformat(),
        "predictions":            results,
        "total_zones":            len(results),
        "total_predicted_orders": round(sum(r['predicted_orders'] for r in results), 1),
        "latency_ms":             round((time.time() - start) * 1000, 2)
    }

@app.get("/predict/next4hours/{zone_id}")
def predict_next_4_hours(zone_id: str, temperature: float = 65.0, is_raining: float = 0.0):
    """Predict demand for next 4 hours — key for driver pre-positioning"""
    if zone_id not in ZONE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown zone_id: {zone_id}")

    start = time.time()
    now = datetime.now().replace(minute=0, second=0, microsecond=0)

    class MockReq:
        def __init__(self):
            self.temperature = temperature
            self.precipitation = 0.0
            self.is_raining = is_raining
            self.wind_speed = 8.0
            self.lag_1h = None
            self.lag_24h = None
            self.lag_168h = None
            self.same_hour_last_week = None

    mock_req = MockReq()
    predictions = []
    for i in range(4):
        ts = now + timedelta(hours=i)
        pred, low, high = predict_single(zone_id, ts, mock_req)
        predictions.append({
            "hour":             ts.strftime("%H:00"),
            "timestamp":        ts.isoformat(),
            "predicted_orders": round(pred, 2),
            "confidence_low":   round(low, 2),
            "confidence_high":  round(high, 2),
        })

    return {
        "zone_id":     zone_id,
        "zone_type":   ZONE_TYPES[zone_id],
        "forecast":    predictions,
        "latency_ms":  round((time.time() - start) * 1000, 2)
    }