
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from pathsnap import pipeline
from pathsnap.data.loader import discover_trips, load_trip
from pathsnap.utils.config import get_path, get_raw_data_dir

app = FastAPI(title="PathSnap API", version="0.1.0")

# --- CORS (two-laptop deployment) -------------------------------------------------------------
# The UI may run on another machine (Laptop 2) and call this API on Laptop 1 over the LAN.
# Default: allow the local dev UI plus any origin on a private network address (192.168.x.x,
# 10.x.x.x, 172.16-31.x.x) on any port. Override with PATHSNAP_CORS_ORIGINS, a comma-separated list
# of exact origins (e.g. "http://192.168.1.20:5173"), or "*" to allow every origin.
LOCAL_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
PRIVATE_LAN_ORIGIN_REGEX = (
    r"^https?://(localhost|127\.0\.0\.1|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?$"
)


def cors_settings() -> dict:
    """CORS options from the PATHSNAP_CORS_ORIGINS environment variable (see module comment)."""
    raw = os.environ.get("PATHSNAP_CORS_ORIGINS", "").strip()
    if raw == "*":
        return {"allow_origins": ["*"], "allow_origin_regex": None}
    if raw:
        return {"allow_origins": [o.strip().rstrip("/") for o in raw.split(",") if o.strip()], "allow_origin_regex": None}
    return {"allow_origins": LOCAL_ORIGINS, "allow_origin_regex": PRIVATE_LAN_ORIGIN_REGEX}


app.add_middleware(
    CORSMiddleware,
    **cors_settings(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Partition shown as "evaluation trips" in the UI. Default: the 10 reserved evaluator trips. For a rehearsal
# without evaluator data set PATHSNAP_EVAL_PARTITION=student_test.
EVAL_PARTITION = os.environ.get("PATHSNAP_EVAL_PARTITION", "evaluator_test")
# GPS display (/api/trips/{id}) is allowed for these partitions; it never returns a route.
ALLOWED_PARTITIONS = {"train", "student_test", EVAL_PARTITION}


def available_trips():
    split_path = get_path("split_file", "configs/split.json")

    with open(split_path, encoding="utf-8") as file:
        splits = json.load(file)

    discovered = discover_trips(get_raw_data_dir())

    available = {}

    for partition in ALLOWED_PARTITIONS:
        for trip_id in splits.get(partition, []):
            trip_id = str(trip_id)

            if trip_id in discovered:
                available[trip_id] = partition

    return available, discovered


@app.get("/api/health")
def health():
    available, _ = available_trips()
    return {
        "status": "ok",
        "service": "PathSnap API",
        "model_available": pipeline.model_path().exists(),
        "evaluation_partition": EVAL_PARTITION,
        "evaluation_trips": sum(1 for p in available.values() if p == EVAL_PARTITION),
    }


@app.get("/api/trips")
def list_trips():
    available, _ = available_trips()

    return {
        "count": len(available),
        "trips": [
            {"trip_id": trip_id, "partition": partition}
            for trip_id, partition in sorted(available.items())
        ],
    }


@app.get("/api/trips/{trip_id}")
def get_trip(trip_id: str):
    available, discovered = available_trips()

    if trip_id not in available:
        raise HTTPException(
            status_code=404,
            detail="Trip not found in available train/student-test data",
        )

    try:
        trip = load_trip(
            trip_id=trip_id,
            trips_index=discovered,
            partition=available[trip_id],
            project_utm=False,
            use_cache=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    points = [
        {
            "lat": float(row.lat),
            "lon": float(row.lon),
            "timestamp": float(row.t),
        }
        for row in trip.track.itertuples(index=False)
    ]

    return {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "gps_count": len(points),
        "nodes_count": len(trip.nodes),
        "arcs_count": len(trip.arcs),
        "gps_points": points,
    }


# ---------------------------------------------------------------------------------------------
# Evaluation trips and route prediction (final evaluation protocol)
# ---------------------------------------------------------------------------------------------
class RouteMetrics(BaseModel):
    route_f1: float
    route_precision: float
    route_recall: float
    arc_precision: float
    arc_recall: float
    points_on_route: float
    points_on_route_ml_only: float
    predicted_route_km: float
    reference_route_km: float
    route_sections: int
    unbridged_breaks: int
    points_without_candidates: int


class PredictionResponse(BaseModel):
    trip_id: str
    partition: str
    gps_count: int
    gps_points: List[List[float]]                 # [[lat, lon], ...] in track order
    predicted_route: List[List[List[float]]]      # route sections, each [[lat, lon], ...]
    predicted_arc_count: int
    metrics: Optional[RouteMetrics]               # null if no reference route is available
    timing_s: Dict[str, float]
    model: Dict[str, Any]
    cached: bool


_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_SIZE = 32
_LOCK = threading.Lock()


def _gps_count(path) -> int:
    with open(path, "rb") as fh:
        return sum(1 for line in fh if line.strip())


@app.get("/api/eval/trips")
def list_evaluation_trips():
    """Trips of the evaluation partition (default evaluator_test) found on this machine."""
    available, discovered = available_trips()
    trips = [
        {"trip_id": t, "partition": p, "gps_count": _gps_count(discovered[t]["track"])}
        for t, p in sorted(available.items()) if p == EVAL_PARTITION
    ]
    return {"partition": EVAL_PARTITION, "count": len(trips), "trips": trips}


@app.post("/api/predict/{trip_id}", response_model=PredictionResponse)
def predict(trip_id: str):
    """Run the full pipeline on one trip, then score the predicted route against the reference route.

    The prediction never reads the reference route (see pathsnap.pipeline); the route is loaded afterwards,
    only to compute the metrics.
    """
    available, _ = available_trips()
    if trip_id not in available:
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' is not available on this server")
    partition = available[trip_id]
    with _LOCK:
        if trip_id in _CACHE:
            _CACHE.move_to_end(trip_id)
            return {**_CACHE[trip_id], "cached": True}
    if not pipeline.model_path().exists():
        raise HTTPException(status_code=503, detail="Model not found. Run: python scripts/export_final_model.py")
    try:
        pipeline.refresh_trips_index()
        pred = pipeline.predict_trip(trip_id)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    t0 = time.perf_counter()
    try:
        metrics = pipeline.score_prediction(pred, pipeline.load_reference_route(trip_id, partition))
    except (FileNotFoundError, ValueError):
        metrics = None  # no reference route on this machine: return the prediction without scores
    timing = {**{k: round(v, 3) for k, v in pred.timing_s.items()}, "scoring_s": round(time.perf_counter() - t0, 3)}

    info = pipeline.model_info()
    body = {
        "trip_id": trip_id,
        "partition": partition,
        "gps_count": pred.n_observations,
        "gps_points": pred.observations[["lat", "lon"]].round(6).to_numpy().tolist(),
        "predicted_route": pipeline.route_polylines(pred),
        "predicted_arc_count": len(pred.route_arcs),
        "metrics": metrics,
        "timing_s": timing,
        "model": {"name": info.get("name", "final_model"), "estimator": info.get("estimator", ""),
                  "decoder": "Viterbi (hmm/viterbi.py) + shortest-path reconstruction"},
    }
    with _LOCK:
        _CACHE[trip_id] = body
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return {**body, "cached": False}
