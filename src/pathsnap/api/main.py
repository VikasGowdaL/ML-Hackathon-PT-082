
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

ALLOWED_PARTITIONS = {"train", "student_test"}


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
    return {"status": "ok", "service": "PathSnap API"}


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
