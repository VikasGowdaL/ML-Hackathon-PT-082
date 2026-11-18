
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathsnap.data.loader import discover_trips, load_trip
from pathsnap.utils.config import get_path, get_raw_data_dir

app = FastAPI(title="PathSnap API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
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
