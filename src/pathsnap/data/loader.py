"""Data loading, caching, UTM projection, and validation module for PathSnap.

Supports:
- Raw trip file discovery across flat or nested folder layouts.
- Reading .track, .nodes, .arcs, .route per the data specification.
- Validation of columns, data types, and row indices.
- Metric UTM coordinate projection via pyproj based on the dataset bounding box.
- Parquet/pickle disk caching in data/cache (gitignored).
- Evaluator partition leakage prevention guardrails.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
try:
    import pyproj
except ImportError:
    pyproj = None

from pathsnap.utils.config import get_path, get_raw_data_dir

logger = logging.getLogger("pathsnap.data.loader")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class TripData:
    """Dataclass holding all components of a single GPS map-matching trip."""

    trip_id: str
    partition: Optional[str]
    track: pd.DataFrame  # columns: lon, lat, t (float64, float64, float64)
    nodes: pd.DataFrame  # columns: node_id, lon, lat (int64, float64, float64)
    arcs: pd.DataFrame  # columns: arc_id, from_node_id, to_node_id (int64, int64, int64)
    route: Optional[np.ndarray] = None  # 1D int array of ground-truth arc_ids (if available/allowed)
    utm_crs: Optional[str] = None
    track_utm: Optional[pd.DataFrame] = None  # columns: x, y, t in meters
    nodes_utm: Optional[pd.DataFrame] = None  # columns: node_id, x, y in meters
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Return a quick summary dict for diagnostic logging."""
        return {
            "trip_id": self.trip_id,
            "partition": self.partition,
            "track_points": len(self.track),
            "nodes_count": len(self.nodes),
            "arcs_count": len(self.arcs),
            "has_route": self.route is not None,
            "route_len": len(self.route) if self.route is not None else 0,
            "utm_crs": self.utm_crs,
        }


def compute_utm_epsg(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> str:
    """Compute the optimal UTM EPSG string for a given longitude/latitude bounding box.

    Design rationale:
    Standard Euclidean distance on raw longitude/latitude degrees is physically distorted
    because 1 degree of longitude scales with cos(latitude) (e.g. at 45°N, 1° lon is ~78.8 km,
    while 1° lat is ~111.1 km).
    Universal Transverse Mercator (UTM) provides a conformal, low-distortion Cartesian metric
    coordinate system (x, y in meters) within 6-degree longitudinal zones. This enables
    exact distance computations (arc lengths, GPS point-to-arc perpendicular distances in meters)
    which are required by HMM Gaussian emission probabilities and transition network routing.
    """
    center_lon = (lon_min + lon_max) / 2.0
    center_lat = (lat_min + lat_max) / 2.0

    zone_number = int((center_lon + 180.0) / 6.0) + 1
    zone_number = max(1, min(60, zone_number))

    if center_lat >= 0:
        epsg_code = 32600 + zone_number  # WGS 84 / UTM zone N
    else:
        epsg_code = 32700 + zone_number  # WGS 84 / UTM zone S

    return f"EPSG:{epsg_code}"


def project_coordinates(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    target_crs: str = "EPSG:32632",
    source_crs: str = "EPSG:4326",
) -> Tuple[np.ndarray, np.ndarray]:
    """Project (lon, lat) coordinates from WGS84 to a metric target CRS (such as UTM).

    Returns:
        (x, y) coordinates in meters as numpy arrays.
    """
    if pyproj is None:
        raise ImportError(
            "pyproj is required for metric UTM projection. Please run 'pip install -r requirements.txt'."
        )
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    lons = df[lon_col].to_numpy(dtype=np.float64)
    lats = df[lat_col].to_numpy(dtype=np.float64)
    x, y = transformer.transform(lons, lats)
    return x, y


def discover_trips(raw_dir: Optional[Path | str] = None) -> Dict[str, Dict[str, Path]]:
    """Scan raw data directory to find available trip files.

    Detects:
    1. Subdirectories per trip: raw_dir / <trip_id> / [*.track, *.nodes, *.arcs, *.route]
    2. Subdirectories per partition: raw_dir / <partition> / <trip_id> / ... or <partition> / <trip_id>.*
    3. Flat directory: raw_dir / <trip_id>.[track|nodes|arcs|route]

    Returns:
        dict: trip_id -> {'track': Path, 'nodes': Path, 'arcs': Path, 'route': Optional[Path], 'dir': Path}
    """
    if raw_dir is None:
        raw_path = get_raw_data_dir()
    else:
        raw_path = Path(raw_dir).resolve()

    trips: Dict[str, Dict[str, Any]] = {}

    if not raw_path.exists():
        logger.warning(f"Raw data directory does not exist: {raw_path}")
        return trips

    # Pattern search across raw directory
    # Find all .track files first, as every valid trip must contain a .track
    track_files = list(raw_path.rglob("*.track"))

    for track_file in track_files:
        trip_name = track_file.stem
        parent_dir = track_file.parent

        # Trip ID heuristic:
        # If the parent folder is named after the trip (e.g. data/raw/trip_01/track.track or data/raw/01/01.track)
        # or if file name is trip_01.track
        if trip_name.lower() in ("track", "data", "points"):
            trip_id = parent_dir.name
        else:
            trip_id = trip_name

        # Look for sibling or relative nodes, arcs, route files
        nodes_candidates = [
            parent_dir / f"{track_file.stem}.nodes",
            parent_dir / "nodes.nodes",
            parent_dir / f"{trip_id}.nodes",
        ]
        arcs_candidates = [
            parent_dir / f"{track_file.stem}.arcs",
            parent_dir / "arcs.arcs",
            parent_dir / f"{trip_id}.arcs",
        ]
        route_candidates = [
            parent_dir / f"{track_file.stem}.route",
            parent_dir / "route.route",
            parent_dir / f"{trip_id}.route",
        ]

        nodes_file = next((p for p in nodes_candidates if p.exists()), None)
        arcs_file = next((p for p in arcs_candidates if p.exists()), None)
        route_file = next((p for p in route_candidates if p.exists()), None)

        trips[trip_id] = {
            "track": track_file,
            "nodes": nodes_file,
            "arcs": arcs_file,
            "route": route_file,
            "dir": parent_dir,
        }

    logger.info(f"Discovered {len(trips)} trips in {raw_path}")
    return trips


def _read_track(path: Path) -> pd.DataFrame:
    """Read .track file: 3 whitespace-separated columns (lon, lat, timestamp in seconds)."""
    if not path.exists():
        raise FileNotFoundError(f"Track file not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "t"],
        dtype={"lon": np.float64, "lat": np.float64, "t": np.float64},
        engine="c",
    )
    if df.empty:
        raise ValueError(f"Track file is empty: {path}")
    if df.shape[1] != 3:
        raise ValueError(f"Track file must have exactly 3 columns (lon, lat, t), got {df.shape[1]} in {path}")
    return df


def _read_nodes(path: Path) -> pd.DataFrame:
    """Read .nodes file: 2 whitespace-separated columns (lon, lat).

    0-based row index = node_id.
    """
    if not path.exists():
        raise FileNotFoundError(f"Nodes file not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": np.float64, "lat": np.float64},
        engine="c",
    )
    if df.empty:
        raise ValueError(f"Nodes file is empty: {path}")
    if df.shape[1] != 2:
        raise ValueError(f"Nodes file must have exactly 2 columns (lon, lat), got {df.shape[1]} in {path}")

    # Add explicit 0-based node_id
    df["node_id"] = np.arange(len(df), dtype=np.int64)
    df = df[["node_id", "lon", "lat"]]
    return df


def _read_arcs(path: Path) -> pd.DataFrame:
    """Read .arcs file: 2 integer columns (from_node_id, to_node_id).

    0-based row index = arc_id.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arcs file not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"],
        dtype={"from_node_id": np.int64, "to_node_id": np.int64},
        engine="c",
    )
    if df.empty:
        raise ValueError(f"Arcs file is empty: {path}")
    if df.shape[1] != 2:
        raise ValueError(f"Arcs file must have exactly 2 columns (from_node_id, to_node_id), got {df.shape[1]} in {path}")

    # Add explicit 0-based arc_id
    df["arc_id"] = np.arange(len(df), dtype=np.int64)
    df = df[["arc_id", "from_node_id", "to_node_id"]]
    return df


def _read_route(path: Path) -> np.ndarray:
    """Read .route file: 1 integer column of chronological ground-truth arc_ids."""
    if not path.exists():
        raise FileNotFoundError(f"Route file not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["arc_id"],
        dtype={"arc_id": np.int64},
        engine="c",
    )
    return df["arc_id"].to_numpy(dtype=np.int64)


def load_trip(
    trip_id: str,
    raw_dir: Optional[Path | str] = None,
    trips_index: Optional[Dict[str, Dict[str, Any]]] = None,
    partition: Optional[str] = None,
    allow_evaluator_labels: bool = False,
    use_cache: bool = True,
    project_utm: bool = True,
) -> TripData:
    """Load a single trip by ID, returning a validated TripData dataclass.

    Validates:
    - File existence (.track, .nodes, .arcs, and optional .route)
    - Column count and expected dtypes
    - Arc endpoint node IDs exist in the nodes table
    - Route arc IDs exist in the arcs table
    - Applies UTM projection dynamically based on the trip's geographic bounding box
    - Evaluator leakage prevention: blocks loading .route for evaluator partition unless
      explicitly permitted by allow_evaluator_labels=True.
    """
    # Check cache first
    cache_dir = get_path("cache_dir", default="data/cache")
    cache_file = cache_dir / f"{trip_id}.pkl"

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                trip_data: TripData = pickle.load(f)
            # Apply evaluator guard even on cached objects
            if trip_data.partition in ("evaluator", "evaluator-testing", "evaluator_test"):
                if not allow_evaluator_labels:
                    trip_data.route = None
                else:
                    logger.warning(
                        f"[DATA LEAKAGE WARNING] Evaluator ground-truth label loaded for trip '{trip_id}'!"
                    )
            return trip_data
        except Exception as e:
            logger.warning(f"Failed reading cache for trip {trip_id}: {e}. Reloading from raw files.")

    if trips_index is None:
        trips_index = discover_trips(raw_dir)

    if trip_id not in trips_index:
        raise KeyError(f"Trip '{trip_id}' not found among discovered trips ({len(trips_index)} available).")

    trip_info = trips_index[trip_id]
    track_path = trip_info.get("track")
    nodes_path = trip_info.get("nodes")
    arcs_path = trip_info.get("arcs")
    route_path = trip_info.get("route")

    if not track_path or not track_path.exists():
        raise FileNotFoundError(f"Missing track file for trip {trip_id}")
    if not nodes_path or not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes file for trip {trip_id}")
    if not arcs_path or not arcs_path.exists():
        raise FileNotFoundError(f"Missing arcs file for trip {trip_id}")

    track_df = _read_track(track_path)
    nodes_df = _read_nodes(nodes_path)
    arcs_df = _read_arcs(arcs_path)

    # Validate arc endpoints exist in nodes
    max_node_id = len(nodes_df) - 1
    if (arcs_df["from_node_id"] > max_node_id).any() or (arcs_df["from_node_id"] < 0).any():
        raise ValueError(f"Trip {trip_id}: arcs table contains invalid from_node_id out of node range.")
    if (arcs_df["to_node_id"] > max_node_id).any() or (arcs_df["to_node_id"] < 0).any():
        raise ValueError(f"Trip {trip_id}: arcs table contains invalid to_node_id out of node range.")

    # Guard evaluator labels
    is_evaluator = partition in ("evaluator", "evaluator-testing", "evaluator_test")
    route_arr: Optional[np.ndarray] = None

    if route_path and route_path.exists():
        if is_evaluator and not allow_evaluator_labels:
            logger.info(
                f"Evaluator partition guard: hiding ground-truth route for trip '{trip_id}'."
            )
            route_arr = None
        else:
            if is_evaluator and allow_evaluator_labels:
                logger.warning(
                    f"[DATA LEAKAGE WARNING] Explicitly loading ground-truth route for evaluator trip '{trip_id}'!"
                )
            route_arr = _read_route(route_path)
            max_arc_id = len(arcs_df) - 1
            if len(route_arr) > 0 and ((route_arr > max_arc_id).any() or (route_arr < 0).any()):
                raise ValueError(f"Trip {trip_id}: route contains invalid arc_id out of arcs range.")

    # Compute bounding box and UTM projection
    utm_crs = None
    track_utm = None
    nodes_utm = None

    if project_utm:
        if pyproj is None:
            logger.warning(
                "pyproj is not installed; skipping metric UTM coordinate projection. "
                "Install requirements.txt to enable metric coordinates."
            )
        else:
            # Bounding box of both track and network nodes
            min_lon = float(min(track_df["lon"].min(), nodes_df["lon"].min()))
            max_lon = float(max(track_df["lon"].max(), nodes_df["lon"].max()))
            min_lat = float(min(track_df["lat"].min(), nodes_df["lat"].min()))
            max_lat = float(max(track_df["lat"].max(), nodes_df["lat"].max()))

            utm_crs = compute_utm_epsg(min_lon, min_lat, max_lon, max_lat)

            tx, ty = project_coordinates(track_df, "lon", "lat", target_crs=utm_crs)
            track_utm = pd.DataFrame({"x": tx, "y": ty, "t": track_df["t"].to_numpy()})

            nx, ny = project_coordinates(nodes_df, "lon", "lat", target_crs=utm_crs)
            nodes_utm = pd.DataFrame({"node_id": nodes_df["node_id"].to_numpy(), "x": nx, "y": ny})

    trip_data = TripData(
        trip_id=trip_id,
        partition=partition,
        track=track_df,
        nodes=nodes_df,
        arcs=arcs_df,
        route=route_arr,
        utm_crs=utm_crs,
        track_utm=track_utm,
        nodes_utm=nodes_utm,
        metadata={
            "track_file": str(track_path),
            "nodes_file": str(nodes_path),
            "arcs_file": str(arcs_path),
            "route_file": str(route_path) if route_path else None,
        },
    )

    # Save to cache if enabled
    if use_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(trip_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.debug(f"Could not write cache for trip {trip_id}: {e}")

    return trip_data


def get_trips(
    partition: str,
    allow_evaluator_labels: bool = False,
    raw_dir: Optional[Path | str] = None,
    split_file: Optional[Path | str] = None,
    use_cache: bool = True,
    project_utm: bool = True,
) -> List[TripData]:
    """Retrieve all trips belonging to a specific partition ('train', 'student-test', 'evaluator-testing').

    Enforces data leakage guard:
    - If partition is evaluator-testing (or 'evaluator'), ground-truth labels (.route)
      are NEVER loaded unless allow_evaluator_labels=True is explicitly supplied.
    - When allow_evaluator_labels=True, a prominent warning is logged.
    """
    norm_part = partition.strip().lower()

    if split_file is None:
        split_path = get_path("split_file", default="configs/split.json")
    else:
        split_path = Path(split_file)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split configuration not found at {split_path}. Run scripts/build_inventory.py first."
        )

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    # Normalize partition keys
    target_ids: List[str] = []
    matched_key = None
    for key, ids in split_data.items():
        if key.lower().replace("-", "_") == norm_part.replace("-", "_"):
            target_ids = ids
            matched_key = key
            break

    if matched_key is None:
        raise ValueError(
            f"Partition '{partition}' not found in split file. Available keys: {list(split_data.keys())}"
        )

    is_eval = norm_part in ("evaluator", "evaluator-testing", "evaluator_test", "test_evaluator")
    if is_eval:
        if not allow_evaluator_labels:
            logger.info("Guard active: evaluator ground-truth routes are withheld.")
        else:
            logger.warning(
                "[DATA LEAKAGE WARNING] allow_evaluator_labels=True! Ground-truth evaluator routes are unmasked!"
            )

    trips_index = discover_trips(raw_dir)
    loaded: List[TripData] = []
    for trip_id in target_ids:
        t = load_trip(
            trip_id=str(trip_id),
            raw_dir=raw_dir,
            trips_index=trips_index,
            partition=matched_key,
            allow_evaluator_labels=allow_evaluator_labels,
            use_cache=use_cache,
            project_utm=project_utm,
        )
        loaded.append(t)

    return loaded
