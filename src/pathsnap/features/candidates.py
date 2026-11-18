"""Spatial candidate arc generation module for PathSnap.

Milestone 2: For every GPS observation in a trip, identifies road network arcs within
a configurable metric radius (e.g. 30m, 50m, 100m, 200m), sorted by distance and capped
at nearest max_candidates.

Key Technical Decisions:
1. Spatial Index Choice (Shapely STRtree vs Midpoint KD-Tree):
   - A KD-tree indexing arc midpoints is inadequate for road networks containing long arcs.
     A long arc (e.g. 500m - 2000m on highways or rural links) can pass within 5m of a GPS point
     while its midpoint is 250m - 1000m away, causing a catastrophic false negative unless the
     search radius is inflated to radius + max_arc_length / 2. Inflating the radius leads to
     massive false positive sets (thousands of midpoints) and heavy filtering overhead.
   - Shapely STRtree (Sort-Tile-Recursive R-tree) indexes 2D axis-aligned bounding boxes (AABBs)
     of the actual segment geometry directly. It handles long and short arcs uniformly without
     search radius inflation.
2. Vectorized Batched Processing:
   - GPS points are queried simultaneously against the STRtree in one batched call
     (predicate="dwithin", distance=radius_m).
   - Point-to-segment Euclidean distances in meters are computed vectorized in C (GEOS)
     without Python loops over arcs or points.
3. Metric Coordinate Invariant:
   - Computations strictly use conformal metric Cartesian coordinates (x, y in meters)
     produced by the loader's dynamic per-trip UTM projection. Never computed in degrees.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shapely

from pathsnap.data.loader import LOADER_VERSION, TripData, compute_utm_epsg, project_coordinates
from pathsnap.utils.config import get_path, get_repo_root

logger = logging.getLogger("pathsnap.features.candidates")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def build_arc_segments(trip: TripData) -> Tuple[np.ndarray, np.ndarray]:
    """Extract metric coordinates for arc endpoints from nodes_utm.

    Returns:
        from_coords: (N_arcs, 2) float64 array of (x, y) start points
        to_coords: (N_arcs, 2) float64 array of (x, y) end points
    """
    if trip.nodes_utm is None or trip.track_utm is None:
        raise ValueError(f"Trip '{trip.trip_id}' has not been projected to UTM. Enable project_utm=True.")

    # Nodes table has 0-based row index = node_id
    nodes_xy = trip.nodes_utm[["x", "y"]].to_numpy(dtype=np.float64)

    from_ids = trip.arcs["from_node_id"].to_numpy(dtype=np.int64)
    to_ids = trip.arcs["to_node_id"].to_numpy(dtype=np.int64)

    from_coords = nodes_xy[from_ids]
    to_coords = nodes_xy[to_ids]
    return from_coords, to_coords


def get_spatial_index(
    trip: TripData,
    use_cache: bool = True,
    cache_dir: Optional[Path | str] = None,
) -> Tuple[shapely.STRtree, np.ndarray]:
    """Build or retrieve a cached shapely STRtree and LineString array for a trip.

    Cached in data/cache/<trip_id>_strtree.pkl with LOADER_VERSION validation.
    """
    if cache_dir is None:
        cache_path = get_path("cache_dir", default="data/cache")
    else:
        cache_path = Path(cache_dir)

    tree_cache_file = cache_path / f"{trip.trip_id}_strtree.pkl"

    if use_cache and tree_cache_file.exists():
        try:
            with open(tree_cache_file, "rb") as f:
                cached_data = pickle.load(f)
            if cached_data.get("loader_version") == LOADER_VERSION and cached_data.get("n_arcs") == len(trip.arcs):
                return cached_data["tree"], cached_data["lines"]
        except Exception as e:
            logger.debug(f"Cache read failed for STRtree {trip.trip_id}: {e}")

    # Build segments
    from_coords, to_coords = build_arc_segments(trip)

    # LineStrings shape: (N_arcs, 2, 2)
    segments = np.stack([from_coords, to_coords], axis=1)
    lines = shapely.linestrings(segments)
    tree = shapely.STRtree(lines)

    if use_cache:
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            with open(tree_cache_file, "wb") as f:
                pickle.dump(
                    {
                        "loader_version": LOADER_VERSION,
                        "n_arcs": len(trip.arcs),
                        "tree": tree,
                        "lines": lines,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as e:
            logger.debug(f"Failed writing STRtree cache for {trip.trip_id}: {e}")

    return tree, lines


def generate_candidates(
    trip: TripData,
    radius_m: float = 50.0,
    max_candidates: int = 30,
    use_index_cache: bool = True,
) -> pd.DataFrame:
    """Generate candidate road network arcs for every GPS point in a trip.

    Args:
        trip: Validated TripData dataclass with track_utm and nodes_utm populated.
        radius_m: Maximum metric distance in meters from GPS observation to road arc.
        max_candidates: Maximum number of nearest candidate arcs to retain per observation.
        use_index_cache: Whether to use disk caching for the trip's STRtree index.

    Returns:
        pd.DataFrame with columns:
          - trip_id: str
          - obs_idx: int64 (0-based row index in trip.track)
          - arc_id: int64 (0-based row index in trip.arcs)
          - distance_m: float64 (exact Euclidean distance to line segment in meters)
        Sorted by [obs_idx, distance_m] ascending.
        Observations with zero candidates are recorded in df.attrs["no_candidates"].
    """
    # Guard against missing UTM projection
    if trip.track_utm is None or trip.nodes_utm is None or trip.utm_crs is None:
        min_lon = float(min(trip.track["lon"].min(), trip.nodes["lon"].min()))
        max_lon = float(max(trip.track["lon"].max(), trip.nodes["lon"].max()))
        min_lat = float(min(trip.track["lat"].min(), trip.nodes["lat"].min()))
        max_lat = float(max(trip.track["lat"].max(), trip.nodes["lat"].max()))
        utm_crs = compute_utm_epsg(min_lon, min_lat, max_lon, max_lat)
        tx, ty = project_coordinates(trip.track, "lon", "lat", target_crs=utm_crs)
        trip.track_utm = pd.DataFrame({"x": tx, "y": ty, "t": trip.track["t"].to_numpy()})
        nx, ny = project_coordinates(trip.nodes, "lon", "lat", target_crs=utm_crs)
        trip.nodes_utm = pd.DataFrame({"node_id": trip.nodes["node_id"].to_numpy(), "x": nx, "y": ny})
        trip.utm_crs = utm_crs

    n_obs = len(trip.track)
    if n_obs == 0 or len(trip.arcs) == 0:
        empty_df = pd.DataFrame(
            columns=["trip_id", "obs_idx", "arc_id", "distance_m"]
        ).astype({
            "trip_id": str,
            "obs_idx": np.int64,
            "arc_id": np.int64,
            "distance_m": np.float64,
        })
        empty_df.attrs["no_candidates"] = list(range(n_obs))
        empty_df.attrs["n_zero_candidates"] = n_obs
        empty_df.attrs["n_obs"] = n_obs
        empty_df.attrs["radius_m"] = radius_m
        empty_df.attrs["max_candidates"] = max_candidates
        return empty_df

    # 1. Build / retrieve spatial index
    tree, lines = get_spatial_index(trip, use_cache=use_index_cache)

    # 2. Build GPS point geometries
    gps_coords = trip.track_utm[["x", "y"]].to_numpy(dtype=np.float64)
    gps_points = shapely.points(gps_coords)

    # 3. Batched spatial query: STRtree.query with predicate 'dwithin'
    # Returns (2, N_pairs): row 0 = indices of gps_points, row 1 = indices of lines (arc_id)
    pt_indices, arc_indices = tree.query(gps_points, predicate="dwithin", distance=float(radius_m))

    if len(pt_indices) == 0:
        df = pd.DataFrame(
            columns=["trip_id", "obs_idx", "arc_id", "distance_m"]
        ).astype({
            "trip_id": str,
            "obs_idx": np.int64,
            "arc_id": np.int64,
            "distance_m": np.float64,
        })
        df.attrs["no_candidates"] = list(range(n_obs))
        df.attrs["n_zero_candidates"] = n_obs
        df.attrs["n_obs"] = n_obs
        df.attrs["radius_m"] = radius_m
        df.attrs["max_candidates"] = max_candidates
        return df

    # 4. Exact point-to-segment Euclidean distance calculation in meters
    dists = shapely.distance(gps_points[pt_indices], lines[arc_indices])

    # Enforce strictly <= radius_m (handles floating-point edge cases)
    valid_mask = dists <= (radius_m + 1e-9)
    pt_indices = pt_indices[valid_mask]
    arc_indices = arc_indices[valid_mask]
    dists = dists[valid_mask]

    # 5. Sort by [obs_idx, distance_m] ascending and cap at max_candidates nearest arcs
    # Using fast lexsort on (distance, obs_idx)
    sort_order = np.lexsort((dists, pt_indices))
    sorted_obs = pt_indices[sort_order]
    sorted_arcs = arc_indices[sort_order]
    sorted_dists = dists[sort_order]

    # Calculate rank per observation
    diff = np.diff(sorted_obs, prepend=-1) != 0
    start_indices = np.where(diff)[0]
    segment_lengths = np.diff(np.append(start_indices, len(sorted_obs)))
    ranks = np.arange(len(sorted_obs)) - np.repeat(start_indices, segment_lengths)

    cap_mask = ranks < max_candidates
    capped_obs = sorted_obs[cap_mask]
    capped_arcs = sorted_arcs[cap_mask]
    capped_dists = sorted_dists[cap_mask]

    # 6. Identify observations with zero candidates
    unique_matched = np.unique(capped_obs)
    all_obs = np.arange(n_obs, dtype=np.int64)
    zero_cand_mask = ~np.isin(all_obs, unique_matched)
    zero_cand_indices = all_obs[zero_cand_mask].tolist()

    df = pd.DataFrame({
        "trip_id": trip.trip_id,
        "obs_idx": capped_obs.astype(np.int64),
        "arc_id": capped_arcs.astype(np.int64),
        "distance_m": capped_dists.astype(np.float64),
    })

    # Record zero-candidate metadata
    df.attrs["no_candidates"] = zero_cand_indices
    df.attrs["n_zero_candidates"] = len(zero_cand_indices)
    df.attrs["n_obs"] = n_obs
    df.attrs["radius_m"] = radius_m
    df.attrs["max_candidates"] = max_candidates

    return df


def save_candidates_parquet(df: pd.DataFrame, out_path: Path | str) -> Path:
    """Save candidates DataFrame to Parquet format."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, engine="pyarrow")
    return p


def load_candidates_parquet(path: Path | str) -> pd.DataFrame:
    """Load candidates DataFrame from Parquet format."""
    return pd.read_parquet(Path(path), engine="pyarrow")
