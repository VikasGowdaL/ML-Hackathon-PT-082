"""Per-candidate geometry features for PathSnap candidate ranking (Milestone 2, Person 3).

Input contract
--------------
``candidates``: one row per (GPS observation, candidate arc), produced by Person 1's candidate
generator, with at least ``trip_id``, ``obs_idx`` and ``arc_id``:

* ``obs_idx`` -- 0-based row index of the point in the trip's ``.track``;
* ``arc_id``  -- 0-based row index of the arc in the trip's ``.arcs`` (never re-indexed).

Extra columns (e.g. Person 1's ``distance_m``) are passed through unchanged; row order is preserved
and no row is dropped.

Coordinates
-----------
All distances and bearings are computed in the trip's **UTM projection** (metres), using Member 1's
loader (``TripData.track_utm`` / ``nodes_utm`` / ``utm_crs``). Bearings are *grid* bearings: degrees
clockwise from grid north, ``atan2(dx, dy)``. GPS heading and arc bearing use the same grid, so their
difference does not depend on grid convergence. :func:`check_projected_geometry` verifies this.

Output columns (see ``docs/geometry_features.md`` for the full data dictionary)
----------------------------------------------------------------------------
Per candidate: ``dist_to_segment_m, proj_offset_m, proj_fraction, proj_remaining_m, proj_is_endpoint,
gps_heading_deg, heading_valid, arc_bearing_deg, bearing_diff_deg, bearing_diff_undirected_deg``.
Per arc: ``seg_length_m, arc_is_zero_length, arc_is_two_way, arc_parallel_count, from_in_degree,
from_out_degree, to_in_degree, to_out_degree, arc_in_largest_scc, arc_wcc_fraction``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd

from pathsnap import road_network as rn

__all__ = [
    "GEOMETRY_FEATURE_COLUMNS",
    "grid_bearing_deg",
    "angle_diff_deg",
    "compute_gps_heading",
    "compute_arc_static_features",
    "compute_geometry_features",
    "check_projected_geometry",
]

ARC_FEATURE_COLUMNS = [
    "seg_length_m",
    "arc_is_zero_length",
    "arc_is_two_way",
    "arc_parallel_count",
    "from_in_degree",
    "from_out_degree",
    "to_in_degree",
    "to_out_degree",
    "arc_in_largest_scc",
    "arc_wcc_fraction",
]
CANDIDATE_FEATURE_COLUMNS = [
    "dist_to_segment_m",
    "proj_offset_m",
    "proj_fraction",
    "proj_remaining_m",
    "proj_is_endpoint",
    "gps_heading_deg",
    "heading_valid",
    "arc_bearing_deg",
    "bearing_diff_deg",
    "bearing_diff_undirected_deg",
]
GEOMETRY_FEATURE_COLUMNS = CANDIDATE_FEATURE_COLUMNS + ARC_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------
def grid_bearing_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Bearing in degrees clockwise from grid north (``[0, 360)``); NaN where ``dx == dy == 0``."""
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    b = np.degrees(np.arctan2(dx, dy)) % 360.0
    return np.where((dx == 0) & (dy == 0), np.nan, b)


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute difference between two bearings, in ``[0, 180]`` (NaN propagates)."""
    d = np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0)
    return np.minimum(d, 360.0 - d)


# ---------------------------------------------------------------------------
# GPS heading (per observation)
# ---------------------------------------------------------------------------
def compute_gps_heading(x: np.ndarray, y: np.ndarray, min_move_m: float = 2.0) -> pd.DataFrame:
    """Heading of each GPS point from its projected neighbours.

    Uses the central difference ``p[i-1] -> p[i+1]``; if that displacement is shorter than
    ``min_move_m`` (vehicle stopped / GPS jitter) it falls back to ``p[i] -> p[i+1]`` and then
    ``p[i-1] -> p[i]``. If no displacement reaches ``min_move_m`` the heading is NaN and
    ``heading_valid`` is False.

    Args:
        x, y: projected (UTM) coordinates in metres, in track order.
        min_move_m: minimum displacement for a heading to be considered reliable.

    Returns:
        DataFrame ``obs_idx, gps_heading_deg, heading_valid, heading_baseline_m``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    heading = np.full(n, np.nan)
    baseline = np.zeros(n)
    if n >= 2:
        nxt = np.r_[np.arange(1, n), n - 1]
        prv = np.r_[0, np.arange(0, n - 1)]
        for a, b in ((prv, nxt), (np.arange(n), nxt), (prv, np.arange(n))):
            dx, dy = x[b] - x[a], y[b] - y[a]
            dist = np.hypot(dx, dy)
            fill = np.isnan(heading) & (dist >= min_move_m)
            heading[fill] = grid_bearing_deg(dx[fill], dy[fill])
            baseline[fill] = dist[fill]
    return pd.DataFrame(
        {
            "obs_idx": np.arange(n),
            "gps_heading_deg": heading,
            "heading_valid": ~np.isnan(heading),
            "heading_baseline_m": baseline,
        }
    )


# ---------------------------------------------------------------------------
# Arc-level features (computed once per trip)
# ---------------------------------------------------------------------------
def compute_arc_static_features(
    nodes_utm: pd.DataFrame,
    arcs_df: pd.DataFrame,
    G: Optional[nx.MultiDiGraph] = None,
) -> pd.DataFrame:
    """Geometry and connectivity of every arc of one trip, indexed by original ``arc_id``.

    Args:
        nodes_utm: ``node_id, x, y`` (metres), row index == node_id.
        arcs_df: ``arc_id, from_node_id, to_node_id``.
        G: optional pre-built graph from :func:`pathsnap.road_network.build_road_graph`
           (built here if omitted).

    Returns:
        DataFrame indexed by ``arc_id`` with ``from_x, from_y, to_x, to_y, arc_bearing_deg`` and
        all ``ARC_FEATURE_COLUMNS``.
    """
    f = arcs_df["from_node_id"].to_numpy(dtype=np.int64)
    t = arcs_df["to_node_id"].to_numpy(dtype=np.int64)
    nx_ = nodes_utm["x"].to_numpy(dtype=float)
    ny_ = nodes_utm["y"].to_numpy(dtype=float)
    fx, fy, tx, ty = nx_[f], ny_[f], nx_[t], ny_[t]
    length = np.hypot(tx - fx, ty - fy)

    n_nodes = len(nodes_utm)
    val = rn.validate_arcs(arcs_df, n_nodes)
    if G is None:
        # topology only (same node / edge / key convention as road_network.build_road_graph)
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(zip(f.tolist(), t.tolist(), arcs_df["arc_id"].tolist()))
    deg = rn.calculate_node_degrees(G).set_index("node_id")
    _, comp = rn.calculate_connectivity_statistics(G, return_sizes=True)
    in_scc = np.zeros(G.number_of_nodes(), dtype=bool)
    in_scc[list(comp["largest_scc"])] = True
    wcc_size = np.zeros(G.number_of_nodes(), dtype=np.int64)
    for c in nx.weakly_connected_components(G):
        wcc_size[list(c)] = len(c)

    out = pd.DataFrame(
        {
            "arc_id": arcs_df["arc_id"].to_numpy(),
            "from_x": fx, "from_y": fy, "to_x": tx, "to_y": ty,
            "seg_length_m": length,
            "arc_bearing_deg": grid_bearing_deg(tx - fx, ty - fy),
            "arc_is_zero_length": length == 0,
            "arc_is_two_way": val["has_reverse"].to_numpy(),
            "arc_parallel_count": val["parallel_count"].to_numpy(),
            "from_in_degree": deg["in_degree"].to_numpy()[f],
            "from_out_degree": deg["out_degree"].to_numpy()[f],
            "to_in_degree": deg["in_degree"].to_numpy()[t],
            "to_out_degree": deg["out_degree"].to_numpy()[t],
            "arc_in_largest_scc": in_scc[f] & in_scc[t],
            "arc_wcc_fraction": wcc_size[f] / max(G.number_of_nodes(), 1),
        }
    )
    return out.set_index("arc_id")


# ---------------------------------------------------------------------------
# Candidate features
# ---------------------------------------------------------------------------
def compute_geometry_features(
    candidates: pd.DataFrame,
    track_utm: pd.DataFrame,
    arc_static: pd.DataFrame,
    heading: Optional[pd.DataFrame] = None,
    min_move_m: float = 2.0,
) -> pd.DataFrame:
    """Add geometry features to the candidates of ONE trip.

    Args:
        candidates: rows with ``obs_idx`` and ``arc_id`` (other columns are kept).
        track_utm: ``x, y, t`` of the trip's GPS points (row index == obs_idx).
        arc_static: output of :func:`compute_arc_static_features` for the same trip.
        heading: optional output of :func:`compute_gps_heading` (computed if omitted).
        min_move_m: passed to :func:`compute_gps_heading`.

    Returns:
        ``candidates`` (same rows, same order) plus ``GEOMETRY_FEATURE_COLUMNS``.

    Raises:
        IndexError if an ``obs_idx`` or ``arc_id`` is outside the trip's track / arcs.
    """
    obs = candidates["obs_idx"].to_numpy(dtype=np.int64)
    arc = candidates["arc_id"].to_numpy(dtype=np.int64)
    if len(obs) and (obs.min() < 0 or obs.max() >= len(track_utm)):
        raise IndexError("obs_idx outside the trip's track")
    if len(arc) and (arc.min() < 0 or arc.max() >= len(arc_static)):
        raise IndexError("arc_id outside the trip's arcs")
    if heading is None:
        heading = compute_gps_heading(track_utm["x"].to_numpy(), track_utm["y"].to_numpy(), min_move_m)

    a = arc_static.iloc[arc]  # arc_id == row position
    px = track_utm["x"].to_numpy(dtype=float)[obs]
    py = track_utm["y"].to_numpy(dtype=float)[obs]
    ax, ay = a["from_x"].to_numpy(), a["from_y"].to_numpy()
    dx, dy = a["to_x"].to_numpy() - ax, a["to_y"].to_numpy() - ay
    L = a["seg_length_m"].to_numpy()
    L2 = np.where(L > 0, L * L, 1.0)
    u_raw = ((px - ax) * dx + (py - ay) * dy) / L2
    u = np.where(L > 0, np.clip(u_raw, 0.0, 1.0), 0.0)
    qx, qy = ax + u * dx, ay + u * dy

    out = candidates.copy()
    out["dist_to_segment_m"] = np.hypot(px - qx, py - qy)
    out["proj_offset_m"] = u * L
    out["proj_fraction"] = u
    out["proj_remaining_m"] = (1.0 - u) * L
    out["proj_is_endpoint"] = (L == 0) | (u_raw <= 0.0) | (u_raw >= 1.0)
    h = heading["gps_heading_deg"].to_numpy()[obs]
    out["gps_heading_deg"] = h
    out["heading_valid"] = heading["heading_valid"].to_numpy()[obs]
    out["arc_bearing_deg"] = a["arc_bearing_deg"].to_numpy()
    d = angle_diff_deg(h, out["arc_bearing_deg"].to_numpy())
    out["bearing_diff_deg"] = d
    out["bearing_diff_undirected_deg"] = np.minimum(d, 180.0 - d)
    for c in ARC_FEATURE_COLUMNS:
        out[c] = a[c].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Projection check
# ---------------------------------------------------------------------------
def check_projected_geometry(trip: Any, sample: int = 5000, seed: int = 42) -> Dict[str, Any]:
    """Verify that a trip's geometry is in a metric UTM projection and quantify its distortion.

    Compares, on a reproducible sample of arcs, the UTM segment length with the haversine length
    and the UTM grid bearing with the geodesic azimuth (WGS84, ``pyproj.Geod``).

    Args:
        trip: ``pathsnap.data.loader.TripData`` loaded with ``project_utm=True``.

    Returns:
        dict with ``utm_crs, is_utm, length_ratio_median, length_ratio_max_dev_pct,
        bearing_offset_median_deg, bearing_offset_spread_deg``. The bearing offset is the grid
        convergence: nearly constant inside a trip, so GPS-heading vs arc-bearing differences
        (both on the grid) are unaffected.
    """
    import pyproj

    crs = trip.utm_crs or ""
    code = int(crs.split(":")[1]) if crs.startswith("EPSG:") else 0
    arcs = trip.arcs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arcs), size=min(sample, len(arcs)), replace=False)
    a = arcs.iloc[idx]
    f, t = a.from_node_id.to_numpy(), a.to_node_id.to_numpy()
    nu, nl = trip.nodes_utm, trip.nodes
    dx = nu.x.to_numpy()[t] - nu.x.to_numpy()[f]
    dy = nu.y.to_numpy()[t] - nu.y.to_numpy()[f]
    utm_len = np.hypot(dx, dy)
    lon, lat = nl.lon.to_numpy(), nl.lat.to_numpy()
    hav = rn.haversine_m(lon[f], lat[f], lon[t], lat[t])
    ok = hav > 5.0  # ignore very short arcs (rounding dominates)
    ratio = utm_len[ok] / hav[ok]
    az, _, _ = pyproj.Geod(ellps="WGS84").inv(lon[f][ok], lat[f][ok], lon[t][ok], lat[t][ok])
    offset = (grid_bearing_deg(dx[ok], dy[ok]) - (np.asarray(az) % 360.0) + 180.0) % 360.0 - 180.0
    return {
        "utm_crs": crs,
        "is_utm": 32601 <= code <= 32660 or 32701 <= code <= 32760,
        "arcs_compared": int(ok.sum()),
        "length_ratio_median": float(np.median(ratio)),
        "length_ratio_max_dev_pct": float(100 * np.max(np.abs(ratio - 1))),
        "bearing_offset_median_deg": float(np.median(offset)),
        "bearing_offset_spread_deg": float(np.percentile(offset, 99) - np.percentile(offset, 1)),
    }
