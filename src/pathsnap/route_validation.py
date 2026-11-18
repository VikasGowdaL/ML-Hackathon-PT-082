"""Ground-truth route validation, GPS-to-route distances and Folium maps for PathSnap.

Owner: Member 4 (route checks and Folium).

Reuses:
* Member 1 loader (``pathsnap.data.loader``): raw readers, UTM zone selection and projection.
* Member 3 road network (``pathsnap.road_network``): ``.nodes``/``.arcs`` loading, the
  ``MultiDiGraph`` (edge key = original ``arc_id``) and segment geometry.

Data contract
-------------
``route``    1-D int array of arc IDs in travel order (row indices of the trip's ``.arcs``).
``arcs_df``  ``arc_id, from_node_id, to_node_id``;  ``nodes_df`` ``node_id, lon, lat``.
``track_df`` ``lon, lat, t``.

Evaluator ``.route`` files must never be read; :func:`analyse_route_trip` refuses evaluator partitions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd

from pathsnap.data.loader import _read_route, _read_track, compute_utm_epsg, project_coordinates
from pathsnap import road_network as rn

__all__ = [
    "validate_route_ids",
    "check_route_continuity",
    "gps_to_route_distances",
    "make_route_map",
    "load_route_trip",
    "analyse_route_trip",
]

EVALUATOR_PARTITIONS = {"evaluator", "evaluator-testing", "evaluator_test"}
DIST_THRESHOLDS_M = (20, 50, 100)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_route_trip(trip_id: str, trips_index: Dict[str, Dict[str, Any]], partition: str) -> Dict[str, Any]:
    """Load track, nodes, arcs and route of one non-evaluator trip without raising on bad route IDs.

    Uses Member 1's raw readers (``load_trip`` would raise on invalid route IDs, which we must report instead).
    Returns dict ``track, nodes, arcs, route, errors``; missing pieces are ``None``.
    """
    if partition.lower() in EVALUATOR_PARTITIONS:
        raise PermissionError("Route validation must not read evaluator ground truth.")
    info = trips_index.get(trip_id, {})
    out: Dict[str, Any] = {"track": None, "route": None, "errors": []}
    nodes, arcs, rep = rn.load_road_network(trip_id, trips_index, scan=False)
    out["nodes"], out["arcs"] = nodes, arcs
    out["errors"] += rep["errors"]
    for key, reader in (("track", _read_track), ("route", _read_route)):
        path = info.get(key)
        if path is None:
            out["errors"].append(f"missing .{key} file")
            continue
        try:
            out[key] = reader(path)
        except Exception as exc:  # malformed / empty file
            out["errors"].append(f".{key} unreadable: {exc}")
    return out


# ---------------------------------------------------------------------------
# 1. Route ID validity
# ---------------------------------------------------------------------------
def validate_route_ids(route: np.ndarray, n_arcs: int) -> Dict[str, Any]:
    """Check that every route arc ID is a valid row index of ``.arcs``.

    Returns dict: ``route_length, invalid_arc_ids, invalid_positions`` (list, first 20),
    ``unique_arcs, consecutive_repeats`` (same arc twice in a row), ``revisited_arcs``
    (arcs that reappear later, non-consecutively).
    """
    route = np.asarray(route, dtype=np.int64)
    invalid = (route < 0) | (route >= n_arcs)
    consecutive = int((route[1:] == route[:-1]).sum()) if len(route) > 1 else 0
    # collapse consecutive repeats, then count arcs that still occur more than once
    collapsed = route[np.r_[True, route[1:] != route[:-1]]] if len(route) else route
    counts = pd.Series(collapsed).value_counts()
    return {
        "route_length": int(len(route)),
        "invalid_arc_ids": int(invalid.sum()),
        "invalid_positions": np.flatnonzero(invalid)[:20].tolist(),
        "unique_arcs": int(len(np.unique(route))),
        "consecutive_repeats": consecutive,
        "revisited_arcs": int((counts > 1).sum()),
    }


# ---------------------------------------------------------------------------
# 2. Route continuity
# ---------------------------------------------------------------------------
def check_route_continuity(
    route: np.ndarray,
    arcs_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    G: Optional[nx.MultiDiGraph] = None,
) -> pd.DataFrame:
    """Check that each consecutive pair of route arcs connects head-to-tail.

    For each transition ``i -> i+1`` (only between valid arc IDs) the status is:

    * ``connected``   -- ``to_node(a_i) == from_node(a_{i+1})`` (normal case);
    * ``same_arc``    -- the same arc repeated;
    * ``shares_node`` -- the arcs share an endpoint, but not head-to-tail (e.g. direction mismatch);
    * ``gap``         -- no shared endpoint.

    For non-connected transitions, ``gap_m`` is the haversine distance from the end of ``a_i`` to
    the start of ``a_{i+1}``; if ``G`` is given, ``path_exists`` / ``path_arcs`` / ``path_m`` report
    the shortest *directed* path (by ``length_m``) between those nodes.

    Returns one row per transition: ``position, arc_id, next_arc_id, end_node, next_start_node,
    status, gap_m, path_exists, path_arcs, path_m``.
    """
    route = np.asarray(route, dtype=np.int64)
    n_arcs = len(arcs_df)
    f = arcs_df["from_node_id"].to_numpy()
    t = arcs_df["to_node_id"].to_numpy()
    valid = (route >= 0) & (route < n_arcs)
    pos = np.flatnonzero(valid[:-1] & valid[1:]) if len(route) > 1 else np.array([], dtype=int)
    a, b = route[pos], route[pos + 1]
    df = pd.DataFrame(
        {"position": pos, "arc_id": a, "next_arc_id": b, "end_node": t[a], "next_start_node": f[b]}
    )
    connected = df.end_node.to_numpy() == df.next_start_node.to_numpy()
    same = a == b
    shares =(t[a] == t[b]) | (f[a] == f[b]) | (f[a] == t[b])
    df["status"] = np.select([same, connected, shares], ["same_arc", "connected", "shares_node"], default="gap")

    lon = nodes_df["lon"].to_numpy(dtype=float)
    lat = nodes_df["lat"].to_numpy(dtype=float)
    df["gap_m"] = 0.0
    df["path_exists"] = pd.NA
    df["path_arcs"] = pd.NA
    df["path_m"] = np.nan
    bad = df.index[~df.status.isin(["connected", "same_arc"])]
    if len(bad):
        u = df.loc[bad, "end_node"].to_numpy()
        v = df.loc[bad, "next_start_node"].to_numpy()
        df.loc[bad, "gap_m"] = rn.haversine_m(lon[u], lat[u], lon[v], lat[v])
        if G is not None:
            for i, uu, vv in zip(bad, u, v):
                try:
                    length, path = nx.single_source_dijkstra(G, int(uu), int(vv), weight="length_m")
                    df.loc[i, ["path_exists", "path_arcs", "path_m"]] = [True, len(path) - 1, length]
                except nx.NetworkXNoPath:
                    df.loc[i, "path_exists"] = False
    return df


# ---------------------------------------------------------------------------
# 3. GPS-to-route distance
# ---------------------------------------------------------------------------
def gps_to_route_distances(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route: np.ndarray,
    chunk: int = 2000,
) -> pd.DataFrame:
    """Perpendicular distance (metres) from every GPS point to the nearest segment of the route.

    Coordinates are projected to the trip's UTM zone with Member 1's ``compute_utm_epsg`` /
    ``project_coordinates``. Each route arc is treated as a straight segment between its nodes.

    Returns DataFrame ``point_index, t, distance_m, nearest_arc_id`` (one row per track point,
    original track order).
    """
    route = np.asarray(route, dtype=np.int64)
    route = np.unique(route[(route >= 0) & (route < len(arcs_df))])
    if len(route) == 0 or len(track_df) == 0:
        return pd.DataFrame(columns=["point_index", "t", "distance_m", "nearest_arc_id"])
    crs = compute_utm_epsg(
        float(min(track_df.lon.min(), nodes_df.lon.min())), float(min(track_df.lat.min(), nodes_df.lat.min())),
        float(max(track_df.lon.max(), nodes_df.lon.max())), float(max(track_df.lat.max(), nodes_df.lat.max())),
    )
    used = np.unique(np.r_[arcs_df.from_node_id.to_numpy()[route], arcs_df.to_node_id.to_numpy()[route]])
    nx_, ny_ = project_coordinates(nodes_df.iloc[used], target_crs=crs)
    node_xy = {int(n): (x, y) for n, x, y in zip(used, nx_, ny_)}
    ax = np.array([node_xy[int(n)][0] for n in arcs_df.from_node_id.to_numpy()[route]])
    ay = np.array([node_xy[int(n)][1] for n in arcs_df.from_node_id.to_numpy()[route]])
    bx = np.array([node_xy[int(n)][0] for n in arcs_df.to_node_id.to_numpy()[route]])
    by = np.array([node_xy[int(n)][1] for n in arcs_df.to_node_id.to_numpy()[route]])
    px, py = project_coordinates(track_df, target_crs=crs)

    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    seg_len2[seg_len2 == 0] = 1e-12  # zero-length arcs -> distance to the point
    best_d = np.empty(len(px))
    best_arc = np.empty(len(px), dtype=np.int64)
    for s in range(0, len(px), chunk):
        qx, qy = px[s:s + chunk, None], py[s:s + chunk, None]
        u = np.clip(((qx - ax) * dx + (qy - ay) * dy) / seg_len2, 0, 1)
        d = np.hypot(qx - (ax + u * dx), qy - (ay + u * dy))
        k = d.argmin(axis=1)
        best_d[s:s + chunk] = d[np.arange(len(k)), k]
        best_arc[s:s + chunk] = route[k]
    return pd.DataFrame(
        {"point_index": np.arange(len(px)), "t": track_df["t"].to_numpy(), "distance_m": best_d, "nearest_arc_id": best_arc}
    )


# ---------------------------------------------------------------------------
# 4. Folium map
# ---------------------------------------------------------------------------
def make_route_map(
    trip_id: str,
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route: np.ndarray,
    continuity: Optional[pd.DataFrame] = None,
    distances: Optional[pd.DataFrame] = None,
    max_points: int = 1500,
    seed: int = 42,
):
    """Build a Folium map: ground-truth route, GPS points (coloured by distance), start/end, breaks.

    GPS points are sub-sampled reproducibly (``seed``) to at most ``max_points``. Returns ``folium.Map``.
    """
    import folium

    route = np.asarray(route, dtype=np.int64)
    route = route[(route >= 0) & (route < len(arcs_df))]
    geo = rn.get_road_segment_geometry(nodes_df, arcs_df, arc_ids=route)
    m = folium.Map(location=[track_df.lat.mean(), track_df.lon.mean()], zoom_start=14, tiles="OpenStreetMap")

    # Route drawn as continuous runs; a new polyline starts wherever the route breaks
    fg_route = folium.FeatureGroup(name=f"Ground-truth route ({len(route)} arcs)")
    run = [(geo.from_lat[0], geo.from_lon[0])]
    for i in range(len(geo)):
        if i > 0 and (geo.from_lat[i], geo.from_lon[i]) != run[-1]:
            folium.PolyLine(run, color="#1f6feb", weight=5, opacity=0.8).add_to(fg_route)
            run = [(geo.from_lat[i], geo.from_lon[i])]
        run.append((geo.to_lat[i], geo.to_lon[i]))
    folium.PolyLine(run, color="#1f6feb", weight=5, opacity=0.8).add_to(fg_route)
    fg_route.add_to(m)

    # GPS points
    idx = np.arange(len(track_df))
    if len(idx) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=max_points, replace=False))
    fg_gps = folium.FeatureGroup(name=f"GPS points ({len(idx)} of {len(track_df)} shown)")
    dist = distances.distance_m.to_numpy() if distances is not None else None
    for i in idx:
        r = track_df.iloc[i]
        if dist is not None:
            d = dist[i]
            color = "#2da44e" if d <= 20 else "#d29922" if d <= 50 else "#cf222e"
            tip = f"point {i}, t={r.t:.0f}s, {d:.1f} m from route"
        else:
            color, tip = "#6e7781", f"point {i}"
        folium.CircleMarker([r.lat, r.lon], radius=2.5, color=color, fill=True, fill_opacity=0.8, weight=0, tooltip=tip).add_to(fg_gps)
    fg_gps.add_to(m)

    # Route breaks
    if continuity is not None:
        brk = continuity[~continuity.status.isin(["connected", "same_arc"])]
        if len(brk):
            fg_b = folium.FeatureGroup(name=f"Route breaks ({len(brk)})")
            lon, lat = nodes_df.lon.to_numpy(), nodes_df.lat.to_numpy()
            for r in brk.itertuples():
                folium.Marker(
                    [lat[r.end_node], lon[r.end_node]],
                    icon=folium.Icon(color="red", icon="exclamation-sign"),
                    tooltip=f"break at position {r.position}: arc {r.arc_id} -> {r.next_arc_id} ({r.status}, {r.gap_m:.0f} m)",
                ).add_to(fg_b)
            fg_b.add_to(m)

    folium.Marker([track_df.lat.iloc[0], track_df.lon.iloc[0]], tooltip="GPS start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([track_df.lat.iloc[-1], track_df.lon.iloc[-1]], tooltip="GPS end", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    m.fit_bounds([[min(track_df.lat.min(), geo.from_lat.min()), min(track_df.lon.min(), geo.from_lon.min())],
                  [max(track_df.lat.max(), geo.from_lat.max()), max(track_df.lon.max(), geo.from_lon.max())]])
    legend = (
        f"<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px 10px;"
        f"border:1px solid #888;border-radius:4px;font-size:12px'><b>Trip {trip_id}</b><br>"
        "<span style='color:#1f6feb'>&#9644;</span> ground-truth route<br>"
        "<span style='color:#2da44e'>&#9679;</span> GPS &le; 20 m &nbsp;"
        "<span style='color:#d29922'>&#9679;</span> 20&ndash;50 m &nbsp;"
        "<span style='color:#cf222e'>&#9679;</span> &gt; 50 m</div>"
    )
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Per-trip driver (used by notebooks/04_route_validation.ipynb)
# ---------------------------------------------------------------------------
def analyse_route_trip(trip_id: str, partition: str, trips_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Run all route checks for one trip. The road graph is only built when the route has breaks.

    Returns dict with ``summary`` (route checks), ``distance_summary`` (GPS-to-route stats),
    ``breaks`` (non-connected transitions) and ``distances`` (per-point distances).
    """
    d = load_route_trip(trip_id, trips_index, partition)
    base = {"trip_id": trip_id, "partition": partition}
    if d["route"] is None or d["arcs"] is None or d["nodes"] is None:
        return {"summary": {**base, "load_errors": "; ".join(d["errors"])}, "ok": False}
    ids = validate_route_ids(d["route"], len(d["arcs"]))
    cont = check_route_continuity(d["route"], d["arcs"], d["nodes"])
    breaks = cont[~cont.status.isin(["connected", "same_arc"])].copy()
    if len(breaks):  # only now pay for the graph
        G, _, _ = rn.build_road_graph(d["nodes"], d["arcs"])
        breaks = check_route_continuity(d["route"], d["arcs"], d["nodes"], G)
        breaks = breaks[~breaks.status.isin(["connected", "same_arc"])].copy()
        del G
    breaks.insert(0, "trip_id", trip_id)
    geo = rn.get_road_segment_geometry(d["nodes"], d["arcs"], arc_ids=d["route"][(d["route"] >= 0) & (d["route"] < len(d["arcs"]))])
    status = cont.status.value_counts()
    summary = {
        **base,
        "n_arcs_in_network": len(d["arcs"]),
        **{k: v for k, v in ids.items() if k != "invalid_positions"},
        "transitions": len(cont),
        "connected": int(status.get("connected", 0)),
        "same_arc": int(status.get("same_arc", 0)),
        "shares_node": int(status.get("shares_node", 0)),
        "gaps": int(status.get("gap", 0)),
        "is_continuous": bool(len(breaks) == 0 and ids["invalid_arc_ids"] == 0),
        "route_length_km": round(float(np.nansum(geo.length_m)) / 1000, 3),
        "load_errors": "; ".join(d["errors"]),
    }
    dist_summary = dict(base)
    distances = None
    if d["track"] is not None:
        distances = gps_to_route_distances(d["track"], d["nodes"], d["arcs"], d["route"])
        x = distances.distance_m.to_numpy()
        dist_summary.update(
            gps_points=len(x),
            mean_m=round(float(x.mean()), 2),
            median_m=round(float(np.median(x)), 2),
            p90_m=round(float(np.percentile(x, 90)), 2),
            p95_m=round(float(np.percentile(x, 95)), 2),
            max_m=round(float(x.max()), 2),
            **{f"pct_over_{k}m": round(100 * float((x > k).mean()), 3) for k in DIST_THRESHOLDS_M},
        )
        distances.insert(0, "trip_id", trip_id)
    return {"summary": summary, "distance_summary": dist_summary, "breaks": breaks, "distances": distances, "ok": True}
