"""Road-network validation, graph construction and connectivity analysis for PathSnap.

Owner: Member 3 (road-network analysis).

This module turns a trip's ``.nodes`` / ``.arcs`` tables into a validated
``networkx.MultiDiGraph`` and provides the connectivity / degree statistics used
in ``notebooks/03_road_network_analysis.ipynb``. It is designed to be reused by
route validation (Member 4) and by the HMM / Viterbi map-matcher.

Data contract (identical to ``pathsnap.data.loader``)
-----------------------------------------------------
``nodes_df``  columns ``node_id, lon, lat``  -- ``node_id`` == 0-based row index in ``.nodes``
``arcs_df``   columns ``arc_id, from_node_id, to_node_id`` -- ``arc_id`` == 0-based row index in ``.arcs``

Arc IDs are only meaningful inside their own trip; this module never re-indexes them.

Graph contract (returned by :func:`build_road_graph`)
-----------------------------------------------------
* ``G`` is a ``nx.MultiDiGraph`` (direction is preserved, parallel arcs are kept).
* Every row of ``.nodes`` is a graph node ``node_id`` with attributes
  ``longitude`` and ``latitude`` (WGS84 degrees).
* Every arc whose two endpoints exist in ``.nodes`` is an edge
  ``(from_node_id, to_node_id, key=arc_id)`` with attributes
  ``arc_id``, ``from_node_id``, ``to_node_id`` and ``length_m`` (haversine metres).
* Arcs that reference a non-existent node cannot be represented in the graph; they are
  *not* silently dropped -- they are reported in the arc-validation table
  (``is_valid == False``) and counted in the summary.
* ``G.graph["arc_endpoints"]`` maps ``arc_id -> (from_node_id, to_node_id)`` for every
  arc in the graph, so an edge can be fetched by arc ID in O(1):
  ``u, v = G.graph["arc_endpoints"][arc_id]; G.edges[u, v, arc_id]`` (see :func:`get_edge_by_arc_id`).

Raw-file access re-uses Member 1's readers (``pathsnap.data.loader._read_nodes`` /
``_read_arcs``). ``load_trip`` itself is not used for the EDA because it raises on
invalid node references (we need to *report* them) and also parses the GPS track,
which this analysis does not need.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from pathsnap.data.loader import _read_arcs, _read_nodes, discover_trips

EARTH_RADIUS_M = 6_371_008.8

__all__ = [
    "scan_raw_file",
    "load_road_network",
    "validate_nodes",
    "validate_arcs",
    "build_road_graph",
    "get_edge_by_arc_id",
    "find_isolated_nodes",
    "calculate_node_degrees",
    "calculate_connectivity_statistics",
    "get_road_segment_geometry",
    "haversine_m",
    "load_split",
    "analyse_trip",
]


# ---------------------------------------------------------------------------
# Raw file loading & integrity checks
# ---------------------------------------------------------------------------
def scan_raw_file(path: Path | str, expected_cols: int, integer: bool = False, max_examples: int = 5) -> Dict[str, Any]:
    """Scan a raw whitespace-separated file line by line for structural problems.

    Args:
        path: file to scan (``.nodes`` or ``.arcs``).
        expected_cols: number of whitespace-separated fields every row must have.
        integer: if True every field must parse as an integer, else as a float.
        max_examples: number of offending (1-based) line numbers to keep as examples.

    Returns:
        dict with ``exists``, ``total_lines``, ``blank_lines``, ``wrong_field_count``,
        ``non_numeric_rows``, ``malformed_rows`` (sum of the two) and ``malformed_examples``.
    """
    path = Path(path)
    report: Dict[str, Any] = {
        "exists": path.exists(),
        "total_lines": 0,
        "blank_lines": 0,
        "wrong_field_count": 0,
        "non_numeric_rows": 0,
        "malformed_rows": 0,
        "malformed_examples": [],
    }
    if not path.exists():
        return report
    caster = int if integer else float
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            report["total_lines"] += 1
            parts = line.split()
            if not parts:
                report["blank_lines"] += 1
                continue
            bad = False
            if len(parts) != expected_cols:
                report["wrong_field_count"] += 1
                bad = True
            else:
                try:
                    for p in parts:
                        caster(p)
                except ValueError:
                    report["non_numeric_rows"] += 1
                    bad = True
            if bad and len(report["malformed_examples"]) < max_examples:
                report["malformed_examples"].append(lineno)
    report["malformed_rows"] = report["wrong_field_count"] + report["non_numeric_rows"]
    return report


def _tolerant_read(path: Path, names: List[str], integer: bool) -> pd.DataFrame:
    """Fallback parser used only if Member 1's strict reader fails on a malformed file.

    Malformed values become NaN (rows are kept so that row index == original ID).
    Rows with the wrong number of fields are padded / truncated for the same reason.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue  # pandas' whitespace reader also skips blank lines
            parts = (parts + [None] * len(names))[: len(names)]
            rows.append(parts)
    df = pd.DataFrame(rows, columns=names)
    for c in names:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if integer:
            df[c] = df[c].astype("Int64")
    return df


def load_road_network(
    trip_id: str,
    trips_index: Optional[Dict[str, Dict[str, Any]]] = None,
    raw_dir: Optional[Path | str] = None,
    scan: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
    """Load ``.nodes`` and ``.arcs`` for one trip using Member 1's readers.

    Args:
        trip_id: trip identifier (e.g. ``"00000001"``).
        trips_index: output of ``pathsnap.data.loader.discover_trips`` (pass it in to avoid
            re-scanning the raw directory for every trip).
        raw_dir: raw data directory, used only when ``trips_index`` is None.
        scan: also run :func:`scan_raw_file` on both files (malformed-row check).

    Returns:
        ``(nodes_df, arcs_df, file_report)``. A missing / unreadable file yields ``None``
        for that table and a description in ``file_report["errors"]`` -- it never raises.
    """
    if trips_index is None:
        trips_index = discover_trips(raw_dir)
    report: Dict[str, Any] = {"trip_id": trip_id, "errors": [], "reader": {}}
    info = trips_index.get(trip_id)
    if info is None:
        report["errors"].append("trip not found in raw directory")
        return None, None, report

    tables: Dict[str, Optional[pd.DataFrame]] = {}
    specs = {
        "nodes": (_read_nodes, ["lon", "lat"], False, "node_id"),
        "arcs": (_read_arcs, ["from_node_id", "to_node_id"], True, "arc_id"),
    }
    for kind, (reader, names, integer, id_col) in specs.items():
        path = info.get(kind)
        if path is None or not Path(path).exists():
            report["errors"].append(f"missing .{kind} file")
            report[f"{kind}_scan"] = scan_raw_file(Path(str(path)), 2) if path else {"exists": False}
            tables[kind] = None
            continue
        malformed = False
        if scan:
            report[f"{kind}_scan"] = scan_raw_file(path, expected_cols=2, integer=integer)
            malformed = report[f"{kind}_scan"]["malformed_rows"] > 0
        try:
            if malformed:
                # pandas can silently mis-parse rows with extra fields (uses them as an index),
                # so never trust the strict reader on a file known to be malformed.
                raise ValueError(f"{report[f'{kind}_scan']['malformed_rows']} malformed rows detected by scan")
            tables[kind] = reader(Path(path))
            report["reader"][kind] = "pathsnap.data.loader"
        except Exception as exc:  # malformed file -> tolerant fallback, keep original row indices
            report["errors"].append(f".{kind} strict reader failed: {exc}")
            try:
                df = _tolerant_read(Path(path), names, integer)
                df.insert(0, id_col, np.arange(len(df), dtype=np.int64))
                tables[kind] = df
                report["reader"][kind] = "tolerant_fallback"
            except Exception as exc2:
                report["errors"].append(f".{kind} unreadable: {exc2}")
                tables[kind] = None
    return tables["nodes"], tables["arcs"], report


def load_split(split_file: Optional[Path | str] = None) -> Dict[str, List[str]]:
    """Read ``configs/split.json`` (trip IDs per partition) via the project path config."""
    if split_file is None:
        from pathsnap.utils.config import get_path

        split_file = get_path("split_file", default="configs/split.json")
    with open(split_file, "r", encoding="utf-8") as fh:
        return {k: [str(t) for t in v] for k, v in json.load(fh).items()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_nodes(nodes_df: pd.DataFrame) -> Dict[str, Any]:
    """Data-quality checks on a nodes table (columns ``node_id, lon, lat``).

    Returns a dict with counts of missing coordinates, invalid coordinates
    (non-finite, |lon|>180, |lat|>90 or exactly (0, 0)), duplicate nodes
    (nodes whose exact coordinate pair already appeared at a lower node_id),
    dtypes and the bounding box.
    """
    lon = pd.to_numeric(nodes_df["lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(nodes_df["lat"], errors="coerce").to_numpy(dtype=float)
    missing = np.isnan(lon) | np.isnan(lat)
    out_of_range = (~missing) & ((np.abs(lon) > 180) | (np.abs(lat) > 90) | ~np.isfinite(lon) | ~np.isfinite(lat))
    null_island = (~missing) & (lon == 0) & (lat == 0)
    invalid = out_of_range | null_island
    dup_mask = nodes_df.duplicated(subset=["lon", "lat"], keep="first").to_numpy() & ~missing
    ok = ~(missing | invalid)
    return {
        "n_nodes": int(len(nodes_df)),
        "lon_dtype": str(nodes_df["lon"].dtype),
        "lat_dtype": str(nodes_df["lat"].dtype),
        "missing_coordinates": int(missing.sum()),
        "invalid_coordinates": int(invalid.sum()),
        "duplicate_nodes": int(dup_mask.sum()),
        "duplicate_node_ids": nodes_df.loc[dup_mask, "node_id"].to_numpy(),
        "min_lon": float(np.min(lon[ok])) if ok.any() else np.nan,
        "max_lon": float(np.max(lon[ok])) if ok.any() else np.nan,
        "min_lat": float(np.min(lat[ok])) if ok.any() else np.nan,
        "max_lat": float(np.max(lat[ok])) if ok.any() else np.nan,
    }


def validate_arcs(arcs_df: pd.DataFrame, n_nodes: int, nodes_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Validate every arc and return one row per arc (original ``arc_id`` preserved).

    Args:
        arcs_df: columns ``arc_id, from_node_id, to_node_id``.
        n_nodes: number of rows in the trip's ``.nodes`` file (valid IDs are ``0..n_nodes-1``).
        nodes_df: optional; if given, endpoints whose node has missing/invalid coordinates are flagged.

    Returns:
        DataFrame with columns

        ``arc_id, from_node_id, to_node_id`` -- original values,
        ``from_valid, to_valid`` -- endpoint exists in ``.nodes``,
        ``is_self_loop`` -- ``from == to``,
        ``parallel_count`` -- number of arcs with the same ordered (from, to) pair,
        ``is_parallel`` -- ``parallel_count > 1`` (part of a multi-edge group),
        ``is_duplicate`` -- repeats a directed connection already seen at a lower arc_id,
        ``has_reverse`` -- an arc ``to -> from`` also exists (two-way street),
        ``endpoint_bad_coords`` -- an endpoint has missing/invalid coordinates,
        ``is_valid`` -- both endpoints exist (the arc can be placed in the graph),
        ``issues`` -- ``;``-separated list of flags (empty string if none).

    Nothing is deleted: the output has exactly ``len(arcs_df)`` rows in the original order.
    """
    f = pd.to_numeric(arcs_df["from_node_id"], errors="coerce")
    t = pd.to_numeric(arcs_df["to_node_id"], errors="coerce")
    out = pd.DataFrame(
        {
            "arc_id": arcs_df["arc_id"].to_numpy(),
            "from_node_id": arcs_df["from_node_id"].to_numpy(),
            "to_node_id": arcs_df["to_node_id"].to_numpy(),
        }
    )
    out["from_valid"] = (f.notna() & (f >= 0) & (f < n_nodes)).to_numpy()
    out["to_valid"] = (t.notna() & (t >= 0) & (t < n_nodes)).to_numpy()
    out["is_self_loop"] = (f == t).fillna(False).to_numpy(dtype=bool)

    pair = pd.DataFrame({"f": f, "t": t})
    out["parallel_count"] = pair.groupby(["f", "t"], dropna=False)["f"].transform("size").to_numpy()
    out["is_parallel"] = out["parallel_count"] > 1
    out["is_duplicate"] = pair.duplicated(keep="first").to_numpy()

    fwd = pd.MultiIndex.from_arrays([f, t])
    rev = pd.MultiIndex.from_arrays([t, f])
    out["has_reverse"] = rev.isin(fwd) & ~out["is_self_loop"].to_numpy()

    bad_coord = np.zeros(len(out), dtype=bool)
    if nodes_df is not None:
        lon = pd.to_numeric(nodes_df["lon"], errors="coerce").to_numpy(dtype=float)
        lat = pd.to_numeric(nodes_df["lat"], errors="coerce").to_numpy(dtype=float)
        node_bad = ~np.isfinite(lon) | ~np.isfinite(lat) | (np.abs(lon) > 180) | (np.abs(lat) > 90) | ((lon == 0) & (lat == 0))
        for col, valid in (("from_node_id", "from_valid"), ("to_node_id", "to_valid")):
            idx = out.loc[out[valid], col].astype(np.int64).to_numpy()
            bad_coord[out[valid].to_numpy()] |= node_bad[idx]
    out["endpoint_bad_coords"] = bad_coord
    out["is_valid"] = out["from_valid"] & out["to_valid"]

    flags = {
        "invalid_from_node": ~out["from_valid"],
        "invalid_to_node": ~out["to_valid"],
        "self_loop": out["is_self_loop"],
        "duplicate_connection": out["is_duplicate"],
        "endpoint_bad_coords": out["endpoint_bad_coords"],
    }
    issues = np.full(len(out), "", dtype=object)
    for name, mask in flags.items():
        m = mask.to_numpy() if hasattr(mask, "to_numpy") else mask
        issues[m] = np.where(issues[m] == "", name, issues[m] + ";" + name)
    out["issues"] = issues
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Vectorised great-circle distance in metres between WGS84 points."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(a, dtype=float)) for a in (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def get_road_segment_geometry(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    arc_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Return the straight-line geometry of road segments (arcs).

    Args:
        nodes_df: columns ``node_id, lon, lat``.
        arcs_df: columns ``arc_id, from_node_id, to_node_id``.
        arc_ids: optional subset of original arc IDs (order is preserved, repeats allowed --
            e.g. pass a ``.route`` sequence). Default: all arcs.

    Returns:
        DataFrame ``arc_id, from_node_id, to_node_id, from_lon, from_lat, to_lon, to_lat,
        length_m, bearing_deg``. Arcs with an invalid endpoint get NaN coordinates.
    """
    if arc_ids is None:
        sub = arcs_df[["arc_id", "from_node_id", "to_node_id"]].reset_index(drop=True)
    else:
        ids = np.asarray(list(arc_ids), dtype=np.int64)
        if ((ids < 0) | (ids >= len(arcs_df))).any():
            raise IndexError("arc_ids contains IDs outside the arcs table range")
        # arc_id == row index, so positional indexing preserves original IDs
        sub = arcs_df.iloc[ids][["arc_id", "from_node_id", "to_node_id"]].reset_index(drop=True)

    lon = nodes_df["lon"].to_numpy(dtype=float)
    lat = nodes_df["lat"].to_numpy(dtype=float)
    n = len(nodes_df)

    def _coords(col):
        ids = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(ids) & (ids >= 0) & (ids < n)
        lo = np.full(len(ids), np.nan)
        la = np.full(len(ids), np.nan)
        lo[ok] = lon[ids[ok].astype(np.int64)]
        la[ok] = lat[ids[ok].astype(np.int64)]
        return lo, la

    flo, fla = _coords("from_node_id")
    tlo, tla = _coords("to_node_id")
    out = sub.copy()
    out["from_lon"], out["from_lat"], out["to_lon"], out["to_lat"] = flo, fla, tlo, tla
    out["length_m"] = haversine_m(flo, fla, tlo, tla)
    y = np.sin(np.radians(tlo - flo)) * np.cos(np.radians(tla))
    x = np.cos(np.radians(fla)) * np.sin(np.radians(tla)) - np.sin(np.radians(fla)) * np.cos(np.radians(tla)) * np.cos(np.radians(tlo - flo))
    out["bearing_deg"] = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return out


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_road_graph(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    include_self_loops: bool = True,
) -> Tuple[nx.MultiDiGraph, pd.DataFrame, Dict[str, Any]]:
    """Build a directed multigraph of the road network of ONE trip.

    Args:
        nodes_df: ``node_id, lon, lat`` (``node_id`` = row index of ``.nodes``).
        arcs_df: ``arc_id, from_node_id, to_node_id`` (``arc_id`` = row index of ``.arcs``).
        include_self_loops: keep ``u -> u`` arcs as edges (default True so that every
            arc that can appear in a ``.route`` is present in the graph).

    Returns:
        ``(G, arc_validation, summary)``

        * ``G`` -- ``nx.MultiDiGraph``; nodes carry ``longitude``/``latitude``; each edge is
          ``(from_node_id, to_node_id, key=arc_id)`` with attributes ``arc_id``,
          ``from_node_id``, ``to_node_id`` and ``length_m``. ``G.graph["arc_endpoints"]``
          maps every arc_id in the graph to ``(u, v)``.
        * ``arc_validation`` -- output of :func:`validate_arcs` plus ``in_graph`` (bool).
        * ``summary`` -- dict of network statistics (node / arc / edge counts and
          validation counts). Connectivity statistics are computed separately by
          :func:`calculate_connectivity_statistics` because they are more expensive.
    """
    n_nodes = len(nodes_df)
    node_report = validate_nodes(nodes_df)
    arc_val = validate_arcs(arcs_df, n_nodes, nodes_df)

    G = nx.MultiDiGraph()
    lon = nodes_df["lon"].to_numpy(dtype=float)
    lat = nodes_df["lat"].to_numpy(dtype=float)
    node_ids = nodes_df["node_id"].to_numpy(dtype=np.int64)
    G.add_nodes_from(
        (int(i), {"longitude": float(x), "latitude": float(y)}) for i, x, y in zip(node_ids, lon, lat)
    )

    in_graph = arc_val["is_valid"].to_numpy().copy()
    if not include_self_loops:
        in_graph &= ~arc_val["is_self_loop"].to_numpy()
    arc_val["in_graph"] = in_graph

    sel = arc_val.loc[in_graph, ["arc_id", "from_node_id", "to_node_id"]]
    u = sel["from_node_id"].to_numpy(dtype=np.int64)
    v = sel["to_node_id"].to_numpy(dtype=np.int64)
    a = sel["arc_id"].to_numpy(dtype=np.int64)
    length = haversine_m(lon[u], lat[u], lon[v], lat[v])
    G.add_edges_from(
        (int(uu), int(vv), int(aa), {"arc_id": int(aa), "from_node_id": int(uu), "to_node_id": int(vv), "length_m": float(ll)})
        for uu, vv, aa, ll in zip(u, v, a, length)
    )
    G.graph["arc_endpoints"] = {int(aa): (int(uu), int(vv)) for uu, vv, aa in zip(u, v, a)}
    G.graph["n_arcs_in_file"] = int(len(arcs_df))

    summary = {
        "n_nodes": int(n_nodes),
        "n_arcs": int(len(arcs_df)),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "missing_coordinates": node_report["missing_coordinates"],
        "invalid_coordinates": node_report["invalid_coordinates"],
        "duplicate_nodes": node_report["duplicate_nodes"],
        "invalid_node_references": int((~arc_val["from_valid"]).sum() + (~arc_val["to_valid"]).sum()),
        "invalid_arcs": int((~arc_val["is_valid"]).sum()),
        "self_loops": int(arc_val["is_self_loop"].sum()),
        "duplicate_arcs": int(arc_val["is_duplicate"].sum()),
        "parallel_edge_groups": int(arc_val.loc[arc_val["is_parallel"]].drop_duplicates(["from_node_id", "to_node_id"]).shape[0]),
        "arcs_in_parallel_groups": int(arc_val["is_parallel"].sum()),
        "two_way_arc_share": float(arc_val["has_reverse"].mean()) if len(arc_val) else np.nan,
        "total_length_km": float(length.sum() / 1000.0),
        "median_arc_length_m": float(np.median(length)) if len(length) else np.nan,
        "zero_length_arcs": int((length == 0).sum()),
        "min_lon": node_report["min_lon"],
        "max_lon": node_report["max_lon"],
        "min_lat": node_report["min_lat"],
        "max_lat": node_report["max_lat"],
        "lon_dtype": node_report["lon_dtype"],
        "lat_dtype": node_report["lat_dtype"],
        "arc_dtype": str(arcs_df["from_node_id"].dtype),
    }
    return G, arc_val, summary


def get_edge_by_arc_id(G: nx.MultiDiGraph, arc_id: int) -> Tuple[int, int, Dict[str, Any]]:
    """Return ``(u, v, edge_attributes)`` for an original arc_id (raises KeyError if absent)."""
    u, v = G.graph["arc_endpoints"][int(arc_id)]
    return u, v, G.edges[u, v, int(arc_id)]


# ---------------------------------------------------------------------------
# Degrees, isolated nodes, connectivity
# ---------------------------------------------------------------------------
def calculate_node_degrees(G: nx.MultiDiGraph) -> pd.DataFrame:
    """Per-node degrees. Parallel arcs count separately; a self-loop adds 1 to in and 1 to out.

    Returns DataFrame ``node_id, in_degree, out_degree, total_degree`` sorted by node_id.
    """
    nodes = np.fromiter(G.nodes(), dtype=np.int64, count=G.number_of_nodes())
    ind = dict(G.in_degree())
    outd = dict(G.out_degree())
    df = pd.DataFrame(
        {
            "node_id": nodes,
            "in_degree": np.fromiter((ind[n] for n in nodes), dtype=np.int64, count=len(nodes)),
            "out_degree": np.fromiter((outd[n] for n in nodes), dtype=np.int64, count=len(nodes)),
        }
    )
    df["total_degree"] = df["in_degree"] + df["out_degree"]
    return df.sort_values("node_id").reset_index(drop=True)


def find_isolated_nodes(G: nx.MultiDiGraph, degrees: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Classify nodes that cannot be traversed normally.

    Categories (mutually exclusive):

    * ``isolated``     -- total degree 0 (no arc touches the node).
    * ``source_only``  -- in-degree 0, out-degree > 0 (can leave, never reach: dangling start).
    * ``sink_only``    -- out-degree 0, in-degree > 0 (can reach, never leave: dead end / dangling end).

    Returns DataFrame ``node_id, longitude, latitude, in_degree, out_degree, total_degree, category``.
    Nodes with in > 0 and out > 0 are not returned.
    """
    if degrees is None:
        degrees = calculate_node_degrees(G)
    d = degrees
    cat = np.select(
        [d["total_degree"] == 0, d["in_degree"] == 0, d["out_degree"] == 0],
        ["isolated", "source_only", "sink_only"],
        default="",
    )
    res = d.loc[cat != ""].copy()
    res["category"] = cat[cat != ""]
    res["longitude"] = [G.nodes[n]["longitude"] for n in res["node_id"]]
    res["latitude"] = [G.nodes[n]["latitude"] for n in res["node_id"]]
    return res[["node_id", "longitude", "latitude", "in_degree", "out_degree", "total_degree", "category"]].reset_index(drop=True)


def calculate_connectivity_statistics(G: nx.MultiDiGraph, return_sizes: bool = False):
    """Weak / strong connectivity statistics of a directed road graph.

    * Weakly connected component (WCC): nodes connected when arc direction is ignored.
    * Strongly connected component (SCC): every node can reach every other node
      *respecting* arc direction -- the part of the network where any route is drivable.

    Returns:
        dict of statistics; if ``return_sizes`` also a dict with sorted (descending)
        numpy arrays ``wcc_sizes`` and ``scc_sizes``, and the node sets of the largest
        WCC / SCC (``largest_wcc``, ``largest_scc``).
    """
    n = G.number_of_nodes()
    wccs = list(nx.weakly_connected_components(G))
    sccs = list(nx.strongly_connected_components(G))
    wcc_sizes = np.sort(np.fromiter((len(c) for c in wccs), dtype=np.int64))[::-1]
    scc_sizes = np.sort(np.fromiter((len(c) for c in sccs), dtype=np.int64))[::-1]
    largest_wcc = max(wccs, key=len) if wccs else set()
    largest_scc = max(sccs, key=len) if sccs else set()
    stats = {
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "n_weak_components": len(wccs),
        "n_strong_components": len(sccs),
        "largest_wcc_size": int(wcc_sizes[0]) if n else 0,
        "largest_scc_size": int(scc_sizes[0]) if n else 0,
        "largest_wcc_pct": 100.0 * wcc_sizes[0] / n if n else np.nan,
        "largest_scc_pct": 100.0 * scc_sizes[0] / n if n else np.nan,
        "singleton_wccs": int((wcc_sizes == 1).sum()),
        "singleton_sccs": int((scc_sizes == 1).sum()),
        "second_wcc_size": int(wcc_sizes[1]) if len(wcc_sizes) > 1 else 0,
        "is_weakly_connected": len(wccs) == 1,
        "is_strongly_connected": len(sccs) == 1,
    }
    if return_sizes:
        return stats, {
            "wcc_sizes": wcc_sizes,
            "scc_sizes": scc_sizes,
            "largest_wcc": largest_wcc,
            "largest_scc": largest_scc,
        }
    return stats


# ---------------------------------------------------------------------------
# One-call per-trip analysis (used by the notebook's loop over all trips)
# ---------------------------------------------------------------------------
def analyse_trip(
    trip_id: str,
    partition: str,
    trips_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Load, validate, build and analyse one trip, returning only small, aggregated results.

    The graph is built and discarded inside the call so that only one trip's graph is in
    memory at a time.

    Returns dict with keys ``summary`` (row for road_network_summary.csv),
    ``connectivity`` (row for connectivity_summary.csv), ``flagged_arcs`` (arcs with any issue),
    ``dangling_nodes`` (output of :func:`find_isolated_nodes`), ``degree_counts``
    (Counter per degree type), ``wcc_sizes``, ``scc_sizes``, ``nodes_head``, ``arcs_head``.
    """
    nodes_df, arcs_df, frep = load_road_network(trip_id, trips_index)
    base = {"trip_id": trip_id, "partition": partition}
    ns, as_ = frep.get("nodes_scan", {}), frep.get("arcs_scan", {})
    file_cols = {
        "nodes_file_lines": ns.get("total_lines", 0),
        "arcs_file_lines": as_.get("total_lines", 0),
        "malformed_node_rows": ns.get("malformed_rows", 0),
        "malformed_arc_rows": as_.get("malformed_rows", 0),
        "blank_lines": ns.get("blank_lines", 0) + as_.get("blank_lines", 0),
        "load_errors": "; ".join(frep["errors"]),
    }
    if nodes_df is None or arcs_df is None:
        return {"summary": {**base, **file_cols}, "connectivity": base, "ok": False}

    G, arc_val, summ = build_road_graph(nodes_df, arcs_df)
    degrees = calculate_node_degrees(G)
    dangling = find_isolated_nodes(G, degrees)
    conn, sizes = calculate_connectivity_statistics(G, return_sizes=True)

    summary = {
        **base,
        **{k: summ[k] for k in ("n_nodes", "n_arcs")},
        "missing_coordinates": summ["missing_coordinates"],
        "invalid_coordinates": summ["invalid_coordinates"],
        "invalid_node_references": summ["invalid_node_references"],
        "invalid_arcs": summ["invalid_arcs"],
        "duplicate_nodes": summ["duplicate_nodes"],
        "duplicate_arcs": summ["duplicate_arcs"],
        "parallel_edge_groups": summ["parallel_edge_groups"],
        "self_loops": summ["self_loops"],
        "zero_length_arcs": summ["zero_length_arcs"],
        "two_way_arc_share": round(summ["two_way_arc_share"], 4),
        "graph_nodes": summ["graph_nodes"],
        "graph_edges": summ["graph_edges"],
        "total_length_km": round(summ["total_length_km"], 3),
        "median_arc_length_m": round(summ["median_arc_length_m"], 2),
        "min_lon": summ["min_lon"],
        "max_lon": summ["max_lon"],
        "min_lat": summ["min_lat"],
        "max_lat": summ["max_lat"],
        "lon_dtype": summ["lon_dtype"],
        "lat_dtype": summ["lat_dtype"],
        "arc_dtype": summ["arc_dtype"],
        **file_cols,
    }
    cats = dangling["category"].value_counts()
    connectivity = {
        **base,
        **{k: v for k, v in conn.items()},
        "largest_wcc_pct": round(conn["largest_wcc_pct"], 3),
        "largest_scc_pct": round(conn["largest_scc_pct"], 3),
        "isolated_nodes": int(cats.get("isolated", 0)),
        "source_only_nodes": int(cats.get("source_only", 0)),
        "sink_only_nodes": int(cats.get("sink_only", 0)),
        "mean_total_degree": round(float(degrees["total_degree"].mean()), 4),
        "max_total_degree": int(degrees["total_degree"].max()),
    }
    flagged = arc_val.loc[arc_val["issues"] != ""].copy()
    flagged.insert(0, "trip_id", trip_id)
    dangling.insert(0, "trip_id", trip_id)
    result = {
        "summary": summary,
        "connectivity": connectivity,
        "flagged_arcs": flagged,
        "dangling_nodes": dangling,
        "degree_counts": {
            "in_degree": Counter(degrees["in_degree"].tolist()),
            "out_degree": Counter(degrees["out_degree"].tolist()),
            "total_degree": Counter(degrees["total_degree"].tolist()),
        },
        "wcc_sizes": sizes["wcc_sizes"],
        "scc_sizes": sizes["scc_sizes"],
        "nodes_head": nodes_df.head(5).assign(trip_id=trip_id),
        "arcs_head": arcs_df.head(5).assign(trip_id=trip_id),
        "ok": True,
    }
    del G, arc_val, degrees
    return result
