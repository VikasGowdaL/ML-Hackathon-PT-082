"""End-to-end prediction and scoring for one trip (used by the API and scripts/evaluate_pipeline.py).

    predict_trip(trip_id)            GPS + road network -> candidates -> features -> final model probabilities
                                     -> Viterbi (Vikas, hmm/viterbi.py) -> shortest-path route reconstruction
                                     (hmm/route_reconstruction.py). The reference route is NEVER read.
    score_prediction(trip_id, pred)  Loads the reference route AFTER prediction and compares.

Leakage guarantee: predict_trip loads the trip with ``partition="evaluator_test"``, ``use_cache=False`` and
``allow_evaluator_labels=False``, so the loader does not even open the ``.route`` file (tested in
tests/test_pipeline.py by making the route reader raise). Only score_prediction reads it.

Route-reconstruction logic mirrors hmm/predict_route.py: the Viterbi chain and the route restart at
observations without candidates, after a time gap (``after_gap``) and at segment starts; consecutive matched arcs
are bridged with the shortest road path up to 2 km, otherwise a new route section starts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

from pathsnap.data.loader import discover_trips, load_trip
from pathsnap.features.assembly import build_trip_feature_table
from pathsnap.hmm.route_reconstruction import reconstruct_route
from pathsnap.hmm.viterbi import ViterbiConfig, decode_viterbi
from pathsnap.models.harness import FEATURES
from pathsnap.road_network import build_road_graph, haversine_m
from pathsnap.utils.config import get_raw_data_dir, get_repo_root

MAX_BRIDGE_M = 2000.0
GUARDED_PARTITION = "evaluator_test"  # loader withholds .route for this partition name


# ---------------------------------------------------------------------------
# Model and configuration
# ---------------------------------------------------------------------------
def model_path() -> Path:
    return get_repo_root() / "artifacts" / "models" / "final_model.joblib"


@lru_cache(maxsize=1)
def load_model() -> Any:
    """The exported final model (scripts/export_final_model.py). Cached after the first call."""
    p = model_path()
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run: python scripts/export_final_model.py")
    return joblib.load(p)


def model_info() -> Dict[str, Any]:
    meta = model_path().with_suffix(".json")
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}


@lru_cache(maxsize=1)
def feature_config() -> Dict[str, Any]:
    return yaml.safe_load((get_repo_root() / "configs" / "features.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def trips_index() -> Dict[str, Dict[str, Any]]:
    return discover_trips(get_raw_data_dir())


def refresh_trips_index() -> None:
    trips_index.cache_clear()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    trip_id: str
    n_observations: int
    observations: pd.DataFrame           # obs_idx, lon, lat, t, top1_arc_id, predicted_arc_id, status
    sections: List[List[int]]            # route sections, each a list of arc_ids in travel order
    breaks: List[Dict[str, int]]         # consecutive matched arcs that could not be bridged
    bridge_arcs: int
    bridge_length_m: float
    timing_s: Dict[str, float] = field(default_factory=dict)
    nodes: Optional[pd.DataFrame] = None  # node_id, lon, lat (for geometry)
    arcs: Optional[pd.DataFrame] = None   # arc_id, from_node_id, to_node_id

    @property
    def route_arcs(self) -> List[int]:
        return [a for s in self.sections for a in s]


def _reconstruct(decoded: pd.DataFrame, reset: np.ndarray, graph) -> tuple:
    """Same section / bridging logic as hmm/predict_route.py."""
    sections: List[List[int]] = []
    breaks: List[Dict[str, int]] = []
    inserted, bridge_len = 0, 0.0
    prev_arc = prev_obs = None
    for row in decoded.sort_values("obs_idx").itertuples(index=False):
        obs, val = int(row.obs_idx), row.predicted_arc_id
        if pd.isna(val):
            prev_arc = prev_obs = None
            continue
        arc = int(val)
        if prev_arc is None or prev_obs is None or obs != prev_obs + 1 or reset[obs]:
            sections.append([arc])
        else:
            res = reconstruct_route([prev_arc, arc], graph, max_bridge_m=MAX_BRIDGE_M)
            if res.breaks:
                breaks.append({"previous_obs": prev_obs, "next_obs": obs, "previous_arc": prev_arc, "next_arc": arc})
                sections.append([arc])
            else:
                sections[-1].extend(res.route_arcs[1:-1])
                if arc != prev_arc:
                    sections[-1].append(arc)
                inserted += res.bridges_inserted
                bridge_len += res.total_bridge_length_m
        prev_arc, prev_obs = arc, obs
    return sections, breaks, inserted, bridge_len


def predict_trip(trip_id: str, model: Any = None, viterbi_config: Optional[ViterbiConfig] = None) -> Prediction:
    """Predict the travelled route of one trip from its GPS track and road network only."""
    timing: Dict[str, float] = {}
    t0 = time.perf_counter()
    idx = trips_index()
    if trip_id not in idx:
        raise KeyError(f"Trip '{trip_id}' not found in {get_raw_data_dir()}")
    trip = load_trip(trip_id, trips_index=idx, partition=GUARDED_PARTITION, allow_evaluator_labels=False,
                     use_cache=False, project_utm=True)
    trip = replace(trip, route=None)  # belt and braces: nothing downstream can see a route
    timing["load_s"] = time.perf_counter() - t0

    cfg = feature_config()
    t0 = time.perf_counter()
    feats = build_trip_feature_table(trip, radius_m=cfg["radius_m"], max_candidates=cfg["max_candidates"],
                                     gap_policy=cfg["gap_policy"])
    timing["features_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = model or load_model()
    prob = model.predict_proba(feats[FEATURES].astype(np.float32).to_numpy())[:, 1] if len(feats) else np.array([])
    timing["model_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_obs = len(trip.track)
    cand = pd.DataFrame({"obs_idx": feats["obs_idx"].to_numpy(), "arc_id": feats["arc_id"].to_numpy(), "probability": prob})
    reset = np.zeros(n_obs, dtype=bool)
    if len(feats):
        flags = feats.groupby("obs_idx")[["after_gap", "is_segment_start"]].max()
        reset[flags.index.to_numpy()] = (flags["after_gap"] | flags["is_segment_start"]).to_numpy()
    decoded = decode_viterbi(cand, trip.arcs, n_observations=n_obs, reset_at=reset, config=viterbi_config or ViterbiConfig())
    top1 = cand.loc[cand.groupby("obs_idx")["probability"].idxmax(), ["obs_idx", "arc_id"]].rename(columns={"arc_id": "top1_arc_id"}) \
        if len(cand) else pd.DataFrame(columns=["obs_idx", "top1_arc_id"])
    timing["viterbi_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    graph, _, _ = build_road_graph(trip.nodes, trip.arcs)
    sections, breaks, inserted, bridge_len = _reconstruct(decoded, reset, graph)
    timing["reconstruction_s"] = time.perf_counter() - t0

    obs = trip.track[["lon", "lat", "t"]].copy()
    obs.insert(0, "obs_idx", np.arange(n_obs))
    obs = obs.merge(top1, on="obs_idx", how="left").merge(decoded, on="obs_idx", how="left")
    timing["total_s"] = sum(timing.values())
    return Prediction(trip_id=trip_id, n_observations=n_obs, observations=obs, sections=sections, breaks=breaks,
                      bridge_arcs=inserted, bridge_length_m=bridge_len, timing_s=timing,
                      nodes=trip.nodes[["node_id", "lon", "lat"]], arcs=trip.arcs[["arc_id", "from_node_id", "to_node_id"]])


def route_polylines(pred: Prediction) -> List[List[List[float]]]:
    """Each route section as a [[lat, lon], ...] polyline (consecutive arcs share their joining node)."""
    lon, lat = pred.nodes["lon"].to_numpy(), pred.nodes["lat"].to_numpy()
    f, t = pred.arcs["from_node_id"].to_numpy(), pred.arcs["to_node_id"].to_numpy()
    lines = []
    for sec in pred.sections:
        pts: List[List[float]] = []
        for a in sec:
            for n in (f[a], t[a]):
                p = [float(lat[n]), float(lon[n])]
                if not pts or pts[-1] != p:
                    pts.append(p)
        lines.append(pts)
    return lines


# ---------------------------------------------------------------------------
# Scoring (reads the reference route -- only after prediction)
# ---------------------------------------------------------------------------
def load_reference_route(trip_id: str, partition: str) -> np.ndarray:
    """The trip's ground-truth arc sequence. Evaluator routes are unmasked explicitly, for scoring only."""
    trip = load_trip(trip_id, trips_index=trips_index(), partition=partition, use_cache=False, project_utm=False,
                     allow_evaluator_labels=partition.startswith("evaluator"))
    if trip.route is None:
        raise ValueError(f"No reference route available for trip {trip_id}")
    return np.asarray(trip.route, dtype=np.int64)


def score_prediction(pred: Prediction, reference_route: np.ndarray) -> Dict[str, Any]:
    """Compare a prediction with the reference route (metric definitions: docs/metrics.md, 'Route reconstruction').

    * route_precision / route_recall / route_f1: over the unique arcs of the predicted and reference routes,
      weighted by arc length (how much of the predicted route is on the true route / how much of the true route
      was found).
    * arc_precision / arc_recall: the same, counted in arcs.
    * points_on_route: share of ALL GPS points whose Viterbi arc is part of the reference route
      (points without a prediction count as wrong).
    """
    f, t = pred.arcs["from_node_id"].to_numpy(), pred.arcs["to_node_id"].to_numpy()
    lon, lat = pred.nodes["lon"].to_numpy(), pred.nodes["lat"].to_numpy()
    length = haversine_m(lon[f], lat[f], lon[t], lat[t])
    P = np.array(sorted(set(pred.route_arcs)), dtype=np.int64)
    R = np.array(sorted(set(reference_route.tolist())), dtype=np.int64)
    both = np.intersect1d(P, R)
    lp, lr, lb = length[P].sum() if len(P) else 0.0, length[R].sum(), length[both].sum() if len(both) else 0.0
    prec = lb / lp if lp else 0.0
    rec = lb / lr if lr else 0.0
    ref_set = set(R.tolist())
    pa = pred.observations["predicted_arc_id"]
    on = pa.notna() & pa.fillna(-1).astype(np.int64).isin(ref_set)
    return {
        "route_precision": float(prec),
        "route_recall": float(rec),
        "route_f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "arc_precision": float(len(both) / len(P)) if len(P) else 0.0,
        "arc_recall": float(len(both) / len(R)) if len(R) else 0.0,
        "points_on_route": float(on.mean()) if len(on) else 0.0,
        "predicted_route_km": float(lp / 1000.0),
        "reference_route_km": float(lr / 1000.0),
        "route_sections": len(pred.sections),
        "unbridged_breaks": len(pred.breaks),
        "points_without_candidates": int(pa.isna().sum()),
    }
