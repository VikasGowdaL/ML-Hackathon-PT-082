"""Per-observation sequence features and per-candidate rank features (Milestone 2, Person 4).

Per observation (one row per GPS point, from the trip's projected track ``track_utm``):
time gaps and distances to the previous / next point, speeds, acceleration, heading (shared
definition with the geometry features), heading change, gap flags and trip-segment ids.

Per candidate: distance rank of the candidate among the competitors of the same observation.

Gap policy (proposed by Person 4, to be confirmed with Person 1; see ``configs/features.yaml``):

* ``after_gap``  -- the time gap to the previous point exceeds ``flag_gap_s`` (default 10 s);
* ``segment_id`` -- a new segment starts when the gap exceeds ``split_gap_s`` (default 60 s).
  Downstream sequence models (HMM / Viterbi) should not chain transitions across segments.
* ``implausible_speed`` -- implied speed from the previous point exceeds ``max_speed_mps`` (default 70 m/s).

All distances are metres in the trip's UTM projection; times are seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.features.geometry import angle_diff_deg, compute_gps_heading, grid_bearing_deg

__all__ = [
    "DEFAULT_GAP_POLICY",
    "OBSERVATION_FEATURE_COLUMNS",
    "RANK_FEATURE_COLUMNS",
    "compute_observation_features",
    "compute_candidate_rank_features",
]

DEFAULT_GAP_POLICY = {"flag_gap_s": 10.0, "split_gap_s": 60.0, "max_speed_mps": 70.0, "min_move_m": 2.0}

OBSERVATION_FEATURE_COLUMNS = [
    "t",
    "dt_prev_s",
    "dt_next_s",
    "dist_prev_m",
    "dist_next_m",
    "speed_prev_mps",
    "speed_next_mps",
    "speed_mps",
    "accel_mps2",
    "heading_deg",
    "heading_valid",
    "heading_change_deg",
    "after_gap",
    "before_gap",
    "segment_id",
    "is_segment_start",
    "is_segment_end",
    "implausible_speed",
]
RANK_FEATURE_COLUMNS = ["dist_rank", "n_candidates", "dist_minus_best_m", "dist_ratio_to_best"]


def compute_observation_features(track_utm: pd.DataFrame, trip_id: str, policy: dict | None = None) -> pd.DataFrame:
    """Sequence features for every GPS point of one trip.

    Args:
        track_utm: ``x, y, t`` in metres / seconds, row index == ``obs_idx`` (Member 1's loader).
        trip_id: identifier copied into the output.
        policy: overrides for :data:`DEFAULT_GAP_POLICY`.

    Returns:
        DataFrame ``trip_id, obs_idx`` + :data:`OBSERVATION_FEATURE_COLUMNS`, one row per point.
        Neighbour-based values are NaN where the neighbour does not exist (first / last point) or
        lies across a split gap (i.e. in another segment).
    """
    p = {**DEFAULT_GAP_POLICY, **(policy or {})}
    x = track_utm["x"].to_numpy(dtype=float)
    y = track_utm["y"].to_numpy(dtype=float)
    t = track_utm["t"].to_numpy(dtype=float)
    n = len(t)

    dt = np.diff(t)                      # dt[i] = t[i+1] - t[i]
    step = np.hypot(np.diff(x), np.diff(y))
    nan1 = np.array([np.nan])
    dt_prev, dt_next = np.r_[nan1, dt], np.r_[dt, nan1]
    dist_prev, dist_next = np.r_[nan1, step], np.r_[step, nan1]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dt > 0, step / dt, np.nan)
    speed_prev, speed_next = np.r_[nan1, v], np.r_[v, nan1]

    after_gap = np.nan_to_num(dt_prev, nan=0.0) > p["flag_gap_s"]
    before_gap = np.nan_to_num(dt_next, nan=0.0) > p["flag_gap_s"]
    split = np.nan_to_num(dt_prev, nan=0.0) > p["split_gap_s"]
    segment_id = np.cumsum(split).astype(np.int32)
    is_start = np.r_[True, segment_id[1:] != segment_id[:-1]] if n else np.array([], bool)
    is_end = np.r_[segment_id[1:] != segment_id[:-1], True] if n else np.array([], bool)

    # neighbour values across a split gap are not meaningful
    for arr in (dt_prev, dist_prev, speed_prev):
        arr[is_start] = np.nan
    for arr in (dt_next, dist_next, speed_next):
        arr[is_end] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        tot_d = np.nansum(np.c_[dist_prev, dist_next], axis=1)
        tot_t = np.nansum(np.c_[dt_prev, dt_next], axis=1)
        speed = np.where(tot_t > 0, tot_d / tot_t, np.nan)
        accel = (speed_next - speed_prev) / ((dt_prev + dt_next) / 2.0)

    # heading: same definition as the geometry features (Person 3), computed per segment
    heading = np.full(n, np.nan)
    for s in np.unique(segment_id):
        m = segment_id == s
        heading[m] = compute_gps_heading(x[m], y[m], p["min_move_m"])["gps_heading_deg"].to_numpy()
    # heading change between the incoming and outgoing step directions
    b = grid_bearing_deg(np.diff(x), np.diff(y))
    b_in, b_out = np.r_[nan1, b], np.r_[b, nan1]
    moving = (np.nan_to_num(dist_prev) >= p["min_move_m"]) & (np.nan_to_num(dist_next) >= p["min_move_m"])
    heading_change = np.where(moving, angle_diff_deg(b_in, b_out), np.nan)

    return pd.DataFrame(
        {
            "trip_id": trip_id,
            "obs_idx": np.arange(n, dtype=np.int64),
            "t": t,
            "dt_prev_s": dt_prev,
            "dt_next_s": dt_next,
            "dist_prev_m": dist_prev,
            "dist_next_m": dist_next,
            "speed_prev_mps": speed_prev,
            "speed_next_mps": speed_next,
            "speed_mps": speed,
            "accel_mps2": accel,
            "heading_deg": heading,
            "heading_valid": ~np.isnan(heading),
            "heading_change_deg": heading_change,
            "after_gap": after_gap,
            "before_gap": before_gap,
            "segment_id": segment_id,
            "is_segment_start": is_start,
            "is_segment_end": is_end,
            "implausible_speed": np.nan_to_num(speed_prev, nan=0.0) > p["max_speed_mps"],
        }
    )


def compute_candidate_rank_features(candidates: pd.DataFrame, distance_col: str = "distance_m") -> pd.DataFrame:
    """Rank each candidate among the competitors of the same observation by distance.

    Adds ``dist_rank`` (1 = nearest; ties share the lowest rank), ``n_candidates``,
    ``dist_minus_best_m`` (distance minus the nearest candidate's distance) and
    ``dist_ratio_to_best`` (distance / (nearest distance + 1 m), > = 1). Row order is preserved.
    """
    out = candidates.copy()
    g = out.groupby(["trip_id", "obs_idx"], sort=False)[distance_col]
    best = g.transform("min")
    out["dist_rank"] = g.rank(method="min").astype(np.int32)
    out["n_candidates"] = g.transform("size").astype(np.int32)
    out["dist_minus_best_m"] = out[distance_col] - best
    out["dist_ratio_to_best"] = (out[distance_col] + 1.0) / (best + 1.0)
    return out
