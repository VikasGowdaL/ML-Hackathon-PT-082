"""Assemble the candidate feature table from all four Milestone 2 parts (Person 4).

Pipeline per trip (one trip in memory at a time):

1. Person 1 -- ``candidates.generate_candidates``: (obs_idx, arc_id, distance_m) within a radius, capped.
2. Person 4 -- ``sequence.compute_candidate_rank_features``: distance rank among competitors.
3. Person 3 -- ``geometry.compute_geometry_features``: per-candidate geometry + arc connectivity.
4. Person 4 -- ``sequence.compute_observation_features``: speed, heading, gaps, segments (joined on obs_idx).
5. Person 2 -- ``reference_alignment.align_reference_route`` + ``candidate_labeling.label_candidates``:
   ``reference_arc_id`` and binary ``label`` (training trips only; Person 2's alignment refuses other partitions).

Each partition is written to its own parquet file (``feature_table_<partition>.parquet``, one row group per trip),
so the training table can never contain a student-test trip. Scaling statistics are fitted on the training file only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from pathsnap.data.loader import load_trip
from pathsnap.features.candidate_labeling import label_candidates
from pathsnap.features.candidates import generate_candidates
from pathsnap.features.geometry import GEOMETRY_FEATURE_COLUMNS, compute_arc_static_features, compute_geometry_features
from pathsnap.features.reference_alignment import align_reference_route
from pathsnap.features.sequence import (
    DEFAULT_GAP_POLICY,
    OBSERVATION_FEATURE_COLUMNS,
    RANK_FEATURE_COLUMNS,
    compute_candidate_rank_features,
    compute_observation_features,
)

__all__ = [
    "KEY_COLUMNS",
    "LABEL_COLUMNS",
    "FEATURE_COLUMNS",
    "build_trip_feature_table",
    "build_feature_tables",
    "fit_scaler",
    "fit_scaler_from_parquet",
    "apply_scaler",
]

KEY_COLUMNS = ["trip_id", "partition", "obs_idx", "arc_id"]
LABEL_COLUMNS = ["reference_arc_id", "label"]
_OBS_COLS = [c for c in OBSERVATION_FEATURE_COLUMNS if c not in ("heading_deg", "heading_valid")]  # heading comes via geometry
FEATURE_COLUMNS = ["distance_m"] + RANK_FEATURE_COLUMNS + GEOMETRY_FEATURE_COLUMNS + _OBS_COLS
COLUMN_ORDER = KEY_COLUMNS + LABEL_COLUMNS + FEATURE_COLUMNS

_BOOL = {"proj_is_endpoint", "heading_valid", "arc_is_zero_length", "arc_is_two_way", "arc_in_largest_scc",
         "after_gap", "before_gap", "is_segment_start", "is_segment_end", "implausible_speed"}
_INT = {"obs_idx": "int64", "arc_id": "int64", "reference_arc_id": "Int64", "label": "Int8", "dist_rank": "int32",
        "n_candidates": "int32", "arc_parallel_count": "int32", "from_in_degree": "int32", "from_out_degree": "int32",
        "to_in_degree": "int32", "to_out_degree": "int32", "segment_id": "int32"}
_NOT_SCALED = set(KEY_COLUMNS) | set(LABEL_COLUMNS) | _BOOL | {"segment_id", "t"}
SCALE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in _NOT_SCALED]


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed column order and dtypes (float32 for continuous features) so every row group matches."""
    df = df[COLUMN_ORDER].copy()
    for c in df.columns:
        if c in ("trip_id", "partition"):
            df[c] = df[c].astype(str)
        elif c in _BOOL:
            df[c] = df[c].astype(bool)
        elif c in _INT:
            df[c] = df[c].astype(_INT[c])
        else:
            df[c] = df[c].astype("float32")
    return df


def build_trip_feature_table(
    trip: Any,
    radius_m: float = 50.0,
    max_candidates: int = 30,
    gap_policy: Optional[dict] = None,
) -> pd.DataFrame:
    """Full feature table for one trip (``TripData`` loaded with ``project_utm=True``).

    Returns one row per (obs_idx, arc_id) candidate with :data:`COLUMN_ORDER` columns. ``label`` and
    ``reference_arc_id`` are filled for training trips and null otherwise.
    """
    if trip.track_utm is None or trip.nodes_utm is None:
        raise ValueError("UTM projection is required (load_trip(..., project_utm=True)).")
    partition = str(trip.partition)

    cand = generate_candidates(trip, radius_m=radius_m, max_candidates=max_candidates, use_index_cache=False)
    cand = compute_candidate_rank_features(cand, distance_col="distance_m")

    obs = compute_observation_features(trip.track_utm, trip.trip_id, gap_policy)
    heading = obs[["obs_idx", "heading_deg", "heading_valid"]].rename(columns={"heading_deg": "gps_heading_deg"})
    arc_static = compute_arc_static_features(trip.nodes_utm, trip.arcs)
    feats = compute_geometry_features(cand, trip.track_utm, arc_static, heading=heading)
    feats = feats.merge(obs[["trip_id", "obs_idx"] + _OBS_COLS], on=["trip_id", "obs_idx"], how="left", validate="many_to_one")

    if partition == "train":
        alignment = align_reference_route(trip)
        feats = label_candidates(feats, alignment)
    else:
        feats["reference_arc_id"] = pd.NA
        feats["label"] = pd.NA
    feats["partition"] = partition
    return _enforce_schema(feats)


def build_feature_tables(
    trips_index: Dict[str, Dict[str, Any]],
    split: Dict[str, List[str]],
    out_dir: Path | str,
    partitions: Iterable[str] = ("train", "student_test"),
    radius_m: float = 50.0,
    max_candidates: int = 30,
    gap_policy: Optional[dict] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Build and write ``feature_table_<partition>.parquet`` for each partition.

    Evaluator partitions are refused. Returns a per-trip summary DataFrame (rows, observations,
    observations with candidates, positives, observations whose reference arc is among the candidates).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for part in partitions:
        if "evaluator" in part:
            raise PermissionError("Feature tables must not be built from evaluator trips.")
        path = out_dir / f"feature_table_{part}.parquet"
        writer = None
        try:
            for tid in split[part]:
                if tid not in trips_index:
                    continue
                trip = load_trip(tid, trips_index=trips_index, partition=part, project_utm=True)
                df = build_trip_feature_table(trip, radius_m, max_candidates, gap_policy)
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
                n_obs = len(trip.track)
                pos = df.groupby("obs_idx")["label"].max() if part == "train" else None
                rows.append({
                    "trip_id": tid, "partition": part, "rows": len(df), "observations": n_obs,
                    "obs_with_candidates": int(df.obs_idx.nunique()),
                    "positives": int(df.label.sum()) if part == "train" else None,
                    "obs_reference_in_candidates": int((pos == 1).sum()) if part == "train" else None,
                    "segments": int(df.segment_id.max() + 1) if len(df) else 0,
                })
                if progress:
                    print(f"{part:12s} {tid}: {len(df):>7,} rows")
                del trip, df, table
        finally:
            if writer is not None:
                writer.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scaling -- fitted on training rows only
# ---------------------------------------------------------------------------
def fit_scaler(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """Mean / std per column, fitted on a training-only DataFrame.

    Raises ``ValueError`` if any row's ``partition`` is not ``train`` (prevents leakage).
    NaNs are ignored; a zero std is stored as 1.0.
    """
    if "partition" not in df.columns or (df["partition"] != "train").any():
        raise ValueError("Scaler must be fitted on training rows only.")
    cols = columns or SCALE_COLUMNS
    stats = {}
    for c in cols:
        x = df[c].astype("float64")
        sd = float(x.std(ddof=0))
        stats[c] = {"mean": float(x.mean()), "std": sd if sd > 0 else 1.0}
    return stats


def fit_scaler_from_parquet(path: Path | str, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """Streaming version of :func:`fit_scaler` over the training parquet file (row group by row group)."""
    import pyarrow.parquet as pq

    cols = columns or SCALE_COLUMNS
    pf = pq.ParquetFile(path)
    n = {c: 0 for c in cols}
    s = {c: 0.0 for c in cols}
    s2 = {c: 0.0 for c in cols}
    for i in range(pf.num_row_groups):
        g = pf.read_row_group(i, columns=cols + ["partition"]).to_pandas()
        if (g["partition"] != "train").any():
            raise ValueError("Scaler must be fitted on training rows only.")
        for c in cols:
            x = g[c].to_numpy(dtype="float64")
            x = x[~np.isnan(x)]
            n[c] += len(x); s[c] += float(x.sum()); s2[c] += float((x * x).sum())
    stats = {}
    for c in cols:
        mean = s[c] / n[c] if n[c] else 0.0
        var = max(s2[c] / n[c] - mean * mean, 0.0) if n[c] else 0.0
        stats[c] = {"mean": mean, "std": float(np.sqrt(var)) if var > 0 else 1.0}
    return stats


def apply_scaler(df: pd.DataFrame, stats: Dict[str, Dict[str, float]], suffix: str = "") -> pd.DataFrame:
    """Return a copy with ``(x - mean) / std`` applied to the columns in ``stats`` (optionally as new ``<col><suffix>``)."""
    out = df.copy()
    for c, st in stats.items():
        out[c + suffix] = ((out[c].astype("float64") - st["mean"]) / st["std"]).astype("float32")
    return out


def save_scaler(stats: Dict[str, Dict[str, float]], path: Path | str) -> None:
    """Write scaler statistics as JSON."""
    Path(path).write_text(json.dumps({"fitted_on": "train", "columns": stats}, indent=2), encoding="utf-8")
