"""Common utilities and feature loading for PathSnap model training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pathsnap.data.loader import load_trip
from pathsnap.features.candidate_labeling import label_candidates
from pathsnap.features.candidates import generate_candidates
from pathsnap.features.geometry import (
    GEOMETRY_FEATURE_COLUMNS,
    compute_arc_static_features,
    compute_geometry_features,
)
from pathsnap.features.reference_alignment import align_reference_route
from pathsnap.utils.config import get_path

# Explicit 20 feature columns for modeling
FEATURE_COLUMNS = list(GEOMETRY_FEATURE_COLUMNS)

# Identifier and non-feature columns
ID_COLUMNS = ["trip_id", "obs_idx", "arc_id"]
EXCLUDE_COLUMNS = ["trip_id", "obs_idx", "arc_id", "reference_arc_id", "label", "distance_m"]


def load_trip_feature_table(
    trip_id: str,
    partition: str = "train",
    radius_m: int = 50,
    max_candidates: int = 30,
    labeled: bool = True,
) -> pd.DataFrame:
    """
    Load or generate features (and labels if labeled=True) for a single trip.

    Uses cached candidate parquets from outputs/candidates/<radius_m>/ if available,
    otherwise generates candidates dynamically. Computes geometric static and observation
    features, and aligns reference route to produce binary labels.
    """
    trip = load_trip(trip_id, partition=partition, project_utm=True)

    cand_path = Path("outputs") / "candidates" / str(radius_m) / f"{trip_id}.parquet"
    if cand_path.exists():
        cands = pd.read_parquet(cand_path)
    else:
        cands = generate_candidates(
            trip,
            radius_m=radius_m,
            max_candidates=max_candidates,
            use_index_cache=True,
        )

    arc_static = compute_arc_static_features(trip.nodes_utm, trip.arcs)
    geom_features = compute_geometry_features(cands, trip.track_utm, arc_static)

    if labeled and partition == "train":
        alignment = align_reference_route(trip)
        result = label_candidates(geom_features, alignment)
    else:
        result = geom_features

    return result


def load_dataset(
    trip_ids: List[str],
    partition: str = "train",
    radius_m: int = 50,
    max_candidates: int = 30,
    labeled: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
    """
    Load feature matrix X, binary labels y, and metadata for a list of trips.

    Returns:
        (X, y, metadata)
        - X: DataFrame containing only FEATURE_COLUMNS
        - y: Series containing binary labels (or None if labeled=False)
        - metadata: DataFrame containing trip_id, obs_idx, arc_id
    """
    frames = []
    for tid in trip_ids:
        df = load_trip_feature_table(
            tid,
            partition=partition,
            radius_m=radius_m,
            max_candidates=max_candidates,
            labeled=labeled,
        )
        frames.append(df)

    if not frames:
        raise ValueError("No trips provided or loaded.")

    combined = pd.concat(frames, ignore_index=True)

    # Cast boolean features to float for consistent modeling
    X = combined[FEATURE_COLUMNS].copy()
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(float)

    metadata = combined[ID_COLUMNS].copy()
    y = combined["label"].astype(int) if labeled and "label" in combined.columns else None

    return X, y, metadata
