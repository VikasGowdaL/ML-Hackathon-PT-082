"""Unit tests for spatial candidate arc generation (Milestone 2, Person 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathsnap.data.loader import TripData
from pathsnap.features.candidates import generate_candidates, get_spatial_index


@pytest.fixture
def synthetic_trip() -> TripData:
    """Create a controlled metric TripData fixture with known geometry and distances.

    Network geometry (UTM coordinates in meters):
      - Arc 0: (0, 0) -> (100, 0)       [along y=0]
      - Arc 1: (0, 20) -> (100, 20)     [along y=20]
      - Arc 2: (0, 50) -> (100, 50)     [along y=50]
      - Arc 3: (500, 500) -> (600, 500) [isolated far away]
      - Arc 4: (0, 0) -> (0, 0)         [zero-length arc edge case]

    GPS observations:
      - Obs 0: (50, 2)   -> distance to Arc 0 is 2m, Arc 1 is 18m, Arc 2 is 48m
      - Obs 1: (50, 25)  -> distance to Arc 1 is 5m, Arc 2 is 25m, Arc 0 is 25m
      - Obs 2: (2000, 2000) -> far away, 0 candidates within 50m
      - Obs 3: (50, 2)   -> duplicate GPS point as Obs 0 (edge case)
    """
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4, 5, 6, 7],
        "lon": [10.0] * 8,
        "lat": [50.0] * 8,
    })
    nodes_utm = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4, 5, 6, 7],
        "x": [0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 500.0, 600.0],
        "y": [0.0, 0.0, 20.0, 20.0, 50.0, 50.0, 500.0, 500.0],
    })
    arcs_df = pd.DataFrame({
        "arc_id": [0, 1, 2, 3, 4],
        "from_node_id": [0, 2, 4, 6, 0],
        "to_node_id": [1, 3, 5, 7, 0],  # Arc 4 is zero-length (0 -> 0)
    })
    track_df = pd.DataFrame({
        "lon": [10.0] * 4,
        "lat": [50.0] * 4,
        "t": [100.0, 101.0, 102.0, 103.0],
    })
    track_utm = pd.DataFrame({
        "x": [50.0, 50.0, 2000.0, 50.0],
        "y": [2.0, 25.0, 2000.0, 2.0],
        "t": [100.0, 101.0, 102.0, 103.0],
    })

    return TripData(
        trip_id="synth_001",
        partition="train",
        track=track_df,
        nodes=nodes_df,
        arcs=arcs_df,
        route=np.array([0, 1]),
        utm_crs="EPSG:32632",
        track_utm=track_utm,
        nodes_utm=nodes_utm,
    )


def test_candidate_generation_schema_and_accuracy(synthetic_trip: TripData):
    """Test candidate extraction outputs correct schema, types, and nearest distances."""
    df = generate_candidates(synthetic_trip, radius_m=30.0, max_candidates=10, use_index_cache=False)

    # Validate output columns and dtypes
    expected_cols = ["trip_id", "obs_idx", "arc_id", "distance_m"]
    assert list(df.columns) == expected_cols
    assert df["trip_id"].dtype == object or df["trip_id"].dtype == "string"
    assert df["obs_idx"].dtype == np.int64
    assert df["arc_id"].dtype == np.int64
    assert df["distance_m"].dtype == np.float64

    # Obs 0 is at (50, 2): within 30m, Arc 0 (dist 2.0m) and Arc 1 (dist 18.0m)
    obs0_cands = df[df["obs_idx"] == 0].sort_values("distance_m")
    assert len(obs0_cands) == 2
    assert obs0_cands.iloc[0]["arc_id"] == 0
    assert pytest.approx(obs0_cands.iloc[0]["distance_m"], abs=1e-3) == 2.0
    assert obs0_cands.iloc[1]["arc_id"] == 1
    assert pytest.approx(obs0_cands.iloc[1]["distance_m"], abs=1e-3) == 18.0

    # Obs 3 is duplicate of Obs 0, should match identical candidate arcs
    obs3_cands = df[df["obs_idx"] == 3].sort_values("distance_m")
    assert len(obs3_cands) == 2
    assert list(obs3_cands["arc_id"]) == list(obs0_cands["arc_id"])


def test_long_arc_midpoint_failure_case():
    """Verify STRtree succeeds on a long arc where the midpoint is far outside the radius.

    Network: One 1,000-meter long arc from (0, 0) to (1000, 0).
    Midpoint of the arc is at (500, 0).
    GPS point is at (10, 15).
    - Distance from point (10, 15) to segment is 15.0m (passes within 50m radius).
    - Distance from point (10, 15) to midpoint (500, 0) is ~490m (fails midpoint search!).
    STRtree on segment geometry MUST catch this arc.
    """
    nodes_df = pd.DataFrame({"node_id": [0, 1], "lon": [10.0, 10.0], "lat": [50.0, 50.0]})
    nodes_utm = pd.DataFrame({"node_id": [0, 1], "x": [0.0, 1000.0], "y": [0.0, 0.0]})
    arcs_df = pd.DataFrame({"arc_id": [0], "from_node_id": [0], "to_node_id": [1]})
    track_df = pd.DataFrame({"lon": [10.0], "lat": [50.0], "t": [1.0]})
    track_utm = pd.DataFrame({"x": [10.0], "y": [15.0], "t": [1.0]})

    trip = TripData(
        trip_id="long_arc_trip",
        partition="train",
        track=track_df,
        nodes=nodes_df,
        arcs=arcs_df,
        route=np.array([0]),
        utm_crs="EPSG:32632",
        track_utm=track_utm,
        nodes_utm=nodes_utm,
    )

    df = generate_candidates(trip, radius_m=50.0, max_candidates=10, use_index_cache=False)
    assert len(df) == 1
    assert df.iloc[0]["obs_idx"] == 0
    assert df.iloc[0]["arc_id"] == 0
    assert pytest.approx(df.iloc[0]["distance_m"], abs=1e-3) == 15.0


def test_cap_behavior(synthetic_trip: TripData):
    """Test candidate capping strictly retains the nearest k arcs."""
    # Radius 100m covers Arc 0 (2m), Arc 1 (18m), Arc 2 (48m), Arc 4 (50.04m) for Obs 0
    # With max_candidates = 2, only Arc 0 and Arc 1 must be kept
    df = generate_candidates(synthetic_trip, radius_m=100.0, max_candidates=2, use_index_cache=False)
    obs0_cands = df[df["obs_idx"] == 0]
    assert len(obs0_cands) == 2
    assert list(obs0_cands["arc_id"]) == [0, 1]


def test_zero_candidate_handling(synthetic_trip: TripData):
    """Test observations outside the search radius are tracked in no_candidates."""
    df = generate_candidates(synthetic_trip, radius_m=30.0, max_candidates=10, use_index_cache=False)

    # Obs 2 at (2000, 2000) has no arcs within 30m
    assert 2 in df.attrs["no_candidates"]
    assert df.attrs["n_zero_candidates"] == 1
    assert 2 not in df["obs_idx"].values
