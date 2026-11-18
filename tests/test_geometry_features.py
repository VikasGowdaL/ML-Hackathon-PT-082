"""Tests for pathsnap.features.geometry (Milestone 2, Person 3 - geometry features)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathsnap.features.geometry import (
    GEOMETRY_FEATURE_COLUMNS,
    angle_diff_deg,
    compute_arc_static_features,
    compute_geometry_features,
    compute_gps_heading,
    grid_bearing_deg,
)


@pytest.fixture
def toy():
    """Metric toy network: 0 -(east 100 m)-> 1 -(north 100 m)-> 2, plus reverse 1->0 and a zero-length arc."""
    nodes_utm = pd.DataFrame({"node_id": [0, 1, 2, 3], "x": [0.0, 100.0, 100.0, 500.0], "y": [0.0, 0.0, 100.0, 500.0]})
    pairs = [(0, 1), (1, 2), (1, 0), (3, 3)]
    arcs = pd.DataFrame({"arc_id": np.arange(4), "from_node_id": [p[0] for p in pairs], "to_node_id": [p[1] for p in pairs]})
    # vehicle drives east along y=5 (5 m north of arc 0)
    track = pd.DataFrame({"x": [10.0, 30.0, 50.0, 50.5, 120.0], "y": [5.0] * 5, "t": np.arange(5.0)})
    return nodes_utm, arcs, track


def test_angles():
    assert grid_bearing_deg(0, 1) == pytest.approx(0)      # north
    assert grid_bearing_deg(1, 0) == pytest.approx(90)     # east
    assert grid_bearing_deg(0, -1) == pytest.approx(180)
    assert grid_bearing_deg(-1, 0) == pytest.approx(270)
    assert np.isnan(grid_bearing_deg(0, 0))
    assert angle_diff_deg(350, 10) == pytest.approx(20)
    assert angle_diff_deg(90, 270) == pytest.approx(180)


def test_gps_heading(toy):
    _, _, track = toy
    h = compute_gps_heading(track.x, track.y, min_move_m=2.0)
    assert np.allclose(h.gps_heading_deg, 90)  # all eastward
    assert h.heading_valid.all()
    still = compute_gps_heading([0, 0.5, 1.0], [0, 0, 0], min_move_m=2.0)
    assert not still.heading_valid.any() and still.gps_heading_deg.isna().all()


def test_arc_static(toy):
    nodes_utm, arcs, _ = toy
    s = compute_arc_static_features(nodes_utm, arcs)
    assert list(s.index) == [0, 1, 2, 3]
    assert s.loc[0, "seg_length_m"] == pytest.approx(100)
    assert s.loc[1, "arc_bearing_deg"] == pytest.approx(0)
    assert s.loc[0, "arc_bearing_deg"] == pytest.approx(90)
    assert bool(s.loc[0, "arc_is_two_way"]) and not bool(s.loc[1, "arc_is_two_way"])
    assert bool(s.loc[3, "arc_is_zero_length"])
    assert s.loc[0, "to_out_degree"] == 2  # node 1 -> 2 and 1 -> 0
    assert bool(s.loc[0, "arc_in_largest_scc"]) and not bool(s.loc[1, "arc_in_largest_scc"])


def test_candidate_features(toy):
    nodes_utm, arcs, track = toy
    s = compute_arc_static_features(nodes_utm, arcs)
    cand = pd.DataFrame({"trip_id": "x", "obs_idx": [1, 1, 1, 4, 4], "arc_id": [0, 2, 1, 3, 0], "distance": [5, 5, 70, 1, 20]})
    f = compute_geometry_features(cand, track, s)
    assert len(f) == 5 and list(f.arc_id) == [0, 2, 1, 3, 0] and "distance" in f  # order + passthrough kept
    assert set(GEOMETRY_FEATURE_COLUMNS) <= set(f.columns)
    r0, r1, r2, r3, r4 = (f.iloc[i] for i in range(5))
    assert r0.dist_to_segment_m == pytest.approx(5)
    assert r0.proj_offset_m == pytest.approx(30) and r0.proj_fraction == pytest.approx(0.3)
    assert r0.proj_remaining_m == pytest.approx(70) and not r0.proj_is_endpoint
    assert r0.bearing_diff_deg == pytest.approx(0)              # same direction
    assert r1.bearing_diff_deg == pytest.approx(180)            # reverse arc
    assert r1.bearing_diff_undirected_deg == pytest.approx(0)
    # (30,5) onto arc 1 (100,0)->(100,100): foot of perpendicular (100,5), inside the segment
    assert not r2.proj_is_endpoint and r2.dist_to_segment_m == pytest.approx(70) and r2.proj_offset_m == pytest.approx(5)
    assert r2.bearing_diff_deg == pytest.approx(90)
    assert r3.arc_is_zero_length and r3.proj_fraction == 0 and np.isnan(r3.bearing_diff_deg)
    # (120,5) beyond the end of arc 0: nearest point is node 1 (100,0)
    assert r4.proj_is_endpoint and r4.proj_fraction == 1 and r4.dist_to_segment_m == pytest.approx(np.hypot(20, 5))


def test_bad_indices(toy):
    nodes_utm, arcs, track = toy
    s = compute_arc_static_features(nodes_utm, arcs)
    with pytest.raises(IndexError):
        compute_geometry_features(pd.DataFrame({"obs_idx": [99], "arc_id": [0]}), track, s)
    with pytest.raises(IndexError):
        compute_geometry_features(pd.DataFrame({"obs_idx": [0], "arc_id": [99]}), track, s)


def test_real_trip_projection_if_present():
    """Distances/bearings come from UTM: check on one real train trip."""
    from pathsnap.data.loader import discover_trips, load_trip
    from pathsnap.features.geometry import check_projected_geometry
    from pathsnap.road_network import load_split
    from pathsnap.utils.config import get_raw_data_dir

    raw = get_raw_data_dir()
    if not raw.exists() or not list(raw.rglob("*.track")):
        pytest.skip("Real raw dataset not present")
    pytest.importorskip("pyproj")
    tid = load_split()["train"][0]
    trip = load_trip(tid, trips_index=discover_trips(raw), partition="train", use_cache=False, project_utm=True)
    chk = check_projected_geometry(trip)
    assert chk["is_utm"]
    assert abs(chk["length_ratio_median"] - 1) < 0.002     # UTM scale factor within 0.2 %
    assert chk["bearing_offset_spread_deg"] < 1.0           # grid convergence ~constant inside a trip
    s = compute_arc_static_features(trip.nodes_utm, trip.arcs)
    cand = pd.DataFrame({"obs_idx": np.zeros(5, dtype=int), "arc_id": np.arange(5)})
    f = compute_geometry_features(cand, trip.track_utm, s)
    assert (f.dist_to_segment_m >= 0).all() and f.proj_fraction.between(0, 1).all()
