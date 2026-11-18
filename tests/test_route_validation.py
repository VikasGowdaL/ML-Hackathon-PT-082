"""Tests for pathsnap.route_validation (Member 4 - route checks and Folium)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathsnap import road_network as rn
from pathsnap.route_validation import (
    check_route_continuity,
    gps_to_route_distances,
    load_route_trip,
    make_route_map,
    validate_route_ids,
)


@pytest.fixture
def toy():
    """Straight east-west street 0-1-2-3 (about 71.5 m per arc at 50N) plus a detached node 4."""
    nodes = pd.DataFrame({"node_id": np.arange(5), "lon": [8.000, 8.001, 8.002, 8.003, 8.010], "lat": [50.0] * 4 + [50.01]})
    pairs = [(0, 1), (1, 2), (2, 3), (1, 0), (3, 4)]
    arcs = pd.DataFrame({"arc_id": np.arange(5), "from_node_id": [p[0] for p in pairs], "to_node_id": [p[1] for p in pairs]})
    return nodes, arcs


def test_validate_route_ids():
    r = validate_route_ids(np.array([0, 1, 1, 9, -1, 0]), n_arcs=5)
    assert r["invalid_arc_ids"] == 2 and r["invalid_positions"] == [3, 4]
    assert r["consecutive_repeats"] == 1
    assert r["revisited_arcs"] == 1  # arc 0 appears again later
    assert r["route_length"] == 6


def test_continuity_statuses(toy):
    nodes, arcs = toy
    G, _, _ = rn.build_road_graph(nodes, arcs)
    # 0->1 ok, 1->1 same arc, 1 then 3 (1->2 then 1->0: shares node 1), 3 then 2 (1->0 then 2->3: gap)
    c = check_route_continuity(np.array([0, 1, 1, 3, 2]), arcs, nodes, G)
    assert list(c.status) == ["connected", "same_arc", "shares_node", "gap"]
    gap = c.iloc[3]
    assert gap.gap_m == pytest.approx(2 * 71.5, abs=2)  # node 0 to node 2
    assert bool(gap.path_exists) and gap.path_arcs == 2


def test_continuity_no_path(toy):
    nodes, arcs = toy
    G, _, _ = rn.build_road_graph(nodes, arcs)
    c = check_route_continuity(np.array([4, 0]), arcs, nodes, G)  # ends at 4, next starts at 0; 4 has no out-arcs
    assert c.status.iloc[0] == "gap" and c.path_exists.iloc[0] == False  # noqa: E712


def test_gps_distances(toy):
    nodes, arcs = toy
    # points 0 m, ~11 m north, and beyond the end of the route
    track = pd.DataFrame({"lon": [8.0005, 8.0015, 8.0040], "lat": [50.0, 50.0001, 50.0], "t": [0.0, 1.0, 2.0]})
    d = gps_to_route_distances(track, nodes, arcs, np.array([0, 1, 2]))
    assert len(d) == 3
    assert d.distance_m[0] == pytest.approx(0, abs=0.5)
    assert d.distance_m[1] == pytest.approx(11.1, abs=0.5)
    assert d.distance_m[2] == pytest.approx(71.5, abs=2)  # distance to node 3, the route end
    assert list(d.nearest_arc_id[:2]) == [0, 1]


def test_make_route_map(toy, tmp_path):
    pytest.importorskip("folium")
    nodes, arcs = toy
    track = pd.DataFrame({"lon": [8.0005, 8.0015], "lat": [50.0, 50.0001], "t": [0.0, 1.0]})
    route = np.array([0, 1, 2])
    m = make_route_map("toy", track, nodes, arcs, route, continuity=check_route_continuity(route, arcs, nodes))
    out = tmp_path / "m.html"
    m.save(str(out))
    assert out.stat().st_size > 1000


def test_evaluator_guard():
    with pytest.raises(PermissionError):
        load_route_trip("x", {}, "evaluator-testing")


def test_real_trips_if_present():
    from pathsnap.data.loader import discover_trips
    from pathsnap.utils.config import get_raw_data_dir

    raw = get_raw_data_dir()
    if not raw.exists() or not list(raw.rglob("*.route")):
        pytest.skip("Real raw dataset not present")
    idx = discover_trips(raw)
    split = rn.load_split()
    for part in ("train", "student-test"):
        tid = split[part][0]
        d = load_route_trip(tid, idx, part)
        assert not d["errors"]
        assert validate_route_ids(d["route"], len(d["arcs"]))["invalid_arc_ids"] == 0
        c = check_route_continuity(d["route"], d["arcs"], d["nodes"])
        assert len(c) == len(d["route"]) - 1
        dist = gps_to_route_distances(d["track"], d["nodes"], d["arcs"], d["route"])
        assert len(dist) == len(d["track"]) and (dist.distance_m >= 0).all()
