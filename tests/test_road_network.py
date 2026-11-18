"""Tests for pathsnap.road_network (Member 3 - road-network analysis)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathsnap.road_network import (
    build_road_graph,
    calculate_connectivity_statistics,
    calculate_node_degrees,
    find_isolated_nodes,
    get_edge_by_arc_id,
    get_road_segment_geometry,
    load_road_network,
    scan_raw_file,
    validate_arcs,
)


@pytest.fixture
def toy_network():
    """6 nodes, 8 arcs covering every edge case.

    node 0..3 form a directed cycle 0->1->2->3->0 (one SCC of size 4)
    node 4 is reachable from 3 only (sink_only); node 5 has no arcs (isolated)
    arc 4 duplicates arc 0 (parallel 0->1); arc 5 is a self-loop on 2;
    arc 6 references a non-existent node 99; arc 7 is 3->4.
    """
    nodes = pd.DataFrame(
        {
            "node_id": np.arange(6, dtype=np.int64),
            "lon": [8.0, 8.001, 8.001, 8.0, 8.002, 8.01],
            "lat": [50.0, 50.0, 50.001, 50.001, 50.002, 50.01],
        }
    )
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1), (2, 2), (1, 99), (3, 4)]
    arcs = pd.DataFrame(
        {
            "arc_id": np.arange(len(pairs), dtype=np.int64),
            "from_node_id": [p[0] for p in pairs],
            "to_node_id": [p[1] for p in pairs],
        }
    )
    return nodes, arcs


def test_validate_arcs_flags(toy_network):
    nodes, arcs = toy_network
    v = validate_arcs(arcs, len(nodes), nodes)
    assert len(v) == len(arcs)
    assert list(v["arc_id"]) == list(range(8))
    assert v.loc[6, "is_valid"] == False and v.loc[6, "to_valid"] == False  # noqa: E712
    assert "invalid_to_node" in v.loc[6, "issues"]
    assert v.loc[5, "is_self_loop"]
    assert v.loc[4, "is_duplicate"] and not v.loc[0, "is_duplicate"]
    assert v.loc[0, "is_parallel"] and v.loc[0, "parallel_count"] == 2
    assert v["is_valid"].sum() == 7


def test_build_road_graph_preserves_ids(toy_network):
    nodes, arcs = toy_network
    G, val, summ = build_road_graph(nodes, arcs)
    assert G.is_directed() and G.is_multigraph()
    assert G.number_of_nodes() == 6
    assert G.number_of_edges() == 7  # arc 6 cannot be placed (invalid node 99)
    assert summ["invalid_arcs"] == 1 and summ["self_loops"] == 1 and summ["duplicate_arcs"] == 1
    # every valid arc present with key == arc_id and original endpoints
    for _, r in val[val["is_valid"]].iterrows():
        u, v, data = get_edge_by_arc_id(G, r["arc_id"])
        assert (u, v) == (r["from_node_id"], r["to_node_id"])
        assert data["arc_id"] == r["arc_id"]
    assert G.number_of_edges(0, 1) == 2  # parallel edges kept
    assert G.nodes[4]["longitude"] == pytest.approx(8.002)
    assert 6 not in G.graph["arc_endpoints"]
    # direction is preserved: 1->0 does not exist
    assert not G.has_edge(1, 0)


def test_isolated_and_degrees(toy_network):
    nodes, arcs = toy_network
    G, _, _ = build_road_graph(nodes, arcs)
    deg = calculate_node_degrees(G)
    assert deg.loc[deg.node_id == 0, "out_degree"].item() == 2
    assert deg.loc[deg.node_id == 2, "in_degree"].item() == 2  # 1->2 and self-loop
    iso = find_isolated_nodes(G, deg).set_index("node_id")["category"].to_dict()
    assert iso == {4: "sink_only", 5: "isolated"}


def test_connectivity(toy_network):
    nodes, arcs = toy_network
    G, _, _ = build_road_graph(nodes, arcs)
    stats, sizes = calculate_connectivity_statistics(G, return_sizes=True)
    assert stats["n_weak_components"] == 2  # {0..4} and {5}
    assert stats["largest_wcc_size"] == 5
    assert stats["n_strong_components"] == 3  # {0,1,2,3}, {4}, {5}
    assert stats["largest_scc_size"] == 4
    assert sizes["largest_scc"] == {0, 1, 2, 3}


def test_segment_geometry(toy_network):
    nodes, arcs = toy_network
    geo = get_road_segment_geometry(nodes, arcs, arc_ids=[1, 0, 6])
    assert list(geo["arc_id"]) == [1, 0, 6]
    # 0.001 deg longitude at 50N is about 71.5 m
    assert geo.loc[1, "length_m"] == pytest.approx(71.5, abs=1.0)
    assert np.isnan(geo.loc[2, "to_lon"])  # invalid endpoint -> NaN, not a crash


def test_scan_and_load_malformed(tmp_path):
    d = tmp_path / "t1"
    d.mkdir()
    (d / "t1.track").write_text("8 50 0\n", encoding="utf-8")
    (d / "t1.nodes").write_text("8.0 50.0\n8.1 abc\n8.2 50.2\n", encoding="utf-8")
    (d / "t1.arcs").write_text("0 1\n1 2 3\n", encoding="utf-8")
    rep = scan_raw_file(d / "t1.nodes", 2)
    assert rep["non_numeric_rows"] == 1 and rep["malformed_examples"] == [2]
    rep = scan_raw_file(d / "t1.arcs", 2, integer=True)
    assert rep["wrong_field_count"] == 1
    idx = {"t1": {"nodes": d / "t1.nodes", "arcs": d / "t1.arcs", "track": d / "t1.track"}}
    nodes, arcs, frep = load_road_network("t1", idx)
    assert nodes is not None and len(nodes) == 3  # row index preserved even for bad row
    assert np.isnan(nodes.loc[1, "lat"])
    assert arcs is not None and len(arcs) == 2
    # missing file handled without raising
    n2, a2, rep2 = load_road_network("nope", idx)
    assert n2 is None and rep2["errors"]


def test_real_trips_if_present():
    """Graph builder on one real train trip and one real student-test trip."""
    from pathsnap.data.loader import discover_trips
    from pathsnap.road_network import load_split
    from pathsnap.utils.config import get_raw_data_dir

    raw = get_raw_data_dir()
    if not raw.exists() or not list(raw.rglob("*.nodes")):
        pytest.skip("Real raw dataset not present")
    idx = discover_trips(raw)
    split = load_split()
    student_trip_id = split.get("student_test", split.get("student-test"))[0]
    for tid in (split["train"][0], student_trip_id):
        nodes, arcs, rep = load_road_network(tid, idx)
        assert not rep["errors"]
        G, val, summ = build_road_graph(nodes, arcs)
        assert G.number_of_nodes() == len(nodes)
        assert G.number_of_edges() == int(val["in_graph"].sum())
        assert summ["invalid_arcs"] == len(arcs) - G.number_of_edges()
        sample = arcs.sample(n=min(200, len(arcs)), random_state=42)
        for r in sample.itertuples():
            if val.loc[r.arc_id, "in_graph"]:
                u, v, data = get_edge_by_arc_id(G, r.arc_id)
                assert (u, v, data["arc_id"]) == (r.from_node_id, r.to_node_id, r.arc_id)
