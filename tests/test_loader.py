"""Unit tests for PathSnap data loader, partition guards, and UTM projections."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathsnap.data.loader import (
    TripData,
    compute_utm_epsg,
    discover_trips,
    get_trips,
    load_trip,
    project_coordinates,
)
from scripts.build_inventory import create_or_load_split


@pytest.fixture
def mock_trip_dir(tmp_path: Path) -> Path:
    """Create a synthetic raw trip directory mimicking the exact organizer file format."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    # 1. Synthetic Trip 1 (Train)
    t1_dir = raw_dir / "trip_001"
    t1_dir.mkdir()

    # .track: 3 whitespace-separated columns (lon, lat, t)
    track_content = (
        "13.404954 52.520008 1000.0\n"
        "13.405100 52.520150 1001.0\n"
        "13.405250 52.520300 1002.0\n"
        "13.405400 52.520450 1003.0\n"
    )
    (t1_dir / "trip_001.track").write_text(track_content, encoding="utf-8")

    # .nodes: 2 whitespace-separated columns (lon, lat)
    nodes_content = (
        "13.404900 52.520000\n"
        "13.405200 52.520200\n"
        "13.405500 52.520500\n"
    )
    (t1_dir / "trip_001.nodes").write_text(nodes_content, encoding="utf-8")

    # .arcs: 2 int columns (from_node_id, to_node_id)
    arcs_content = (
        "0 1\n"
        "1 2\n"
    )
    (t1_dir / "trip_001.arcs").write_text(arcs_content, encoding="utf-8")

    # .route: 1 int column (arc_id)
    route_content = (
        "0\n"
        "1\n"
    )
    (t1_dir / "trip_001.route").write_text(route_content, encoding="utf-8")

    # 2. Synthetic Trip 2 (Evaluator)
    t2_dir = raw_dir / "trip_002"
    t2_dir.mkdir()
    (t2_dir / "trip_002.track").write_text(track_content, encoding="utf-8")
    (t2_dir / "trip_002.nodes").write_text(nodes_content, encoding="utf-8")
    (t2_dir / "trip_002.arcs").write_text(arcs_content, encoding="utf-8")
    (t2_dir / "trip_002.route").write_text(route_content, encoding="utf-8")

    return raw_dir


def test_discover_trips(mock_trip_dir: Path):
    """Test trip discovery identifies all trips and files accurately."""
    discovered = discover_trips(mock_trip_dir)
    assert "trip_001" in discovered
    assert "trip_002" in discovered
    info = discovered["trip_001"]
    assert info["track"].exists()
    assert info["nodes"].exists()
    assert info["arcs"].exists()
    assert info["route"].exists()


def test_load_trip_shapes(mock_trip_dir: Path):
    """Test load_trip loads dataframes with valid shapes, types, and indices."""
    trip = load_trip("trip_001", raw_dir=mock_trip_dir, partition="train", project_utm=False)

    # Check track shape and columns
    assert trip.track.shape == (4, 3)
    assert list(trip.track.columns) == ["lon", "lat", "t"]
    assert np.issubdtype(trip.track["t"].dtype, np.floating)

    # Check nodes shape and columns
    assert trip.nodes.shape == (3, 3)
    assert list(trip.nodes.columns) == ["node_id", "lon", "lat"]
    assert list(trip.nodes["node_id"]) == [0, 1, 2]

    # Check arcs shape and columns
    assert trip.arcs.shape == (2, 3)
    assert list(trip.arcs.columns) == ["arc_id", "from_node_id", "to_node_id"]
    assert list(trip.arcs["arc_id"]) == [0, 1]

    # Check route
    assert trip.route is not None
    assert len(trip.route) == 2
    assert list(trip.route) == [0, 1]


def test_utm_projection(mock_trip_dir: Path):
    """Test dynamic UTM EPSG computation and coordinate projection."""
    pytest.importorskip("pyproj")
    trip = load_trip("trip_001", raw_dir=mock_trip_dir, partition="train", project_utm=True)

    # Check UTM projection
    assert trip.utm_crs is not None
    assert "EPSG:326" in trip.utm_crs
    assert trip.track_utm is not None
    assert list(trip.track_utm.columns) == ["x", "y", "t"]
    assert len(trip.track_utm) == 4
    assert trip.nodes_utm is not None
    assert list(trip.nodes_utm.columns) == ["node_id", "x", "y"]
    assert len(trip.nodes_utm) == 3


def test_evaluator_label_guard(mock_trip_dir: Path):
    """Test that evaluator partition labels are guarded and never leaked by default."""
    # By default, evaluator trip route must be withheld (None)
    eval_trip_guarded = load_trip(
        "trip_002",
        raw_dir=mock_trip_dir,
        partition="evaluator-testing",
        allow_evaluator_labels=False,
        use_cache=False,
    )
    assert eval_trip_guarded.route is None, "Evaluator route MUST be None when guard is active!"

    # When explicitly allowed with flag, route can be loaded
    eval_trip_unmasked = load_trip(
        "trip_002",
        raw_dir=mock_trip_dir,
        partition="evaluator-testing",
        allow_evaluator_labels=True,
        use_cache=False,
    )
    assert eval_trip_unmasked.route is not None
    assert len(eval_trip_unmasked.route) == 2


def test_split_disjointness_and_ratios(tmp_path: Path):
    """Test that split generation creates exactly 80 train, 10 student-test, 10 evaluator trips, strictly disjoint."""
    trip_ids = [f"trip_{i:03d}" for i in range(1, 101)]
    split_file = tmp_path / "split.json"

    split = create_or_load_split(
        trip_ids=trip_ids,
        detected_partitions={},
        split_file=split_file,
        seed=42,
    )

    train_set = set(split["train"])
    student_set = set(split["student-test"])
    eval_set = set(split["evaluator-testing"])

    # 1. Total counts must be exactly 80 / 10 / 10
    assert len(train_set) == 80, f"Expected 80 train trips, got {len(train_set)}"
    assert len(student_set) == 10, f"Expected 10 student-test trips, got {len(student_set)}"
    assert len(eval_set) == 10, f"Expected 10 evaluator-testing trips, got {len(eval_set)}"

    # 2. Partitions must be strictly pairwise disjoint
    assert len(train_set & student_set) == 0, "Train and student-test overlap!"
    assert len(train_set & eval_set) == 0, "Train and evaluator-testing overlap!"
    assert len(student_set & eval_set) == 0, "Student-test and evaluator-testing overlap!"

    # 3. Union must equal all 100 trips
    assert (train_set | student_set | eval_set) == set(trip_ids)


def test_get_trips_guard(mock_trip_dir: Path, tmp_path: Path):
    """Test get_trips guard blocks route loading for evaluator partition."""
    split_file = tmp_path / "split.json"
    split_data = {
        "train": ["trip_001"],
        "student-test": [],
        "evaluator-testing": ["trip_002"],
    }
    split_file.write_text(json.dumps(split_data), encoding="utf-8")

    # Train partition loads route
    train_trips = get_trips(
        "train",
        allow_evaluator_labels=False,
        raw_dir=mock_trip_dir,
        split_file=split_file,
        use_cache=False,
    )
    assert len(train_trips) == 1
    assert train_trips[0].route is not None

    # Evaluator partition suppresses route
    eval_trips = get_trips(
        "evaluator-testing",
        allow_evaluator_labels=False,
        raw_dir=mock_trip_dir,
        split_file=split_file,
        use_cache=False,
    )
    assert len(eval_trips) == 1
    assert eval_trips[0].route is None


def test_real_raw_dataset_if_present():
    """Verify loading real data if data/raw is populated."""
    from pathsnap.utils.config import get_raw_data_dir

    raw_dir = get_raw_data_dir()
    if not raw_dir.exists() or not list(raw_dir.rglob("*.track")):
        pytest.skip("Real raw dataset not present")

    trips = discover_trips(raw_dir)
    assert len(trips) >= 90
    assert "00000001" in trips
    trip = load_trip("00000001", raw_dir=raw_dir, partition="train", project_utm=False)
    assert trip.track.shape[1] == 3
    assert trip.nodes.shape[1] == 3
    assert trip.arcs.shape[1] == 3
    assert trip.route is not None
    assert len(trip.route) > 0
