
import pandas as pd
import pytest
from pathsnap.features.candidate_labeling import find_worst_trips

from pathsnap.features.candidate_labeling import (
    label_candidates,
    calculate_miss_rate,
)


def test_binary_candidate_labels():
    alignment = pd.DataFrame({
        "trip_id": ["trip1", "trip1", "trip1"],
        "obs_idx": [0, 1, 2],
        "reference_arc_id": [100, 200, 300],
    })

    candidates = pd.DataFrame({
        "trip_id": ["trip1"] * 4,
        "obs_idx": [0, 0, 1, 2],
        "arc_id": [100, 101, 201, 300],
        "radius_m": [30, 30, 30, 30],
    })

    labeled = label_candidates(candidates, alignment)

    assert labeled["label"].tolist() == [1, 0, 0, 1]


def test_candidate_miss_rate():
    alignment = pd.DataFrame({
        "trip_id": ["trip1"] * 3,
        "obs_idx": [0, 1, 2],
        "reference_arc_id": [100, 200, 300],
    })

    candidates = pd.DataFrame({
        "trip_id": ["trip1"] * 4,
        "obs_idx": [0, 1, 0, 2],
        "arc_id": [100, 201, 100, 300],
        "radius_m": [30, 30, 50, 50],
    })

    labeled = label_candidates(candidates, alignment)
    summary = calculate_miss_rate(labeled, alignment)

    r30 = summary.loc[summary["radius_m"] == 30].iloc[0]
    r50 = summary.loc[summary["radius_m"] == 50].iloc[0]

    assert r30["missed_observations"] == 2
    assert r30["miss_rate"] == pytest.approx(2 / 3)

    assert r50["missed_observations"] == 1
    assert r50["miss_rate"] == pytest.approx(1 / 3)


def test_missing_alignment_rejected():
    alignment = pd.DataFrame({
        "trip_id": ["trip1"],
        "obs_idx": [0],
        "reference_arc_id": [100],
    })

    candidates = pd.DataFrame({
        "trip_id": ["trip1"],
        "obs_idx": [5],
        "arc_id": [100],
    })

    with pytest.raises(ValueError):
        label_candidates(candidates, alignment)


def test_find_worst_trips():
    summary = pd.DataFrame({
        "trip_id": ["A", "B", "C", "A", "B", "C"],
        "radius_m": [30, 30, 30, 50, 50, 50],
        "miss_rate": [0.1, 0.7, 0.4, 0.0, 0.3, 0.2],
        "missed_observations": [1, 7, 4, 0, 3, 2],
    })

    worst = find_worst_trips(summary, top_n=2)

    r30 = worst[worst["radius_m"] == 30]
    r50 = worst[worst["radius_m"] == 50]

    assert r30["trip_id"].tolist() == ["B", "C"]
    assert r50["trip_id"].tolist() == ["B", "C"]
