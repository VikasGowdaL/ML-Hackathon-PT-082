"""Tests for the end-to-end prediction / scoring pipeline (src/pathsnap/pipeline.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathsnap import pipeline as P


def _toy_prediction(route_arcs, predicted):
    nodes = pd.DataFrame({"node_id": [0, 1, 2, 3], "lon": [8.0, 8.001, 8.002, 8.003], "lat": [50.0] * 4})
    arcs = pd.DataFrame({"arc_id": [0, 1, 2], "from_node_id": [0, 1, 2], "to_node_id": [1, 2, 3]})
    obs = pd.DataFrame({"obs_idx": range(len(predicted)), "predicted_arc_id": pd.array(predicted, dtype="Int64")})
    return P.Prediction(trip_id="toy", n_observations=len(predicted), observations=obs, sections=[route_arcs],
                        breaks=[], bridge_arcs=0, bridge_length_m=0.0, nodes=nodes, arcs=arcs)


def test_score_perfect_and_partial():
    perfect = P.score_prediction(_toy_prediction([0, 1, 2], [0, 1, 2]), np.array([0, 1, 2]))
    assert perfect["route_f1"] == pytest.approx(1.0) and perfect["points_on_route"] == 1.0
    half = P.score_prediction(_toy_prediction([0], [0, None]), np.array([0, 1]))
    assert half["route_precision"] == pytest.approx(1.0)
    assert half["route_recall"] == pytest.approx(0.5, abs=0.01)   # arcs have equal length
    assert half["points_on_route"] == 0.5 and half["points_without_candidates"] == 1


def _real_trip_available(tid="00000052"):
    return tid in P.trips_index() and P.model_path().exists()


def test_predict_never_reads_reference_route(monkeypatch):
    if not _real_trip_available():
        pytest.skip("raw data or final model not present")
    import pathsnap.data.loader as loader

    def forbidden(*a, **k):
        raise AssertionError("prediction tried to read a .route file")

    monkeypatch.setattr(loader, "_read_route", forbidden)
    pred = P.predict_trip("00000052")
    assert pred.n_observations == len(pred.observations) > 0
    assert pred.sections and all(len(s) > 0 for s in pred.sections)


def test_route_sections_are_connected():
    if not _real_trip_available():
        pytest.skip("raw data or final model not present")
    pred = P.predict_trip("00000052")
    f, t = pred.arcs["from_node_id"].to_numpy(), pred.arcs["to_node_id"].to_numpy()
    for sec in pred.sections:
        for a, b in zip(sec, sec[1:]):
            assert t[a] == f[b], "consecutive arcs inside a section must connect"
    lines = P.route_polylines(pred)
    assert len(lines) == len(pred.sections) and all(len(l) >= 2 for l in lines)
    score = P.score_prediction(pred, P.load_reference_route("00000052", "student_test"))
    assert 0.5 < score["route_f1"] <= 1.0
