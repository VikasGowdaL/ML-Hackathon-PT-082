"""Tests for sequence features and feature-table assembly (Milestone 2, Person 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathsnap.features.assembly import (
    COLUMN_ORDER,
    SCALE_COLUMNS,
    apply_scaler,
    fit_scaler,
    fit_scaler_from_parquet,
)
from pathsnap.features.sequence import compute_candidate_rank_features, compute_observation_features

FEATURE_DIR_PARTS = ("outputs", "features")


# ---------------------------------------------------------------------------
# Sequence features (synthetic)
# ---------------------------------------------------------------------------
@pytest.fixture
def track():
    # east at 10 m/s for 3 s, then 20 m/s, then a 100 s gap, then stationary
    return pd.DataFrame({"x": [0, 10, 20, 30, 50, 60, 60.5], "y": [0.0] * 7, "t": [0, 1, 2, 3, 4, 104, 105]})


def test_observation_features(track):
    f = compute_observation_features(track, "trip", policy={"flag_gap_s": 10, "split_gap_s": 60})
    assert list(f.obs_idx) == list(range(7))
    assert np.isnan(f.dt_prev_s[0]) and f.dt_next_s[0] == 1
    assert f.speed_prev_mps[2] == pytest.approx(10) and f.speed_next_mps[3] == pytest.approx(20)
    assert f.accel_mps2[3] == pytest.approx(10)          # 10 -> 20 m/s over 1 s
    assert f.heading_deg[1] == pytest.approx(90)         # east
    assert f.heading_change_deg[2] == pytest.approx(0)
    assert f.after_gap.tolist() == [False] * 5 + [True, False]
    assert f.before_gap.tolist() == [False] * 4 + [True, False, False]
    assert f.segment_id.tolist() == [0] * 5 + [1, 1]     # split at the 100 s gap
    assert f.is_segment_start.tolist() == [True, False, False, False, False, True, False]
    assert np.isnan(f.dt_prev_s[5]) and np.isnan(f.dt_next_s[4])  # no neighbour across a split
    assert not f.heading_valid[5] and not f.heading_valid[6]     # only 0.5 m moved in segment 1 -> no reliable heading
    assert not f.implausible_speed.any()


def test_implausible_speed():
    t = pd.DataFrame({"x": [0, 0, 500], "y": [0.0] * 3, "t": [0, 1, 2]})
    f = compute_observation_features(t, "trip")
    assert f.implausible_speed.tolist() == [False, False, True]


def test_rank_features():
    c = pd.DataFrame({"trip_id": "a", "obs_idx": [0, 0, 0, 1], "arc_id": [5, 6, 7, 5], "distance_m": [3.0, 1.0, 3.0, 9.0]})
    r = compute_candidate_rank_features(c)
    assert r.dist_rank.tolist() == [2, 1, 2, 1]
    assert r.n_candidates.tolist() == [3, 3, 3, 1]
    assert r.dist_minus_best_m.tolist() == [2.0, 0.0, 2.0, 0.0]
    assert list(r.arc_id) == [5, 6, 7, 5]  # order preserved


# ---------------------------------------------------------------------------
# Scaling: training rows only
# ---------------------------------------------------------------------------
def test_scaler_refuses_non_training_rows():
    df = pd.DataFrame({"partition": ["train", "student_test"], "distance_m": [1.0, 2.0]})
    with pytest.raises(ValueError):
        fit_scaler(df, ["distance_m"])
    ok = fit_scaler(df.iloc[:1], ["distance_m"])
    assert ok["distance_m"]["mean"] == 1.0 and ok["distance_m"]["std"] == 1.0  # zero std -> 1
    scaled = apply_scaler(pd.DataFrame({"distance_m": [3.0]}), {"distance_m": {"mean": 1.0, "std": 2.0}})
    assert scaled.distance_m[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Built feature tables (skipped until scripts/build_feature_table.py has been run)
# ---------------------------------------------------------------------------
def _tables():
    from pathsnap.utils.config import get_repo_root

    d = get_repo_root().joinpath(*FEATURE_DIR_PARTS)
    train, test = d / "feature_table_train.parquet", d / "feature_table_student_test.parquet"
    if not train.exists() or not test.exists():
        pytest.skip("Feature tables not built (run scripts/build_feature_table.py)")
    return d, train, test


def test_no_duplicate_trip_obs_arc_rows():
    _, train, test = _tables()
    for p in (train, test):
        k = pd.read_parquet(p, columns=["trip_id", "obs_idx", "arc_id"])
        assert not k.duplicated().any(), f"duplicate (trip, obs, arc) rows in {p.name}"


def test_no_student_test_trip_in_training_table():
    from pathsnap.road_network import load_split

    _, train, test = _tables()
    split = load_split()
    tr = pd.read_parquet(train, columns=["trip_id", "partition"])
    assert set(tr.partition) == {"train"}
    assert not set(tr.trip_id) & set(split["student_test"])
    assert set(tr.trip_id) <= set(split["train"])
    te = pd.read_parquet(test, columns=["trip_id", "partition", "label"])
    assert set(te.trip_id) <= set(split["student_test"]) and set(te.partition) == {"student_test"}


def test_schema_and_labels():
    _, train, _ = _tables()
    df = pd.read_parquet(train, columns=COLUMN_ORDER)
    assert list(df.columns) == COLUMN_ORDER
    assert set(df.label.dropna().unique()) <= {0, 1}
    per_obs = df.groupby(["trip_id", "obs_idx"]).label.sum()
    assert (per_obs <= df.groupby(["trip_id", "obs_idx"]).arc_id.nunique()).all()


def test_saved_scaler_fitted_on_training_only():
    import json

    d, train, _ = _tables()
    sc = d / "scaler_train.json"
    if not sc.exists():
        pytest.skip("scaler_train.json not built")
    saved = json.loads(sc.read_text(encoding="utf-8"))
    assert saved["fitted_on"] == "train" and set(saved["columns"]) == set(SCALE_COLUMNS)
    refit = fit_scaler_from_parquet(train, ["distance_m"])
    assert refit["distance_m"]["mean"] == pytest.approx(saved["columns"]["distance_m"]["mean"], rel=1e-6)
