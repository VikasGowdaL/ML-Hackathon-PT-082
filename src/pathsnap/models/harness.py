"""Shared training / evaluation harness for candidate-scoring models (Milestone 3).

Every model (LR, RF, KNN, XGBoost/LightGBM, MLP) imports this module -- never copy it.

Data rules enforced here
------------------------
* Fit and cross-validate on the ``train`` partition only (80 trips). ``student_test`` is used only for
  development comparison (:func:`evaluate_on_student`). Evaluator data is never read.
* Cross-validation is ``GroupKFold`` by ``trip_id`` (5 folds); trip disjointness is asserted for every fold.
* All learned preprocessing (imputation, scaling) lives inside an sklearn ``Pipeline`` built by
  :func:`make_pipeline`, so it is fitted on the training part of each fold only. Every model gets the same
  feature set and the same imputation; ``scale=True`` adds a ``StandardScaler`` (LR, KNN, MLP).
* One fixed seed (:data:`SEED`).

Typical use (see ``train_logistic_regression.py``)::

    from pathsnap.models import harness as H
    factory = lambda: H.make_pipeline(LogisticRegression(...), scale=True)
    H.run_experiment(factory, name="logistic_regression", family="linear", config={...})

Metric definitions: ``docs/metrics.md``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.data.loader import _read_arcs, discover_trips, load_trip
from pathsnap.features.assembly import FEATURE_COLUMNS
from pathsnap.features.reference_alignment import align_reference_route
from pathsnap.road_network import load_split
from pathsnap.utils.config import get_raw_data_dir, get_repo_root
from pathsnap.utils.experiment_log import log_experiment

SEED = 42
N_SPLITS = 5
TARGET = "label"
GROUP = "trip_id"
OBS_KEYS = ["trip_id", "obs_idx"]

# Columns of the Milestone 2 table that are NOT used as model inputs (and why).
EXCLUDED_FEATURES = {
    "t": "absolute timestamp: trip-specific, carries no matching information",
    "segment_id": "an ordinal id, not a measurement (gap information is in after_gap / is_segment_start)",
    "gps_heading_deg": "absolute direction; the model uses the relative bearing_diff_* features",
    "arc_bearing_deg": "absolute direction; the model uses the relative bearing_diff_* features",
    "distance_m": "identical to dist_to_segment_m (Person 1 and Person 3 compute the same distance)",
}
FEATURES: List[str] = [c for c in FEATURE_COLUMNS if c not in EXCLUDED_FEATURES]

ROW_METRICS = ["precision", "recall", "macro_f1", "accuracy", "log_loss"]
OBS_METRICS = ["top1_all_points", "top1_when_candidate", "candidate_miss_rate", "route_continuity"]
ALL_METRICS = ROW_METRICS + OBS_METRICS


def _paths() -> Dict[str, Path]:
    root = get_repo_root()
    return {
        "features": root / "outputs" / "features",
        "results": root / "outputs" / "results",
        "folds": root / "outputs" / "results" / "folds",
        "student": root / "outputs" / "results" / "student",
        "models": root / "artifacts" / "models",
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
_TABLE_COLUMNS = ["trip_id", "partition", "obs_idx", "arc_id", "reference_arc_id", TARGET, "segment_id"] + FEATURES


def load_train_table() -> pd.DataFrame:
    """Training feature table (80 train trips, labelled). Asserts it contains no other partition."""
    df = pd.read_parquet(_paths()["features"] / "feature_table_train.parquet", columns=list(dict.fromkeys(_TABLE_COLUMNS)))
    split = load_split()
    assert (df["partition"] == "train").all(), "non-train rows in training table"
    assert not set(df[GROUP]) & set(split["student_test"]), "student_test trip in training table"
    df[TARGET] = df[TARGET].astype(np.int8)
    return df


def student_test_labels(refresh: bool = False) -> pd.DataFrame:
    """Reference arc per student_test observation (Person 2's alignment), cached to parquet.

    Used for evaluation only. Returns ``trip_id, obs_idx, reference_arc_id`` for every GPS point.
    """
    cache = _paths()["features"] / "student_test_labels.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    idx = discover_trips(get_raw_data_dir())
    parts = []
    for tid in load_split()["student_test"]:
        trip = load_trip(tid, trips_index=idx, partition="student_test", project_utm=True)
        parts.append(align_reference_route(trip)[["trip_id", "obs_idx", "reference_arc_id"]])
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


def load_student_table() -> pd.DataFrame:
    """Student-test feature table with evaluation labels attached (feature table itself is unchanged)."""
    df = pd.read_parquet(_paths()["features"] / "feature_table_student_test.parquet", columns=list(dict.fromkeys(_TABLE_COLUMNS)))
    assert (df["partition"] == "student_test").all()
    ref = student_test_labels()
    df = df.drop(columns=["reference_arc_id", TARGET]).merge(ref, on=OBS_KEYS, how="left", validate="many_to_one")
    assert df["reference_arc_id"].notna().all(), "student_test observation without alignment"
    df[TARGET] = (df["arc_id"] == df["reference_arc_id"]).astype(np.int8)
    return df


def observation_counts() -> Dict[str, Dict[str, int]]:
    """Per trip: total GPS observations and observations whose reference arc is among the candidates.

    Needed because observations without any candidate have no row in the feature table.
    """
    s = pd.read_csv(_paths()["features"] / "feature_table_summary.csv", dtype={"trip_id": str})
    return {r.trip_id: {"observations": int(r.observations)} for r in s.itertuples()}


_ARC_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def arc_endpoints(trip_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """(from_node, to_node) arrays indexed by arc_id for one trip (cached; used for route continuity)."""
    if trip_id not in _ARC_CACHE:
        if not hasattr(arc_endpoints, "_idx"):
            arc_endpoints._idx = discover_trips(get_raw_data_dir())  # type: ignore[attr-defined]
        a = _read_arcs(arc_endpoints._idx[trip_id]["arcs"])  # type: ignore[attr-defined]
        _ARC_CACHE[trip_id] = (a["from_node_id"].to_numpy(), a["to_node_id"].to_numpy())
    return _ARC_CACHE[trip_id]


def X_y(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Model inputs (float32, booleans as 0/1, NaN kept for the imputer) and labels."""
    return df[FEATURES].astype(np.float32).to_numpy(), df[TARGET].to_numpy()


# ---------------------------------------------------------------------------
# Preprocessing (identical for all models)
# ---------------------------------------------------------------------------
def make_pipeline(estimator: Any, scale: bool) -> Pipeline:
    """Median imputation with missing-value indicators (+ StandardScaler if ``scale``) followed by the estimator.

    Everything is fitted inside ``Pipeline.fit``, i.e. on the training part of a fold only.
    """
    steps: List[Tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(df: pd.DataFrame, proba: np.ndarray, obs_counts: Optional[Dict[str, Dict[str, int]]] = None,
                    threshold: float = 0.5) -> Dict[str, Any]:
    """All Milestone 3 metrics for scored candidate rows (definitions in docs/metrics.md).

    Args:
        df: rows with ``trip_id, obs_idx, arc_id, segment_id, label``.
        proba: predicted P(label = 1) per row.
        obs_counts: from :func:`observation_counts`; gives the number of GPS points per trip, including
            points without candidates. Defaults to the observations present in ``df``.
    """
    y = df[TARGET].to_numpy()
    p = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m: Dict[str, Any] = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    # Observation level: highest-probability candidate per GPS point
    s = df[["trip_id", "obs_idx", "arc_id", "segment_id", TARGET]].copy()
    s["p"] = p
    s = s.sort_values(["trip_id", "obs_idx", "p"], ascending=[True, True, False], kind="mergesort")
    top = s.drop_duplicates(OBS_KEYS, keep="first")
    has_ref = s.groupby(OBS_KEYS, sort=False)[TARGET].max()
    trips = top["trip_id"].unique()
    n_all = sum(obs_counts[t]["observations"] for t in trips) if obs_counts else len(top)
    n_ref = int(has_ref.sum())
    n_correct = int(top[TARGET].sum())
    m["top1_all_points"] = n_correct / n_all if n_all else float("nan")
    m["top1_when_candidate"] = n_correct / n_ref if n_ref else float("nan")
    m["candidate_miss_rate"] = 1.0 - n_ref / n_all if n_all else float("nan")
    m["n_observations"] = int(n_all)
    m["n_observations_with_reference_candidate"] = n_ref

    # Route continuity of the top-1 sequence
    ok = total = 0
    for tid, g in top.groupby("trip_id", sort=False):
        f, t = arc_endpoints(tid)
        a = g["arc_id"].to_numpy()
        seg = g["segment_id"].to_numpy()
        same_seg = seg[1:] == seg[:-1]
        a0, a1 = a[:-1][same_seg], a[1:][same_seg]
        cont = (a0 == a1) | (t[a0] == f[a1])
        ok += int(cont.sum()); total += len(cont)
    m["route_continuity"] = ok / total if total else float("nan")
    return m


# ---------------------------------------------------------------------------
# Cross-validation and student-test evaluation
# ---------------------------------------------------------------------------
def _timed_proba(model: Any, X: np.ndarray) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    p = model.predict_proba(X)[:, 1]
    return p, time.perf_counter() - t0


def cross_validate(model_factory: Callable[[], Any], name: str, family: str, config: Dict[str, Any],
                   train: Optional[pd.DataFrame] = None, n_splits: int = N_SPLITS,
                   verbose: bool = True) -> pd.DataFrame:
    """Grouped (by trip) K-fold CV on the train partition. Returns one row per fold with train and validation metrics.

    ``model_factory()`` must return a fresh, unfitted estimator (normally :func:`make_pipeline`).
    """
    train = load_train_table() if train is None else train
    counts = observation_counts()
    X, y = X_y(train)
    groups = train[GROUP].to_numpy()
    rows = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        tr_trips, va_trips = set(groups[tr]), set(groups[va])
        assert not tr_trips & va_trips, f"fold {k}: trip in both training and validation"
        model = model_factory()
        t0 = time.perf_counter(); model.fit(X[tr], y[tr]); fit_s = time.perf_counter() - t0
        p_va, inf_s = _timed_proba(model, X[va])
        p_tr, _ = _timed_proba(model, X[tr])
        mv = compute_metrics(train.iloc[va], p_va, counts)
        mt = compute_metrics(train.iloc[tr], p_tr, counts)
        row = {"model": name, "fold": k, "train_trips": len(tr_trips), "val_trips": len(va_trips),
               "train_rows": len(tr), "val_rows": len(va), "val_positive_rate": float(y[va].mean()),
               "fit_time_s": fit_s, "infer_ms_per_1k_rows": 1000.0 * inf_s / (len(va) / 1000.0)}
        row.update({f"val_{m}": mv[m] for m in ALL_METRICS})
        row.update({f"train_{m}": mt[m] for m in ALL_METRICS})
        row["val_confusion_matrix"] = json.dumps(mv["confusion_matrix"])
        rows.append(row)
        if verbose:
            print(f"[{name}] fold {k}: fit {fit_s:6.1f}s | val macroF1 {mv['macro_f1']:.4f} "
                  f"top1 {mv['top1_when_candidate']:.4f} logloss {mv['log_loss']:.4f} | train macroF1 {mt['macro_f1']:.4f}")
    folds = pd.DataFrame(rows)
    p = _paths(); p["folds"].mkdir(parents=True, exist_ok=True)
    folds.to_csv(p["folds"] / f"{name}_folds.csv", index=False)
    return folds


def evaluate_on_student(model_factory: Callable[[], Any], name: str, train: Optional[pd.DataFrame] = None,
                        save: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """Fit on all 80 train trips, score student_test, save the model to ``artifacts/models/<name>.joblib``.

    Returns ``{"model", "train_metrics", "student_metrics", "fit_time_s", "infer_ms_per_1k_rows",
    "largest_trip": {...}, "per_trip": DataFrame}``.
    """
    train = load_train_table() if train is None else train
    student = load_student_table()
    counts = observation_counts()
    X, y = X_y(train)
    model = model_factory()
    t0 = time.perf_counter(); model.fit(X, y); fit_s = time.perf_counter() - t0
    p_tr, _ = _timed_proba(model, X)
    Xs, _ = X_y(student)
    p_st, inf_s = _timed_proba(model, Xs)
    out = {
        "model": model,
        "fit_time_s": fit_s,
        "infer_ms_per_1k_rows": 1000.0 * inf_s / (len(Xs) / 1000.0),
        "train_metrics": compute_metrics(train, p_tr, counts),
        "student_metrics": compute_metrics(student, p_st, counts),
    }
    # cost on the biggest trip (most candidate rows) in either partition
    sizes = pd.concat([train[GROUP], student[GROUP]]).value_counts()
    big = sizes.index[0]
    src = train if big in set(train[GROUP]) else student
    Xb, _ = X_y(src[src[GROUP] == big])
    _, tb = _timed_proba(model, Xb)
    out["largest_trip"] = {"trip_id": big, "rows": int(sizes.iloc[0]), "predict_s": tb}
    # per-trip student Top-1 (for error analysis)
    per = []
    for tid, g in student.assign(p=p_st).groupby(GROUP):
        mm = compute_metrics(g, g["p"].to_numpy(), counts)
        per.append({"trip_id": tid, **{k: mm[k] for k in ["top1_all_points", "top1_when_candidate", "candidate_miss_rate",
                                                           "route_continuity", "macro_f1", "log_loss"]}})
    out["per_trip"] = pd.DataFrame(per)
    if save:
        p = _paths()
        p["models"].mkdir(parents=True, exist_ok=True); p["student"].mkdir(parents=True, exist_ok=True)
        joblib.dump(model, p["models"] / f"{name}.joblib")
        out["per_trip"].to_csv(p["student"] / f"{name}_per_trip.csv", index=False)
    if verbose:
        sm, tm = out["student_metrics"], out["train_metrics"]
        print(f"[{name}] full fit {fit_s:.1f}s | train macroF1 {tm['macro_f1']:.4f} | student macroF1 {sm['macro_f1']:.4f} "
              f"top1 {sm['top1_when_candidate']:.4f} logloss {sm['log_loss']:.4f} | "
              f"{out['infer_ms_per_1k_rows']:.2f} ms / 1k rows | largest trip {big} ({sizes.iloc[0]:,} rows) {tb:.3f}s")
    return out


def summarise(folds: pd.DataFrame, student: Dict[str, Any]) -> Dict[str, Any]:
    """Flat metric dict for experiments.csv: CV mean/std, fold-train mean, full-train and student-test values."""
    m: Dict[str, Any] = {}
    for k in ALL_METRICS:
        m[f"cv_{k}_mean"] = float(folds[f"val_{k}"].mean())
        m[f"cv_{k}_std"] = float(folds[f"val_{k}"].std(ddof=1))
        m[f"cv_train_{k}_mean"] = float(folds[f"train_{k}"].mean())
        m[f"full_train_{k}"] = float(student["train_metrics"][k])
        m[f"student_{k}"] = float(student["student_metrics"][k])
    m["student_confusion_matrix"] = student["student_metrics"]["confusion_matrix"]
    m["cv_fit_time_s_mean"] = float(folds["fit_time_s"].mean())
    m["cv_infer_ms_per_1k_rows_mean"] = float(folds["infer_ms_per_1k_rows"].mean())
    m["full_fit_time_s"] = float(student["fit_time_s"])
    m["student_infer_ms_per_1k_rows"] = float(student["infer_ms_per_1k_rows"])
    m["largest_trip"] = student["largest_trip"]
    return m


def run_experiment(model_factory: Callable[[], Any], name: str, family: str, config: Dict[str, Any],
                   notes: str = "", train: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """CV + student-test evaluation + logging to outputs/results/experiments.csv. Returns the metric summary."""
    train = load_train_table() if train is None else train
    folds = cross_validate(model_factory, name, family, config, train=train)
    student = evaluate_on_student(model_factory, name, train=train)
    summary = summarise(folds, student)
    full_config = {**config, "features": "harness.FEATURES", "n_features": len(FEATURES), "cv": f"GroupKFold({N_SPLITS}) by trip_id"}
    log_experiment(model=name, family=family, config=full_config, metrics=summary, seed=SEED,
                   train_time_sec=student["fit_time_s"], inference_time_sec=student["infer_ms_per_1k_rows"] / 1000.0,
                   notes=notes or "inference_time_sec = seconds per 1,000 student_test rows")
    return summary
