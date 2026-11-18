#!/usr/bin/env python3
"""Evaluate the full prediction pipeline (final model + Viterbi + route reconstruction) on a partition.

Usage:
    python scripts/evaluate_pipeline.py                      # student_test (default)
    python scripts/evaluate_pipeline.py --partition train --limit 10

For every trip: predict_trip() (no route access), then score_prediction() against the reference route.
For train / student_test it also reports observation-level Top-1 of the model alone ("ML only", highest
probability per GPS point) and of Viterbi ("ML + HMM") against Person 2's aligned reference arc per point.
Evaluator trips are refused here: they are scored only live by the API.

Writes outputs/results/pipeline_<partition>.csv (per trip) and pipeline_<partition>_summary.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from pathsnap import pipeline as P
from pathsnap.data.loader import load_trip
from pathsnap.features.reference_alignment import align_reference_route
from pathsnap.road_network import load_split
from pathsnap.utils.config import get_repo_root


def continuity(arcs: pd.Series, f: np.ndarray, t: np.ndarray) -> float:
    a = arcs.dropna().astype(np.int64).to_numpy()
    if len(a) < 2:
        return float("nan")
    return float(((a[1:] == a[:-1]) | (t[a[:-1]] == f[a[1:]])).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", default="student_test", choices=["student_test", "train"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    trips = load_split()[args.partition][: args.limit]
    model = P.load_model()
    rows = []
    for tid in trips:
        pred = P.predict_trip(tid, model=model)
        score = P.score_prediction(pred, P.load_reference_route(tid, args.partition))
        trip = load_trip(tid, trips_index=P.trips_index(), partition=args.partition, use_cache=False, project_utm=True)
        ref = align_reference_route(trip)[["obs_idx", "reference_arc_id"]]
        o = pred.observations.merge(ref, on="obs_idx", how="left")
        f, t = pred.arcs["from_node_id"].to_numpy(), pred.arcs["to_node_id"].to_numpy()
        row = {
            "trip_id": tid, "gps_points": pred.n_observations,
            "ml_top1": float((o.top1_arc_id == o.reference_arc_id).mean()),
            "hmm_top1": float((o.predicted_arc_id == o.reference_arc_id).fillna(False).mean()),
            "ml_continuity": continuity(o.top1_arc_id, f, t),
            "hmm_continuity": continuity(o.predicted_arc_id, f, t),
            **score,
            "predict_s": pred.timing_s["total_s"],
        }
        rows.append(row)
        print(f"{tid}: ML {row['ml_top1']:.4f} -> HMM {row['hmm_top1']:.4f} | route F1 {row['route_f1']:.4f} "
              f"(P {row['route_precision']:.3f} R {row['route_recall']:.3f}) | {row['predict_s']:.2f}s")

    df = pd.DataFrame(rows)
    w = df.gps_points / df.gps_points.sum()
    summary = {
        "partition": args.partition, "trips": len(df), "gps_points": int(df.gps_points.sum()),
        "model": "artifacts/models/final_model.joblib",
        "ml_top1": float(np.average(df.ml_top1, weights=w)),
        "hmm_top1": float(np.average(df.hmm_top1, weights=w)),
        "ml_continuity": float(np.average(df.ml_continuity, weights=w)),
        "hmm_continuity": float(np.average(df.hmm_continuity, weights=w)),
        "points_on_route": float(np.average(df.points_on_route, weights=w)),
        "route_precision_mean": float(df.route_precision.mean()),
        "route_recall_mean": float(df.route_recall.mean()),
        "route_f1_mean": float(df.route_f1.mean()),
        "unbridged_breaks_total": int(df.unbridged_breaks.sum()),
        "predict_s_mean": float(df.predict_s.mean()),
        "predict_s_max": float(df.predict_s.max()),
        "note": "point-weighted Top-1 / continuity; route metrics are per-trip means",
    }
    out = get_repo_root() / "outputs" / "results"
    df.to_csv(out / f"pipeline_{args.partition}.csv", index=False)
    (out / f"pipeline_{args.partition}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
