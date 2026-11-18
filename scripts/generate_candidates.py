#!/usr/bin/env python3
"""Batch Candidate Generation & Evaluation Script for Milestone 2.

Runs spatial candidate generation across the 80 train and 10 student_test trips
for configured metric radii (30m, 50m, 100m, 200m) and default cap (30).

Outputs:
  - outputs/candidates/<radius>/<trip_id>.parquet
  - outputs/candidates/summary_<radius>.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil
import yaml
from tqdm import tqdm

from pathsnap.data.loader import get_trips, load_trip
from pathsnap.features.candidates import generate_candidates, save_candidates_parquet
from pathsnap.utils.config import get_path, get_repo_root


def get_current_memory_mb() -> float:
    """Return process resident memory in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def load_candidate_config() -> Dict[str, Any]:
    """Load configuration from configs/candidates.yaml."""
    cfg_file = get_repo_root() / "configs" / "candidates.yaml"
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "default_radius_m": 50.0,
        "default_max_candidates": 30,
        "test_radii_m": [30.0, 50.0, 100.0, 200.0],
    }


def process_partition_candidates(
    trips: List[Any],
    radius_m: float,
    max_candidates: int,
    output_dir: Path,
    summary_file: Path,
) -> pd.DataFrame:
    """Generate and save candidates for all trips in a list, writing per-trip parquet and summary CSV."""
    radius_dir = output_dir / f"{int(radius_m) if radius_m.is_integer() else radius_m}"
    radius_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print(f"\n[*] Processing {len(trips)} trips at radius={radius_m}m, cap={max_candidates}...")
    for trip in tqdm(trips, desc=f"Radius {radius_m}m"):
        t0 = time.perf_counter()
        mem_start = get_current_memory_mb()

        cand_df = generate_candidates(
            trip=trip,
            radius_m=radius_m,
            max_candidates=max_candidates,
            use_index_cache=True,
        )

        t1 = time.perf_counter()
        runtime_s = round(t1 - t0, 4)
        mem_used = round(get_current_memory_mb() - mem_start, 2)

        # Save per-trip parquet
        parquet_path = radius_dir / f"{trip.trip_id}.parquet"
        save_candidates_parquet(cand_df, parquet_path)

        n_obs = len(trip.track)
        n_candidates = len(cand_df)
        n_zero_candidates = cand_df.attrs.get("n_zero_candidates", 0)

        # Compute per-observation candidate statistics
        if n_candidates > 0:
            cands_per_obs = cand_df.groupby("obs_idx")["arc_id"].count()
            # Count observations with 0 candidates as 0
            all_counts = np.zeros(n_obs, dtype=np.int64)
            all_counts[cands_per_obs.index.to_numpy()] = cands_per_obs.to_numpy()
            mean_cands = round(float(all_counts.mean()), 2)
            max_cands = int(all_counts.max())
        else:
            mean_cands = 0.0
            max_cands = 0

        summary_rows.append({
            "trip_id": str(trip.trip_id),
            "partition": str(trip.partition),
            "n_obs": n_obs,
            "n_candidates": n_candidates,
            "mean_candidates_per_obs": mean_cands,
            "max_candidates_per_obs": max_cands,
            "n_obs_with_zero_candidates": n_zero_candidates,
            "zero_candidate_pct": round(100.0 * n_zero_candidates / max(1, n_obs), 2),
            "runtime_s": runtime_s,
            "mem_delta_mb": mem_used,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_file, index=False)
    print(f"[+] Saved summary to: {summary_file}")
    return summary_df


def run_candidate_benchmark(
    radii: List[float],
    max_candidates: int = 30,
    specific_trip_id: Optional[str] = None,
) -> None:
    """Run candidate extraction across train and student_test for all specified radii."""
    repo_root = get_repo_root()
    output_base = repo_root / "outputs" / "candidates"

    print("=" * 65)
    print(" PathSnap Milestone 2: Spatial Candidate Generation Benchmark")
    print("=" * 65)

    if specific_trip_id:
        print(f"[*] Single trip mode: {specific_trip_id}")
        t = load_trip(specific_trip_id, project_utm=True)
        trips = [t]
    else:
        print("[*] Loading train trips (80) and student_test trips (10)...")
        train_trips = get_trips("train", project_utm=True)
        test_trips = get_trips("student_test", project_utm=True)
        trips = train_trips + test_trips
        print(f"[+] Loaded {len(trips)} total trips (train: {len(train_trips)}, student_test: {len(test_trips)}).")

    for r in radii:
        summary_path = output_base / f"summary_{int(r) if r.is_integer() else r}.csv"
        df = process_partition_candidates(
            trips=trips,
            radius_m=r,
            max_candidates=max_candidates,
            output_dir=output_base,
            summary_file=summary_path,
        )

        total_runtime = df["runtime_s"].sum()
        total_cands = df["n_candidates"].sum()
        total_zero_obs = df["n_obs_with_zero_candidates"].sum()
        total_obs = df["n_obs"].sum()
        pct_zero = 100.0 * total_zero_obs / max(1, total_obs)

        print(f"\n--- Benchmark Summary for Radius={r}m (cap={max_candidates}) ---")
        print(f"  Total Candidates Generated: {total_cands:,}")
        print(f"  Avg Candidates / Obs     : {total_cands / max(1, total_obs):.2f}")
        print(f"  Observations with 0 Cands : {total_zero_obs:,} / {total_obs:,} ({pct_zero:.2f}%)")
        print(f"  Total Runtime             : {total_runtime:.2f}s (avg {total_runtime / len(df):.3f}s / trip)")


def main() -> int:
    parser = argparse.ArgumentParser(description="PathSnap Spatial Candidate Generation")
    parser.add_argument("--radius", type=float, default=None, help="Specific search radius in meters")
    parser.add_argument("--cap", type=int, default=None, help="Max candidate arcs per observation")
    parser.add_argument("--trip-id", type=str, default=None, help="Process a single trip ID")
    parser.add_argument("--all-radii", action="store_true", help="Run benchmark across all radii in config")
    args = parser.parse_args()

    cfg = load_candidate_config()
    cap = args.cap if args.cap is not None else cfg.get("default_max_candidates", 30)

    if args.all_radii:
        radii = cfg.get("test_radii_m", [30.0, 50.0, 100.0, 200.0])
    elif args.radius is not None:
        radii = [args.radius]
    else:
        radii = [cfg.get("default_radius_m", 50.0)]

    run_candidate_benchmark(radii=radii, max_candidates=cap, specific_trip_id=args.trip_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
