#!/usr/bin/env python3
"""Build dataset inventory and create/verify the 80/10/10 train/student_test/evaluator_test split.

Outputs:
  - outputs/inventory/trip_inventory.csv
  - configs/split.json (only trip IDs, strictly disjoint partitions, NO labels)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pathsnap.data.loader import clear_cache, discover_trips, load_trip
from pathsnap.utils.config import get_path, get_raw_data_dir, get_repo_root


def detect_existing_partition(trip_info: Dict[str, Any]) -> Optional[str]:
    """Check if directory structure already indicates a partition.

    Standardized to underscores: 'train', 'student_test', 'evaluator_test'.
    """
    parent_dir = trip_info.get("dir")
    if not parent_dir:
        return None

    path_parts = [p.lower().replace("-", "_") for p in parent_dir.parts]
    for part in path_parts:
        if "student" in part:
            return "student_test"
        if "train" in part:
            return "train"
        if "eval" in part:
            return "evaluator_test"
    return None


def create_or_load_split(
    trip_ids: List[str],
    detected_partitions: Dict[str, Optional[str]],
    split_file: Path,
    seed: int = 42,
    force_resplit: bool = False,
) -> Dict[str, List[str]]:
    """Determine the 80/10/10 split by trip using uniform underscore partition names.

    1. Checks if split_file already exists.
    2. Checks if raw data directory already had partition folders.
    3. Otherwise generates deterministic 80/10/10 split by trip ID using a fixed seed.
    """
    if split_file.exists() and not force_resplit:
        print(f"[*] Found existing split file at: {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            existing = json.load(f)

        # Normalize any legacy hyphenated keys to underscores
        normalized: Dict[str, List[str]] = {
            "train": existing.get("train", []),
            "student_test": existing.get("student_test", existing.get("student-test", [])),
            "evaluator_test": existing.get("evaluator_test", existing.get("evaluator-testing", [])),
        }
        return normalized

    known = {tid: p for tid, p in detected_partitions.items() if p is not None}
    if len(known) == len(trip_ids) and len(trip_ids) > 0:
        print("[*] Partition detected directly from raw data folder structure.")
        split_dict: Dict[str, List[str]] = {
            "train": sorted([t for t in trip_ids if known[t] == "train"]),
            "student_test": sorted([t for t in trip_ids if known[t] == "student_test"]),
            "evaluator_test": sorted([t for t in trip_ids if known[t] == "evaluator_test"]),
        }
    else:
        print(f"[*] Generating deterministic 80/10/10 partition for {len(trip_ids)} trips (seed={seed})...")
        rng = random.Random(seed)
        shuffled = sorted(trip_ids)
        rng.shuffle(shuffled)

        n = len(shuffled)
        if n == 100:
            n_train = 80
            n_student = 10
            n_eval = 10
        else:
            n_train = int(round(n * 0.80))
            n_student = int(round(n * 0.10))
            n_eval = n - n_train - n_student

        split_dict = {
            "train": sorted(shuffled[:n_train]),
            "student_test": sorted(shuffled[n_train : n_train + n_student]),
            "evaluator_test": sorted(shuffled[n_train + n_student :]),
        }

    # Verify disjointness
    s_train = set(split_dict["train"])
    s_student = set(split_dict["student_test"])
    s_eval = set(split_dict["evaluator_test"])

    assert len(s_train.intersection(s_student)) == 0, "Train and student_test must be disjoint!"
    assert len(s_train.intersection(s_eval)) == 0, "Train and evaluator_test must be disjoint!"
    assert len(s_student.intersection(s_eval)) == 0, "Student_test and evaluator_test must be disjoint!"

    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)
    print(f"[+] Saved split (trip IDs only, strictly disjoint) to: {split_file}")
    print(f"    Train: {len(split_dict['train'])}, Student-test: {len(split_dict['student_test'])}, Evaluator (held by organisers): {len(split_dict['evaluator_test'])}")

    return split_dict


def build_inventory(raw_dir: Optional[Path] = None, seed: int = 42, force_resplit: bool = False, purge_cache: bool = False) -> Path:
    """Scan raw trips, compute summary statistics, and save inventory CSV."""
    if purge_cache:
        clear_cache()

    if raw_dir is None:
        raw_dir = get_raw_data_dir()

    print(f"[*] Discovering trips in: {raw_dir}")
    discovered = discover_trips(raw_dir)

    if not discovered:
        print(f"[!] Warning: No trips discovered in {raw_dir}.")
        return Path()

    trip_ids = sorted(discovered.keys())
    detected_parts = {tid: detect_existing_partition(discovered[tid]) for tid in trip_ids}

    split_file = get_path("split_file", default="configs/split.json")
    split = create_or_load_split(trip_ids, detected_parts, split_file, seed=seed, force_resplit=force_resplit)

    # Invert split for lookup
    trip_partition_map: Dict[str, str] = {}
    for part_name, t_list in split.items():
        for t in t_list:
            trip_partition_map[str(t)] = part_name

    inventory_rows = []

    print(f"[*] Analyzing {len(trip_ids)} trips...")
    for tid in tqdm(trip_ids, desc="Processing trips"):
        info = discovered[tid]
        part = trip_partition_map.get(tid, "unassigned")

        track_p = info.get("track")
        nodes_p = info.get("nodes")
        arcs_p = info.get("arcs")
        route_p = info.get("route")

        track_sz = track_p.stat().st_size if track_p and track_p.exists() else 0
        nodes_sz = nodes_p.stat().st_size if nodes_p and nodes_p.exists() else 0
        arcs_sz = arcs_p.stat().st_size if arcs_p and arcs_p.exists() else 0
        route_sz = route_p.stat().st_size if route_p and route_p.exists() else 0
        total_sz = track_sz + nodes_sz + arcs_sz + route_sz

        missing = []
        if not track_p or not track_p.exists():
            missing.append(".track")
        if not nodes_p or not nodes_p.exists():
            missing.append(".nodes")
        if not arcs_p or not arcs_p.exists():
            missing.append(".arcs")
        if not route_p or not route_p.exists():
            missing.append(".route")

        try:
            trip = load_trip(
                trip_id=tid,
                raw_dir=raw_dir,
                trips_index=discovered,
                partition=part,
                allow_evaluator_labels=True,  # For inventory metadata statistics
                use_cache=True,
                project_utm=False,
            )

            track_df = trip.track
            t_diff = track_df["t"].diff().dropna()

            duration = float(track_df["t"].max() - track_df["t"].min()) if len(track_df) > 1 else 0.0
            min_gap = float(t_diff.min()) if len(t_diff) > 0 else 0.0
            median_gap = float(t_diff.median()) if len(t_diff) > 0 else 0.0
            max_gap = float(t_diff.max()) if len(t_diff) > 0 else 0.0
            gaps_gt_1s = int((t_diff > 1.05).sum()) if len(t_diff) > 0 else 0

            # Check for fractional timestamps
            has_fractional_t = bool((track_df["t"] % 1.0 != 0.0).any())

            min_lon = float(track_df["lon"].min())
            min_lat = float(track_df["lat"].min())
            max_lon = float(track_df["lon"].max())
            max_lat = float(track_df["lat"].max())

            node_count = len(trip.nodes)
            arc_count = len(trip.arcs)
            route_len = len(trip.route) if trip.route is not None else -1

            inventory_rows.append({
                "trip_id": str(tid),
                "partition": part,
                "has_track": track_p is not None and track_p.exists(),
                "has_nodes": nodes_p is not None and nodes_p.exists(),
                "has_arcs": arcs_p is not None and arcs_p.exists(),
                "has_route": route_p is not None and route_p.exists(),
                "track_rows": len(track_df),
                "duration_sec": round(duration, 3),
                "min_time_gap": round(min_gap, 3),
                "median_time_gap": round(median_gap, 3),
                "max_time_gap": round(max_gap, 3),
                "gaps_gt_1s": gaps_gt_1s,
                "has_fractional_t": has_fractional_t,
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
                "node_count": node_count,
                "arc_count": arc_count,
                "route_length": route_len,
                "track_size_bytes": track_sz,
                "nodes_size_bytes": nodes_sz,
                "arcs_size_bytes": arcs_sz,
                "route_size_bytes": route_sz,
                "total_size_bytes": total_sz,
                "total_size_kb": round(total_sz / 1024.0, 2),
                "missing_files": ";".join(missing) if missing else "none",
            })
        except Exception as e:
            inventory_rows.append({
                "trip_id": str(tid),
                "partition": part,
                "has_track": track_p is not None and track_p.exists(),
                "has_nodes": nodes_p is not None and nodes_p.exists(),
                "has_arcs": arcs_p is not None and arcs_p.exists(),
                "has_route": route_p is not None and route_p.exists(),
                "track_rows": -1,
                "duration_sec": 0.0,
                "min_time_gap": 0.0,
                "median_time_gap": 0.0,
                "max_time_gap": 0.0,
                "gaps_gt_1s": -1,
                "has_fractional_t": False,
                "min_lon": 0.0,
                "min_lat": 0.0,
                "max_lon": 0.0,
                "max_lat": 0.0,
                "node_count": -1,
                "arc_count": -1,
                "route_length": -1,
                "track_size_bytes": track_sz,
                "nodes_size_bytes": nodes_sz,
                "arcs_size_bytes": arcs_sz,
                "route_size_bytes": route_sz,
                "total_size_bytes": total_sz,
                "total_size_kb": round(total_sz / 1024.0, 2),
                "missing_files": f"ERROR: {str(e)}",
            })

    inv_df = pd.DataFrame(inventory_rows)
    inv_dir = get_path("inventory_dir", default="outputs/inventory")
    inv_dir.mkdir(parents=True, exist_ok=True)
    out_file = inv_dir / "trip_inventory.csv"
    inv_df.to_csv(out_file, index=False)
    print(f"[+] Successfully wrote trip inventory to: {out_file}")
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser(description="PathSnap Trip Inventory Builder & Split Manager")
    parser.add_argument("--raw-dir", type=str, default=None, help="Path to raw data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting if unpartitioned")
    parser.add_argument("--force-resplit", action="store_true", help="Force recalculation of configs/split.json")
    parser.add_argument("--clear-cache", action="store_true", help="Purge cached files before inventory")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir).resolve() if args.raw_dir else None
    build_inventory(raw_dir=raw_path, seed=args.seed, force_resplit=args.force_resplit, purge_cache=args.clear_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
