#!/usr/bin/env python3
"""Build the Milestone 2 candidate feature tables (parquet) and the train-only scaler.

Usage:
    python scripts/build_feature_table.py [--config configs/features.yaml]

Writes to ``output_dir`` (default ``outputs/features``):
    feature_table_train.parquet, feature_table_student_test.parquet   (git-ignored, regenerate with this script)
    feature_table_summary.csv                                          per-trip row / label counts
    scaler_train.json                                                  mean/std fitted on training rows only
"""

from __future__ import annotations

import argparse
import time

import yaml

from pathsnap.data.loader import discover_trips
from pathsnap.features.assembly import build_feature_tables, fit_scaler_from_parquet, save_scaler
from pathsnap.road_network import load_split
from pathsnap.utils.config import get_raw_data_dir, get_repo_root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/features.yaml")
    args = ap.parse_args()

    root = get_repo_root()
    cfg = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    out_dir = root / cfg["output_dir"]

    t0 = time.time()
    summary = build_feature_tables(
        discover_trips(get_raw_data_dir()),
        load_split(),
        out_dir,
        partitions=cfg["partitions"],
        radius_m=cfg["radius_m"],
        max_candidates=cfg["max_candidates"],
        gap_policy=cfg["gap_policy"],
        progress=True,
    )
    summary.to_csv(out_dir / "feature_table_summary.csv", index=False)
    save_scaler(fit_scaler_from_parquet(out_dir / "feature_table_train.parquet"), out_dir / "scaler_train.json")
    print(f"Done in {time.time() - t0:.0f} s -> {out_dir}")


if __name__ == "__main__":
    main()
