
"""Evaluate candidate coverage across all training trips."""

from pathlib import Path

import pandas as pd

from pathsnap.data.loader import get_trips
from pathsnap.features.candidates import generate_candidates
from pathsnap.features.reference_alignment import align_reference_route
from pathsnap.features.candidate_labeling import (
    label_candidates,
    calculate_miss_rate,
    find_worst_trips,
)

RADII = (30, 50, 100, 200)
MAX_CANDIDATES = 30
OUTPUT_DIR = Path("reports/candidate_evaluation")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trips = get_trips(
        "train",
        use_cache=False,
        project_utm=True,
    )

    all_summaries = []
    all_labeled = []
    failures = []

    print(f"Evaluating {len(trips)} training trips")

    for index, trip in enumerate(trips, start=1):
        print(
            f"[{index}/{len(trips)}] "
            f"Processing trip {trip.trip_id}"
        )

        try:
            alignment = align_reference_route(trip)

            candidate_frames = []

            for radius in RADII:
                candidates = generate_candidates(
                    trip,
                    radius_m=radius,
                    max_candidates=MAX_CANDIDATES,
                    use_index_cache=False,
                )

                candidates["radius_m"] = radius
                candidate_frames.append(candidates)

            combined = pd.concat(
                candidate_frames,
                ignore_index=True,
            )

            labeled = label_candidates(combined, alignment)

            # Ensure all radii are evaluated, even if a
            # radius produces zero candidates.
            for radius in RADII:
                radius_labeled = labeled[
                    labeled["radius_m"] == radius
                ]

                # Supply the radius explicitly for an empty set.
                if radius_labeled.empty:
                    total = len(alignment)
                    summary = pd.DataFrame([{
                        "trip_id": trip.trip_id,
                        "radius_m": radius,
                        "total_observations": total,
                        "matched_observations": 0,
                        "missed_observations": total,
                        "miss_rate": 1.0,
                    }])
                else:
                    summary = calculate_miss_rate(
                        radius_labeled,
                        alignment,
                    )

                all_summaries.append(summary)

            all_labeled.append(labeled)

        except Exception as exc:
            print(f"FAILED {trip.trip_id}: {exc}")
            failures.append({
                "trip_id": trip.trip_id,
                "error": str(exc),
            })

    if not all_summaries:
        raise RuntimeError("No trips evaluated successfully.")

    summary_df = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    worst_df = find_worst_trips(
        summary_df,
        top_n=10,
    )

    summary_df.to_csv(
        OUTPUT_DIR / "miss_rates_by_trip.csv",
        index=False,
    )

    worst_df.to_csv(
        OUTPUT_DIR / "worst_trips.csv",
        index=False,
    )

    if all_labeled:
        pd.concat(
            all_labeled,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR / "labeled_candidates.csv",
            index=False,
        )

    pd.DataFrame(
        failures,
        columns=["trip_id", "error"],
    ).to_csv(
        OUTPUT_DIR / "failed_trips.csv",
        index=False,
    )

    print("\nOverall miss rates by radius:")

    overall = (
        summary_df.groupby("radius_m")[
            ["missed_observations", "total_observations"]
        ]
        .sum()
    )

    overall["miss_rate"] = (
        overall["missed_observations"]
        / overall["total_observations"]
    )

    print(overall.to_string())

    print("\nWorst-performing trips:")
    print(worst_df.to_string(index=False))

    print(f"\nFailed trips: {len(failures)}")
    print(f"Reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
