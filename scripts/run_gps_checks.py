
from pathlib import Path
import sys

import pandas as pd

# Make the project's src package importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.data.loader import discover_trips, load_trip
from pathsnap.features.gps_validation import validate_gps
from pathsnap.features.gps_visualization import plot_gps_track


RAW_DIR = ROOT / "data" / "raw" / "Student folder"
OUTPUT_DIR = ROOT / "outputs" / "gps_validation"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    plots_created = 0

    for partition in ["train", "student_test"]:
        partition_dir = RAW_DIR / partition

        if not partition_dir.exists():
            print(f"Missing partition: {partition_dir}")
            continue

        # Use Person 1's trip discovery function
        trips_index = discover_trips(partition_dir)

        print(f"\n{partition}: {len(trips_index)} trips discovered")

        for trip_id in sorted(trips_index):
            try:
                # Use Person 1's loader, without modifying it.
                # Disable cache and projection for these raw GPS checks.
                trip = load_trip(
                    trip_id=trip_id,
                    raw_dir=partition_dir,
                    trips_index=trips_index,
                    partition=partition,
                    use_cache=False,
                    project_utm=False
                )

                stats = validate_gps(trip.track)

                results.append({
                    "partition": partition,
                    "trip_id": trip_id,
                    **stats
                })

                # Plot five training trips
                if partition == "train" and plots_created < 5:
                    plot_gps_track(
                        trip.track,
                        trip_id=trip_id,
                        save_path=OUTPUT_DIR / f"{trip_id}.png"
                    )
                    plots_created += 1

            except Exception as error:
                print(f"Error processing {partition}/{trip_id}: {error}")

    if not results:
        print("No trips were processed successfully.")
        return

    df = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "gps_validation_results.csv"
    df.to_csv(output_file, index=False)

    print("\nGPS Validation Summary")
    print("----------------------")
    print(f"Trips processed: {len(df)}")
    print(f"Total GPS points: {df['total_points'].sum()}")

    for column in df.columns:
        if column not in ["partition", "trip_id", "total_points"]:
            print(f"{column}: {df[column].sum()}")

    print(f"\nResults saved to: {output_file}")
    print(f"GPS plots generated: {plots_created}")


if __name__ == "__main__":
    main()
