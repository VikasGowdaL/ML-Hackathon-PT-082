
import pandas as pd


def validate_gps(df):
    """
    Validate GPS observations.

    Expected columns:
        longitude, latitude, timestamp

    Returns a dictionary containing validation counts.
    """

    required = ["longitude", "latitude", "timestamp"]

    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    gps = df[required].copy()

    for col in required:
        gps[col] = pd.to_numeric(
            gps[col], errors="coerce"
        )

    # 1. Missing coordinates
    missing_coordinates = gps[
        ["longitude", "latitude"]
    ].isna().any(axis=1)

    # 2. Invalid coordinate ranges
    invalid_longitude = (
        gps["longitude"].notna()
        & ~gps["longitude"].between(-180, 180)
    )

    invalid_latitude = (
        gps["latitude"].notna()
        & ~gps["latitude"].between(-90, 90)
    )

    # 3. Potentially swapped coordinates
    # Detect cases where the original latitude is invalid
    # but swapping produces valid coordinate ranges.
    swapped_coordinates = (
        gps["longitude"].notna()
        & gps["latitude"].notna()
        & ~gps["latitude"].between(-90, 90)
        & gps["latitude"].between(-180, 180)
        & gps["longitude"].between(-90, 90)
    )

    # 4. Duplicate GPS coordinates
    duplicates = gps.duplicated(
        subset=["longitude", "latitude"],
        keep="first"
    )

    # 5. Timestamp order
    time_difference = gps["timestamp"].diff()

    out_of_order = time_difference < 0

    # 6. Timestamp gaps greater than 1 second
    large_gaps = time_difference > 1

    # Additional: missing or invalid timestamps
    missing_timestamps = gps["timestamp"].isna()

    results = {
        "total_points": len(gps),
        "missing_coordinates": int(
            missing_coordinates.sum()
        ),
        "invalid_longitude": int(
            invalid_longitude.sum()
        ),
        "invalid_latitude": int(
            invalid_latitude.sum()
        ),
        "potentially_swapped": int(
            swapped_coordinates.sum()
        ),
        "duplicate_coordinates": int(
            duplicates.sum()
        ),
        "missing_timestamps": int(
            missing_timestamps.sum()
        ),
        "out_of_order_timestamps": int(
            out_of_order.sum()
        ),
        "gaps_over_1_second": int(
            large_gaps.sum()
        ),
    }

    return results


if __name__ == "__main__":
    # Synthetic test data; no dependency on Person 1.
    sample = pd.DataFrame({
        "longitude": [44.27, 44.28, 44.28, 200.0],
        "latitude": [40.36, 40.37, 40.37, 40.38],
        "timestamp": [1, 2, 5, 4],
    })

    print("GPS Validation Results:")
    for key, value in validate_gps(sample).items():
        print(f"{key}: {value}")
