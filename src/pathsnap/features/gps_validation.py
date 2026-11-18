
import pandas as pd


def validate_gps(df):
    """
    Validate GPS observations.

    Supports:
        Person 1's loader: lon, lat, t
        Original format: longitude, latitude, timestamp

    Returns a dictionary containing GPS validation statistics.
    """

    gps = df.copy()

    # Standardize column names
    gps = gps.rename(columns={
        "lon": "longitude",
        "lat": "latitude",
        "t": "timestamp"
    })

    required = ["longitude", "latitude", "timestamp"]

    missing_columns = [
        col for col in required if col not in gps.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    gps = gps[required].copy()

    # Convert columns to numeric values
    for col in required:
        gps[col] = pd.to_numeric(gps[col], errors="coerce")

    # 1. Missing coordinates
    missing_coordinates = gps[
        ["longitude", "latitude"]
    ].isna().any(axis=1)

    # 2. Invalid longitude (-180 to 180)
    invalid_longitude = (
        gps["longitude"].notna()
        & ~gps["longitude"].between(-180, 180)
    )

    # 3. Invalid latitude (-90 to 90)
    invalid_latitude = (
        gps["latitude"].notna()
        & ~gps["latitude"].between(-90, 90)
    )

    # 4. Potentially swapped longitude and latitude
    # Detects points where latitude is invalid,
    # but swapping the two coordinates could fix it.
    swapped_coordinates = (
        gps["longitude"].notna()
        & gps["latitude"].notna()
        & ~gps["latitude"].between(-90, 90)
        & gps["latitude"].between(-180, 180)
        & gps["longitude"].between(-90, 90)
    )

    # 5. Duplicate GPS coordinates
    duplicates = gps.duplicated(
        subset=["longitude", "latitude"],
        keep="first"
    )

    # 6. Missing timestamps
    missing_timestamps = gps["timestamp"].isna()

    # 7. Timestamp differences
    time_difference = gps["timestamp"].diff()

    # 8. Out-of-order timestamps
    out_of_order = time_difference < 0

    # 9. Timestamp gaps greater than 1 second
    large_gaps = time_difference > 1

    # Return validation statistics
    return {
        "total_points": len(gps),
        "missing_coordinates": int(missing_coordinates.sum()),
        "invalid_longitude": int(invalid_longitude.sum()),
        "invalid_latitude": int(invalid_latitude.sum()),
        "potentially_swapped": int(swapped_coordinates.sum()),
        "duplicate_coordinates": int(duplicates.sum()),
        "missing_timestamps": int(missing_timestamps.sum()),
        "out_of_order_timestamps": int(out_of_order.sum()),
        "gaps_over_1_second": int(large_gaps.sum()),
    }


if __name__ == "__main__":

    # Test using Person 1's column names
    sample = pd.DataFrame({
        "lon": [44.27, 44.28, 44.28, 200.0],
        "lat": [40.36, 40.37, 40.37, 40.38],
        "t": [1, 2, 5, 4]
    })

    results = validate_gps(sample)

    print("\nGPS Validation Results")
    print("----------------------")

    for key, value in results.items():
        print(f"{key}: {value}")
