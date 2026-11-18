
import matplotlib.pyplot as plt
import pandas as pd


def plot_gps_track(df, trip_id="Sample Trip", save_path=None):
    """
    Plot GPS coordinates in observation order.

    Supports:
        Person 1's loader: lon, lat, t
        Original format: longitude, latitude, timestamp
    """

    gps = df.copy()

    gps = gps.rename(columns={
        "lon": "longitude",
        "lat": "latitude",
        "t": "timestamp"
    })

    required = ["longitude", "latitude"]

    missing = [col for col in required if col not in gps.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in required:
        gps[col] = pd.to_numeric(gps[col], errors="coerce")

    valid = gps[
        gps["longitude"].between(-180, 180)
        & gps["latitude"].between(-90, 90)
    ]

    if valid.empty:
        print(f"No valid GPS coordinates for {trip_id}")
        return

    plt.figure(figsize=(10, 7))

    plt.plot(
        valid["longitude"],
        valid["latitude"],
        marker=".",
        markersize=3,
        linewidth=1,
        label="GPS Track"
    )

    plt.scatter(
        valid["longitude"].iloc[0],
        valid["latitude"].iloc[0],
        color="green",
        s=70,
        label="Start"
    )

    plt.scatter(
        valid["longitude"].iloc[-1],
        valid["latitude"].iloc[-1],
        color="red",
        s=70,
        label="End"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"GPS Track - {trip_id}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"GPS plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":

    sample = pd.DataFrame({
        "lon": [44.27, 44.271, 44.272, 44.273],
        "lat": [40.36, 40.361, 40.362, 40.363],
        "t": [1, 2, 3, 4]
    })

    plot_gps_track(
        sample,
        trip_id="Synthetic Test",
        save_path="gps_test.png"
    )
