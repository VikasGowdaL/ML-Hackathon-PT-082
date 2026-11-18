
import matplotlib.pyplot as plt
import pandas as pd


def plot_gps_track(df, trip_id="Sample Trip", save_path=None):
    """
    Plot GPS coordinates in their original observation order.

    Expected DataFrame columns:
        longitude, latitude, timestamp
    """

    required = ["longitude", "latitude"]

    if not all(col in df.columns for col in required):
        raise ValueError("Missing GPS coordinate columns")

    gps = df.copy()

    gps["longitude"] = pd.to_numeric(
        gps["longitude"], errors="coerce"
    )
    gps["latitude"] = pd.to_numeric(
        gps["latitude"], errors="coerce"
    )

    valid = gps[
        gps["longitude"].between(-180, 180)
        & gps["latitude"].between(-90, 90)
    ]

    if valid.empty:
        print(f"No valid coordinates for {trip_id}")
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
    plt.title(f"GPS Track — {trip_id}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.savefig("gps_test.png", dpi=150)
    plt.close()
    print("GPS plot saved successfully!")


if __name__ == "__main__":
    # Synthetic track to test plotting independently
    sample = pd.DataFrame({
        "longitude": [44.27, 44.271, 44.272, 44.273],
        "latitude": [40.36, 40.361, 40.362, 40.363],
        "timestamp": [1, 2, 3, 4],
    })

    plot_gps_track(sample, "Synthetic Test")
