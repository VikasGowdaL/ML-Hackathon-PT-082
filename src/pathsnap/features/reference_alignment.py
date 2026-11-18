
"""Align GPS observations with an ordered reference route."""

import numpy as np
import pandas as pd


def align_reference_route(trip):
    """
    Assign each GPS observation to an ordered reference route arc.

    Uses dynamic programming to minimize squared GPS-to-segment
    distances while ensuring route positions never decrease.

    Returns:
        DataFrame with trip_id, obs_idx, route_position,
        reference_arc_id.
    """

    if trip.partition != "train":
        raise ValueError("Only training trips can be aligned.")

    if trip.route is None or len(trip.route) == 0:
        raise ValueError("Reference route is missing.")

    if trip.track_utm is None or trip.nodes_utm is None:
        raise ValueError("UTM projection is required.")

    route = np.asarray(trip.route, dtype=np.int64)

    arcs = trip.arcs.set_index("arc_id")
    nodes = trip.nodes_utm.set_index("node_id")

    from_ids = arcs.loc[route, "from_node_id"].to_numpy()
    to_ids = arcs.loc[route, "to_node_id"].to_numpy()

    starts = nodes.loc[
        from_ids, ["x", "y"]
    ].to_numpy(dtype=float)

    ends = nodes.loc[
        to_ids, ["x", "y"]
    ].to_numpy(dtype=float)

    points = trip.track_utm[
        ["x", "y"]
    ].to_numpy(dtype=float)

    n = len(points)
    m = len(route)

    if n == 0:
        raise ValueError("No GPS observations found.")

    vectors = ends - starts
    lengths_sq = np.sum(vectors ** 2, axis=1)
    safe_lengths_sq = np.maximum(lengths_sq, 1e-12)

    dp = np.zeros(m, dtype=np.float64)

    predecessors = np.zeros((n, m), dtype=np.int32)

    for i, point in enumerate(points):

        relative = point - starts

        fractions = (
            np.sum(relative * vectors, axis=1)
            / safe_lengths_sq
        )

        fractions = np.clip(fractions, 0.0, 1.0)

        projections = (
            starts + fractions[:, None] * vectors
        )

        differences = projections - point

        distances_sq = np.sum(
            differences ** 2, axis=1
        )

        if i == 0:
            dp = distances_sq.copy()
            predecessors[i] = np.arange(
                m, dtype=np.int32
            )
            continue

        prefix_cost = np.minimum.accumulate(dp)

        prefix_index = np.empty(
            m, dtype=np.int32
        )

        best_index = 0
        best_cost = dp[0]

        for j in range(m):
            if dp[j] < best_cost:
                best_cost = dp[j]
                best_index = j

            prefix_index[j] = best_index

        dp = prefix_cost + distances_sq
        predecessors[i] = prefix_index

    positions = np.empty(
        n, dtype=np.int32
    )

    positions[-1] = int(np.argmin(dp))

    for i in range(n - 1, 0, -1):
        positions[i - 1] = predecessors[
            i, positions[i]
        ]

    if np.any(np.diff(positions) < 0):
        raise AssertionError(
            "Route alignment moved backwards."
        )

    return pd.DataFrame({
        "trip_id": [trip.trip_id] * n,
        "obs_idx": np.arange(n),
        "route_position": positions,
        "reference_arc_id": route[positions],
    })
