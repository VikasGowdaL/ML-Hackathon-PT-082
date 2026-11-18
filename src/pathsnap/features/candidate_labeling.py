
"""Binary candidate labels and candidate coverage evaluation."""

import pandas as pd


def label_candidates(candidates, alignment):
    """
    Label each candidate arc using the aligned reference arc.

    Required candidates columns:
        trip_id, obs_idx, arc_id

    Required alignment columns:
        trip_id, obs_idx, reference_arc_id

    Returns:
        Original candidate columns plus reference_arc_id and label.
    """
    candidate_cols = {"trip_id", "obs_idx", "arc_id"}
    alignment_cols = {"trip_id", "obs_idx", "reference_arc_id"}

    if not candidate_cols.issubset(candidates.columns):
        raise ValueError("Candidate columns are missing.")

    if not alignment_cols.issubset(alignment.columns):
        raise ValueError("Alignment columns are missing.")

    keys = ["trip_id", "obs_idx"]

    if alignment.duplicated(keys).any():
        raise ValueError("Duplicate alignment observations.")

    duplicate_keys = keys + ["arc_id"]

    if "radius_m" in candidates.columns:
        duplicate_keys.append("radius_m")

    if candidates.duplicated(duplicate_keys).any():
        raise ValueError(
            "Duplicate candidate arcs within the same observation and radius."
        )

    result = candidates.merge(
        alignment[keys + ["reference_arc_id"]],
        on=keys,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if (result["_merge"] != "both").any():
        raise ValueError("Some candidates have no reference alignment.")

    result = result.drop(columns=["_merge"])

    result["label"] = (
        result["arc_id"] == result["reference_arc_id"]
    ).astype(int)

    return result


def calculate_miss_rate(labeled_candidates, alignment):
    """
    Calculate candidate miss rate by trip and search radius.

    A miss occurs when the reference arc is absent from the
    candidate set for an observation.

    Requires a 'radius_m' column in labeled_candidates.
    """
    if "radius_m" not in labeled_candidates.columns:
        raise ValueError("Missing radius_m column.")

    if labeled_candidates["radius_m"].isna().any():
        raise ValueError("Radius cannot be null.")

    if labeled_candidates["radius_m"].le(0).any():
        raise ValueError("Radius must be positive.")

    observations = alignment[
        ["trip_id", "obs_idx"]
    ].drop_duplicates()

    radii = sorted(
        labeled_candidates["radius_m"].unique()
    )

    if not radii:
        raise ValueError("No search radii available.")

    results = []

    for radius in radii:
        subset = labeled_candidates[
            labeled_candidates["radius_m"] == radius
        ]

        hits = (
            subset.groupby(["trip_id", "obs_idx"])["label"]
            .max()
            .reset_index(name="hit")
        )

        coverage = observations.merge(
            hits,
            on=["trip_id", "obs_idx"],
            how="left",
        )

        coverage["hit"] = coverage["hit"].fillna(0)

        summary = (
            coverage.groupby("trip_id")
            .agg(
                total_observations=("hit", "size"),
                matched_observations=("hit", "sum"),
            )
            .reset_index()
        )

        summary["radius_m"] = radius

        summary["missed_observations"] = (
            summary["total_observations"]
            - summary["matched_observations"]
        )

        summary["miss_rate"] = (
            summary["missed_observations"]
            / summary["total_observations"]
        )

        results.append(summary)

    return pd.concat(results, ignore_index=True)


def find_worst_trips(miss_rate_summary, top_n=10):
    """
    Return the trips with the highest candidate miss rates
    for each search radius.
    """
    required = {
        "trip_id",
        "radius_m",
        "miss_rate",
        "missed_observations",
    }

    if not required.issubset(miss_rate_summary.columns):
        raise ValueError("Missing required miss-rate columns.")

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    return (
        miss_rate_summary.sort_values(
            ["radius_m", "miss_rate", "missed_observations"],
            ascending=[True, False, False],
        )
        .groupby("radius_m", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
