
"""Evaluate candidate coverage on one training trip."""

from pathsnap.data.loader import load_trip
from pathsnap.features.candidates import generate_candidates
from pathsnap.features.reference_alignment import align_reference_route
from pathsnap.features.candidate_labeling import (
    label_candidates,
    calculate_miss_rate,
    find_worst_trips,
)

trip = load_trip(
    "00000001",
    partition="train",
    use_cache=False,
    project_utm=True,
)

alignment = align_reference_route(trip)

all_candidates = []
import pandas as pd

for radius in (30, 50, 100, 200):
    candidates = generate_candidates(
        trip,
        radius_m=radius,
        max_candidates=30,
        use_index_cache=False,
    )
    candidates["radius_m"] = radius
    all_candidates.append(candidates)

combined = pd.concat(all_candidates, ignore_index=True)
labeled = label_candidates(combined, alignment)

summary = calculate_miss_rate(labeled, alignment)

print("\nCandidate miss rates:")
print(summary.to_string(index=False))

print("\nWorst-performing trips:")
print(find_worst_trips(summary).to_string(index=False))
