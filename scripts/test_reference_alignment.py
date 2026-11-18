
from pathsnap.data.loader import get_trips
from pathsnap.features.reference_alignment import align_reference_route

# Load one training trip without using cached data
trip = get_trips(
    "train",
    use_cache=False,
    project_utm=True,
)[0]

print("Trip ID:", trip.trip_id)
print("GPS observations:", len(trip.track))
print("Reference route arcs:", len(trip.route))
print("UTM CRS:", trip.utm_crs)

# Align GPS observations to the reference route
alignment = align_reference_route(trip)

print("\nFirst 10 aligned observations:")
print(alignment.head(10).to_string(index=False))

print("\nLast 10 aligned observations:")
print(alignment.tail(10).to_string(index=False))

print("\nTotal aligned observations:", len(alignment))

# Validate monotonic route positions
assert alignment["route_position"].is_monotonic_increasing
assert len(alignment) == len(trip.track)

print("\nReference alignment checks passed!")
