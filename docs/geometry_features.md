# Geometry Features – Data Dictionary (Milestone 2 · Person 3)

Produced by `pathsnap.features.geometry.compute_geometry_features()`; analysis in
`notebooks/05_geometry_features.ipynb`.

## Input

One row per (GPS observation, candidate arc) from Person 1's `pathsnap.features.candidates.generate_candidates()`:

| Column | Type | Meaning |
|---|---|---|
| `trip_id` | str | Trip identifier (e.g. `00000001`) |
| `obs_idx` | int | 0-based row index of the GPS point in the trip's `.track` |
| `arc_id` | int | 0-based row index of the candidate arc in the trip's `.arcs` (original ID, never re-indexed) |
| *other* | – | Passed through unchanged (e.g. Person 1's `distance_m`, which equals `dist_to_segment_m`) |

Row order is preserved and no row is dropped. An `obs_idx` / `arc_id` outside the trip raises `IndexError`.

## Coordinate conventions

* All lengths and distances are **metres in the trip's UTM zone** (Member 1's loader: `track_utm`, `nodes_utm`, `utm_crs`).
* Bearings are **grid bearings**: degrees clockwise from UTM grid north, in `[0, 360)`, computed as `atan2(dx, dy)`.
* The GPS heading and the arc bearing are both on the same grid, so their difference does not depend on grid convergence
  (checked per trip in notebook 05: the convergence varies by < 0.9° inside any trip).
* Each arc is a straight segment from `from_node_id` to `to_node_id`.

## Per-candidate features

| Column | Type | Unit / range | Definition | Missing values |
|---|---|---|---|---|
| `dist_to_segment_m` | float | m, ≥ 0 | Distance from the GPS point to the closest point of the arc segment (perpendicular foot, or the nearer endpoint if the foot falls outside) | never |
| `proj_offset_m` | float | m, `[0, seg_length_m]` | Distance along the arc from its start node to the closest point | never |
| `proj_fraction` | float | `[0, 1]` | `proj_offset_m / seg_length_m` (0 = start node, 1 = end node); 0 for zero-length arcs | never |
| `proj_remaining_m` | float | m | `seg_length_m − proj_offset_m`: distance left to the end node | never |
| `proj_is_endpoint` | bool | – | True if the perpendicular foot falls before the start or after the end of the segment (closest point is a node), or the arc has zero length | never |
| `gps_heading_deg` | float | °, `[0, 360)` | Grid heading of the vehicle at the point: central difference `p[i−1] → p[i+1]`, falling back to `p[i] → p[i+1]`, then `p[i−1] → p[i]`, using the first displacement ≥ 2 m | NaN if the vehicle moved < 2 m in all three |
| `heading_valid` | bool | – | `gps_heading_deg` is not NaN | never |
| `arc_bearing_deg` | float | °, `[0, 360)` | Grid bearing of the arc from start node to end node | NaN for zero-length arcs |
| `bearing_diff_deg` | float | °, `[0, 180]` | Smallest angle between GPS heading and arc bearing. ≈ 0 driving along the arc, ≈ 180 driving against it | NaN if heading invalid or zero-length arc |
| `bearing_diff_undirected_deg` | float | °, `[0, 90]` | `min(bearing_diff_deg, 180 − bearing_diff_deg)`: alignment ignoring direction | as above |

## Per-arc features (same value for every candidate on the arc)

| Column | Type | Unit / range | Definition |
|---|---|---|---|
| `seg_length_m` | float | m, ≥ 0 | UTM length of the arc segment |
| `arc_is_zero_length` | bool | – | Start and end node have identical coordinates (includes self-loops) |
| `arc_is_two_way` | bool | – | The reverse arc `to → from` also exists |
| `arc_parallel_count` | int | ≥ 1 | Number of arcs with the same ordered `(from, to)` pair (> 1 = duplicate arcs) |
| `from_in_degree` | int | ≥ 0 | Arcs entering the start node |
| `from_out_degree` | int | ≥ 1 | Arcs leaving the start node (includes this arc) |
| `to_in_degree` | int | ≥ 1 | Arcs entering the end node (includes this arc) |
| `to_out_degree` | int | ≥ 0 | Arcs leaving the end node: number of ways to continue (0 = dead end) |
| `arc_in_largest_scc` | bool | – | Both endpoints are in the trip network's largest strongly connected component |
| `arc_wcc_fraction` | float | `(0, 1]` | Size of the arc's weakly connected component ÷ number of nodes in the trip network (small = isolated fragment) |

Degrees are counted on the directed multigraph: parallel arcs count separately and a self-loop adds 1 to both in- and
out-degree.

## Leakage / scaling

No feature is fitted, learned or scaled; every value depends only on the trip's own GPS points and road network.
Labels (`.route`) are not used. Any normalisation must be fitted on training trips only at the assembly step (Person 4).

## Helper functions

| Function | Use |
|---|---|
| `compute_arc_static_features(nodes_utm, arcs, G=None)` | Arc-level table, once per trip (≈ 6 s on the largest 307k-arc network) |
| `compute_gps_heading(x, y, min_move_m=2.0)` | Per-observation heading; Person 4 can reuse it for sequence features |
| `compute_geometry_features(candidates, track_utm, arc_static)` | Adds all columns above to one trip's candidates |
| `check_projected_geometry(trip)` | Verifies UTM CRS, length scale and grid-convergence spread |
| `grid_bearing_deg`, `angle_diff_deg` | Angle helpers |
