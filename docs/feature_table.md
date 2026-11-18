# Candidate Feature Table – Data Dictionary (Milestone 2)

Built by `scripts/build_feature_table.py` (configuration: `configs/features.yaml`) using
`pathsnap.features.assembly`. Analysis: `notebooks/06_feature_assembly.ipynb`.

## Files

| File | Content | In git? |
|---|---|---|
| `outputs/features/feature_table_train.parquet` | 80 training trips, labelled | no (`*.parquet` is git-ignored; regenerate with the script, ~2–3 min) |
| `outputs/features/feature_table_student_test.parquet` | 10 student-test trips, `label` / `reference_arc_id` null (see below) | no |
| `outputs/features/feature_table_summary.csv` | Per trip: rows, observations, observations with candidates, positives, segments | yes |
| `outputs/features/scaler_train.json` | Mean / std of every continuous feature, fitted on **training rows only** | yes |

One row = one (GPS observation, candidate arc) pair. One parquet row group = one trip. Training and student-test
trips are written to separate files, so the training table cannot contain a student-test trip (tested).

## Sources

| Part | Owner | Module |
|---|---|---|
| Candidates (radius 50 m, cap 30) | Person 1 | `features/candidates.py` |
| Reference arc + label | Person 2 | `features/reference_alignment.py`, `features/candidate_labeling.py` |
| Geometry features | Person 3 | `features/geometry.py` (details: `docs/geometry_features.md`) |
| Sequence + rank features, assembly | Person 4 | `features/sequence.py`, `features/assembly.py` |

All distances are metres in the trip's UTM projection, times are seconds, angles are degrees on the UTM grid.

## Columns

### Keys and labels

| Column | Type | Description |
|---|---|---|
| `trip_id` | str | Trip identifier |
| `partition` | str | `train` or `student_test` |
| `obs_idx` | int64 | 0-based row of the GPS point in the trip's `.track` |
| `arc_id` | int64 | 0-based row of the candidate arc in the trip's `.arcs` (original ID) |
| `reference_arc_id` | Int64 (nullable) | Arc the observation is aligned to on the ground-truth route (Person 2's monotonic alignment). Null for student_test |
| `label` | Int8 (nullable) | 1 if `arc_id == reference_arc_id`, else 0. Null for student_test |

Unique key: (`trip_id`, `obs_idx`, `arc_id`), tested to have no duplicates. Observations with no arc within the radius have
no rows; `feature_table_summary.csv` reports how many.

### Candidate distance and rank (Persons 1 and 4)

| Column | Type | Description |
|---|---|---|
| `distance_m` | float32 | Distance from GPS point to arc segment (Person 1) |
| `dist_rank` | int32 | Rank of the candidate by distance within its observation, 1 = nearest (ties share the lowest rank) |
| `n_candidates` | int32 | Number of candidates of the observation |
| `dist_minus_best_m` | float32 | `distance_m` − nearest candidate's distance |
| `dist_ratio_to_best` | float32 | (`distance_m` + 1) / (nearest distance + 1), ≥ 1 |

### Geometry (Person 3)

`dist_to_segment_m, proj_offset_m, proj_fraction, proj_remaining_m, proj_is_endpoint, gps_heading_deg, heading_valid,
arc_bearing_deg, bearing_diff_deg, bearing_diff_undirected_deg, seg_length_m, arc_is_zero_length, arc_is_two_way,
arc_parallel_count, from_in_degree, from_out_degree, to_in_degree, to_out_degree, arc_in_largest_scc, arc_wcc_fraction`.
Definitions in [`geometry_features.md`](geometry_features.md). In this table `gps_heading_deg` / `heading_valid` are
computed **per segment** (headings never use a point across a split gap), and `bearing_diff_*` use that heading.

### Observation sequence features (Person 4, same value for all candidates of an observation)

| Column | Type | Description | Missing when |
|---|---|---|---|
| `t` | float32 | Timestamp (s) from `.track` | never |
| `dt_prev_s`, `dt_next_s` | float32 | Time to previous / next point | first / last point of a segment |
| `dist_prev_m`, `dist_next_m` | float32 | Straight-line distance to previous / next point | as above |
| `speed_prev_mps`, `speed_next_mps` | float32 | `dist / dt` to previous / next point | as above |
| `speed_mps` | float32 | (dist_prev + dist_next) / (dt_prev + dt_next), using available neighbours | single-point segment |
| `accel_mps2` | float32 | (speed_next − speed_prev) / mean(dt_prev, dt_next) | either neighbour missing |
| `heading_change_deg` | float32 | Turn angle between incoming and outgoing step, [0, 180] | either step < 2 m or neighbour missing |
| `after_gap` | bool | `dt_prev_s` > 10 s (`flag_gap_s`) | – |
| `before_gap` | bool | `dt_next_s` > 10 s | – |
| `segment_id` | int32 | 0-based trip segment; increments when the gap to the previous point > 60 s (`split_gap_s`) | – |
| `is_segment_start`, `is_segment_end` | bool | First / last point of its segment | – |
| `implausible_speed` | bool | Implied speed from the previous point > 70 m/s (`max_speed_mps`) | – |

## Gap policy (proposed by Person 4, to confirm with Person 1)

Over 227,487 GPS intervals in 90 trips: 0.27 % are > 10 s (42 trips) and 0.03 % are > 60 s (23 trips).

* **Flag at 10 s.** These gaps are rare but already break the ~1 Hz assumption; models can use `after_gap` directly.
* **Split at 60 s.** At urban speeds a vehicle can travel ~1 km in 60 s, so the route between the two points is
  ambiguous. The HMM / Viterbi decoder should restart at `is_segment_start` instead of chaining a transition across the gap.
* Values are configurable in `configs/features.yaml`.

## Scaling and leakage

* No feature is fitted; every value depends only on the trip's own GPS points and road network.
* `scaler_train.json` holds mean / std for the continuous features (`SCALE_COLUMNS` in `assembly.py`: everything except keys,
  labels, booleans, `segment_id` and `t`). It is fitted on the training parquet only. `fit_scaler` raises if any non-train
  row is passed. Apply it to any partition with `apply_scaler`.
* Labels for `student_test` are not produced, because Person 2's `align_reference_route` only accepts training trips.
  To evaluate on student_test, the team must decide whether to allow alignment on student_test (the labels are not used
  for fitting, so this is allowed by the leakage rules) and extend that guard.
