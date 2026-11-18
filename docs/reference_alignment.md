
# Reference Route Alignment and Candidate Labeling

## Objective

Assign GPS observations to reference road arcs and generate
binary labels for candidate road segments.

## Reference Alignment

The `.route` file contains an ordered sequence of road arcs,
not one label per GPS observation.

We use dynamic programming to align GPS observations to
positions in the reference route.

### Method

1. Load GPS observations and reference route arcs.
2. Convert geographic coordinates into metric UTM coordinates.
3. Calculate squared point-to-segment distances.
4. Use dynamic programming to minimize total alignment cost.
5. Enforce non-decreasing reference route positions.
6. Assign the aligned reference arc ID to each observation.

The algorithm preserves route order, including repeated arc IDs.

Time complexity: O(N × M)
Memory complexity: O(N × M)

N = number of GPS observations
M = number of reference route arcs

## Candidate Labeling

For each GPS observation and candidate arc:

- Label 1: candidate arc matches the aligned reference arc.
- Label 0: candidate arc does not match.

## Candidate Miss Rate

A candidate miss occurs when the aligned reference arc is
absent from the candidate set.

Miss Rate = Missed Observations / Total Observations

We evaluate radii of 30, 50, 100, and 200 metres,
with a maximum of 30 candidates per observation.

Observations with zero candidates count as misses.

## Limitations

- Alignment minimizes geometric distance, not travel time.
- GPS noise, route loops and gaps can create ambiguous matches.
- Route positions are monotonic, but large forward jumps are allowed.
- Candidate coverage depends on the 30-candidate cap.
- Alignment-derived labels should be reviewed before training.
- Only training trips are used to generate training labels.
