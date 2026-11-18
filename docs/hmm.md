# HMM / Viterbi Route Reconstruction

Code: `src/pathsnap/hmm/viterbi.py` (decoder), `src/pathsnap/hmm/route_reconstruction.py` (shortest-path bridging),
`src/pathsnap/pipeline.py` (end-to-end use). Results: `outputs/results/pipeline_student_test.csv` and
`pipeline_student_test_summary.json` (`python scripts/evaluate_pipeline.py`), maps in `outputs/maps/prediction_*.html`.

## 1. Hidden states

For each GPS observation *t*, the hidden states are its **candidate road arcs**: every arc of the trip's road network
within 50 m of the point, at most the 30 nearest (`generate_candidates`, `configs/features.yaml`). The decoder chooses one
candidate arc per observation; arc IDs are the original row indices of `.arcs`.

## 2. Emission scores

`emission(t, a) = emission_weight · log(max(p(t, a), 1e-9))` with `emission_weight = 1`, where `p(t, a)` is the final
model's probability that arc *a* is the true arc for point *t* (LightGBM `predict_proba`, `artifacts/models/final_model.joblib`).
The probabilities are used as relative scores; they are not renormalised per observation.

## 3. Transition scores

Log-domain score between the arc chosen at *t−1* (`a`) and at *t* (`b`), using the directed road graph:

| Relation | Score | `ViterbiConfig` field |
|---|---|---|
| Same arc (`a == b`) | 0 | `same_arc_bonus = 0` |
| Connected head-to-tail (`to_node(a) == from_node(b)`) | 0 | `connected_bonus = 0` |
| Anything else | −5 (≈ factor e⁻⁵ ≈ 0.007) | `disconnected_penalty = 5` |

So the decoder prefers sequences that stay on an arc or move to a directly connected arc, but can still jump when the
emission evidence is strong. Road distance versus GPS distance and travel direction are **not** used in the transition
(the direction is already inside the emission via the `bearing_diff_deg` feature).

## 4. Missing candidates and large GPS gaps

* An observation with **no candidate within 50 m** gets no prediction (`status = no_candidates`); the Viterbi chain is
  closed before it and restarted after it.
* The chain is also **restarted** at observations flagged `after_gap` (time gap to the previous point > 10 s) or
  `is_segment_start` (gap > 60 s; `configs/features.yaml`). Each restart starts a new route section.

## 5. Disconnected consecutive arcs

After decoding, consecutive predicted arcs are joined into a route (`reconstruct_route`):

* identical arcs are merged;
* connected arcs are appended;
* otherwise the **shortest directed road path** (by length) between the end of the first arc and the start of the next
  is inserted (bridge arcs), if one exists and is ≤ 2 km;
* if no such path exists, the pair is recorded as an **unbridged break** and a new route section starts. Breaks are never
  drawn as straight lines.

## 6. Did HMM decoding improve the route? (student_test, 10 trips, final model)

| | Classifier only (highest probability per point) | Classifier + Viterbi |
|---|---|---|
| Top-1 over all 19,160 GPS points | 0.8561 | **0.8763** |
| Consecutive predicted arcs connected | 0.9238 | **0.9960** |

Per trip:

| Trip | Points | Top-1 ML | Top-1 ML+HMM | Gain | Route F1 | Precision | Recall | Sections | Unbridged breaks |
|---|---|---|---|---|---|---|---|---|---|
| 00000000 | 2503 | 0.9393 | 0.9612 | +0.0220 | 0.9875 | 0.9768 | 0.9985 | 1 | 0 |
| 00000007 | 1354 | 0.9180 | 0.9217 | +0.0037 | 0.9853 | 0.9777 | 0.9930 | 1 | 0 |
| 00000009 | 1049 | 0.8580 | 0.8656 | +0.0076 | 0.9329 | 0.8879 | 0.9828 | 1 | 0 |
| 00000040 | 2995 | 0.9199 | 0.9295 | +0.0097 | 0.9402 | 0.8872 | 1.0000 | 1 | 0 |
| 00000052 | 262 | 0.9504 | 0.9427 | **−0.0076** | 0.9556 | 0.9468 | 0.9645 | 3 | 2 |
| 00000056 | 3958 | 0.8871 | 0.9156 | +0.0285 | 0.9556 | 0.9349 | 0.9773 | 10 | 9 |
| 00000065 | 768 | 0.8255 | 0.8581 | +0.0326 | 0.9433 | 0.8927 | 1.0000 | 4 | 0 |
| 00000076 | 1486 | 0.8668 | 0.8957 | +0.0289 | 0.8745 | 0.7840 | 0.9887 | 25 | 0 |
| 00000077 | 3911 | 0.6970 | 0.7154 | +0.0184 | 0.8171 | 0.6908 | 1.0000 | 1 | 0 |
| 00000091 | 874 | 0.8535 | 0.8982 | +0.0446 | 0.9353 | 0.8823 | 0.9951 | 1 | 0 |

**Where it helped:** 9 of 10 trips; the largest gains (+0.03 to +0.045) are on trips where the classifier alternates
between neighbouring or opposite-direction arcs (00000091, 00000065, 00000076, 00000056). Continuity rises to ≥ 0.98 on
every trip, so the output is a drivable route instead of a set of independent guesses.

**Where it did not help:**

* **00000052** (short trip, 262 points): Top-1 drops by 0.0076. With only a −5 penalty and no distance term, the decoder
  occasionally prefers a connected but wrong arc over the classifier's correct choice; the trip also has 2 unbridged
  breaks.
* **00000077:** Top-1 improves only to 0.715 and route precision is 0.69 although recall is 1.0. The classifier is
  often wrong in the same, consistent way, so Viterbi follows a plausible but wrong road for long stretches, and bridges
  add extra arcs. The map `outputs/maps/prediction_00000077.html` shows these detours.
* **00000056 / 00000076:** many route sections (10 and 25) from GPS gaps and 9 unbridged breaks in 00000056; the
  route is correct in each section but not one continuous path.

## Limitations and possible improvements

* Transition scores ignore the road distance between arcs compared with the GPS distance, and the time between points;
  a distance-based transition (e.g. exponential in |road distance − GPS distance|) would penalise implausible detours.
* `disconnected_penalty` and `emission_weight` were not tuned (they could be tuned on train trips only).
* Probabilities from the class-balanced model are inflated; per-observation normalisation was not used.
