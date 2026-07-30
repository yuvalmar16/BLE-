# Stage D: Cross-Sensor Identity Linking & Direction — Design

## Context

This is Stage D of the hiker-monitoring pipeline described in
`hiker_monitoring_project_summary.md`. Stage C resolved multiple MAC
addresses into a single `Traveler_ID`, but **independently per sensor** —
it never attempted to connect a traveler's Sensor A presence to their
Sensor B presence. Stage D's job is exactly that connection, plus
deriving walking direction from it.

This design was motivated by a new asset: `DATA/combined_file_manual.csv`,
a human observer's manual click-counter log (direction `A`/`B`, one row
per person, at second resolution) covering 8 short windows across 8
distinct dates (17–237 people each). This is the project's first and only
ground truth. It exists specifically to validate Stage D, not Stages A–C.

## Key findings from ground-truth exploration (motivating this design)

Before designing the algorithm, the manual data was used to test three
naive approaches, all of which failed for identifiable reasons:

1. **Summing Sensor A's and Sensor B's independent traveler counts is
   wrong.** Cross-checking raw MAC overlap between sensors in each GT
   window showed 60–90% of one sensor's travelers are also seen by the
   other sensor within the same short window (consistent with the ~20m
   physical gap between sensors, a ~20–30s walking transit time). Sensor A
   and Sensor B counts are two mostly-overlapping views of the *same*
   people, not disjoint populations — so neither a single sensor's count,
   nor their sum, estimates true foot traffic.

2. **Raw-MAC-address matching between sensors over-counts.** Matching
   Sensor A and B readings by literal MAC equality in each GT window
   produced union totals 2–4x the manual ground truth in most windows —
   worse than either sensor's own (already-inflated) traveler count. MAC
   randomization mid-transit (the same physical device presenting a
   different MAC to Sensor B than it showed Sensor A moments earlier) and
   simple MAC-level rather than trip-level matching both amplify
   overcounting when used naively.

3. **"Which sensor detected the MAC first" is not a usable direction
   signal.** Comparing the earliest raw timestamp each shared MAC was seen
   at each sensor, and correlating the resulting A-first/B-first split
   against the manual direction counts, showed the *opposite* skew from
   ground truth on multiple dates (e.g. 2026-05-09: raw signal says 3:1
   toward "A first", manual says 1:3 toward direction B). This is
   consistent with Sensor A having a longer/more sensitive detection
   radius than Sensor B, so it systematically "sees" devices earlier
   regardless of true walking direction — a range-asymmetry confound, not
   a direction signal.

These three findings directly shape the design below: match at the
traveler level (not raw MAC) using signal shape (not MAC equality) at the
*boundary* of each sensor's view (not first-ever-seen), and calibrate
against the only ground truth available rather than assume a threshold.

## Scope

In scope: linking `DATA/travelers_sensor_A.csv` and
`DATA/travelers_sensor_B.csv` (Stage C output) plus their underlying
`sessions_sensor_{A,B}_with_traveler` tables, for the 14 dates with data
from both sensors. Producing a matched-pair table with a direction label,
calibrated against `DATA/combined_file_manual.csv`.

Out of scope (explicitly deferred):
- The case where a device disappears from a *single* sensor's view for
  more than Stage B/C's 60s block-gap threshold and reappears under a new
  MAC. This never reaches Stage D today, because Stage C's blocking
  already split it into two unrelated, un-comparable blocks before this
  stage ever sees the data. Fixing it means revisiting Stage B/C's
  gap threshold or blocking logic, not Stage D's cross-sensor matching.
- Any use of the manual data's *absolute* counts as a global calibration
  target beyond the 8 GT windows (e.g. extrapolating a fixed correction
  factor to apply project-wide) — the 8 windows are for fitting Stage D's
  few parameters only, not for re-scaling Stage A–C's output.

## Method

### 1. Matching unit and anchors

Match at the **Traveler level** (Stage C's already MAC-consolidated
output), not raw sessions or MACs. For each traveler, use the session
**closest to the boundary with the other sensor** as the comparison
anchor, not the traveler's overall visit span:

- Sensor A traveler → their **last** session (`end_time`, that session's
  `mean_rssi`/`rssi_trend`) — the reading nearest to leaving A's range.
- Sensor B traveler → their **first** session (`start_time`, that
  session's `mean_rssi`/`rssi_trend`) — the reading nearest to entering
  B's range. (And symmetrically for the B→A direction.)

This is the direct fix for finding 3 above: comparing the moment of
transition between sensors, rather than each traveler's first-ever
detection, removes the range-asymmetry confound.

### 2. Candidate pairs and features

Within each date, compare every Sensor A traveler against every Sensor B
traveler whose visit windows are plausibly close (a coarse time-window
prefilter, generous enough not to exclude real matches, to keep the pair
count tractable on busy days). For each candidate pair compute:

- `transit_gap_s`: seconds between the two anchor timestamps
  (A's-last→B's-first, or B's-last→A's-first — whichever is positive and
  smaller). This also fixes the candidate's direction.
- `rssi_shape_score`: similarity between "A's signal fading out" and "B's
  signal fading in" at the anchors (e.g. comparing `rssi_trend` sign/
  magnitude and the anchor `mean_rssi` values) — a soft, continuous score,
  **not** a MAC-equality check. This is the fix for finding 2.

### 3. Matching: optimal bipartite assignment

Build a bipartite graph (Sensor A travelers × Sensor B travelers) per
date, edge weight = a combination of a transit-time plausibility score
(peaked around the expected ~20–30s transit, given the 20m sensor
spacing) and `rssi_shape_score`. Solve with optimal bipartite assignment
(`scipy.optimize.linear_sum_assignment`) rather than greedy nearest-match
— greedy can lock in a suboptimal pair and block a better one; optimal
assignment is the principled tool for a two-sided matching problem and is
cheap at this data scale. Edges below a minimum score are forbidden;
travelers left unmatched are reported as "seen by only one sensor," not
forced into a low-confidence match.

### 4. Direction

For each matched pair, direction is whichever anchor timestamp is
earlier: A's-last-seen before B's-first-seen → direction A→B, and vice
versa. No direction is assigned to unmatched travelers.

### 5. Calibration against ground truth (the ML/fitting step)

Free parameters (small, on purpose, given only 8 GT windows): the
transit-time window's center/spread, the `rssi_shape_score` weight
relative to timing, and the minimum match-score threshold.

- **Loss** per GT date: squared error between predicted
  `(A→B count, B→A count, matched total)` and the manual ground truth's
  `(A, B, A+B)` for that date's window.
- **Fitting**: leave-one-date-out cross-validation across the 8 GT dates
  — tune parameters on 7, evaluate held-out error on the 8th, rotate
  through all 8. Report both in-sample and held-out error explicitly, so
  the writeup doesn't overstate generalization from a fit to 8 numbers.
- If held-out error is large or unstable across folds, that is itself a
  reportable finding (it would mean the feature set isn't sufficient),
  not a failure to hide — the next escalation (a richer pairwise
  classifier, per the earlier brainstorm) would be considered only after
  seeing this result, not preemptively.

### 6. Output

- `DATA/travelers_matched_direction.csv` / `.parquet`: one row per matched
  pair (`Traveler_ID_A`, `Traveler_ID_B`, `direction`, `transit_gap_s`,
  `match_score`), plus rows for unmatched travelers on either side
  (`direction = NULL`), across all 14 dual-sensor dates.
- A validation table/report comparing predicted vs. manual counts for the
  8 GT dates (in-sample and leave-one-out), in the same style already
  used in this conversation.

## Validation / sanity checks (in-notebook)

- The three ground-truth findings above, reproduced as notebook cells
  (not just asserted in this doc), so the "why" of the design is visible
  next to the code, matching Stages A–C's convention of documenting
  discovered issues in-notebook.
- Leave-one-date-out error table (8 rows: held-out date, predicted vs.
  actual A/B/total, error) — the primary evidence of whether calibration
  generalizes.
- Distribution of `match_score` for matched vs. unmatched candidate pairs,
  to sanity-check the threshold sits at a reasonable separation point
  (same "inspect the distribution, document the choice" style as Stages
  A–C's threshold decisions).
- Spot-check a handful of matched pairs directly (print both travelers'
  anchor sessions side by side) for physical plausibility, the same way
  Stage C spot-checks multi-device travelers.

## Deliverable

`notebooks/04_stage_d_direction_linking.ipynb`:
1. Load Stage C traveler + session tables (both sensors) and the manual
   ground-truth file.
2. Reproduce the three ground-truth findings (sum-double-counts, raw-MAC
   over-counts, first-seen-order is confounded) as documented evidence.
3. Compute per-traveler boundary anchors (last session for A, first for
   B, and the reverse).
4. Generate candidate pairs + features within each date.
5. Bipartite assignment (parameterized), producing matched pairs +
   direction.
6. Calibration: leave-one-date-out fit over the 8 GT dates, with an
   explicit in-sample vs. held-out error report.
7. Apply the calibrated parameters to all 14 dual-sensor dates.
8. Write output CSV + Parquet.

## Out of scope for this design

Extending direction/linking to dates with only one sensor's data (7 of 30
collection days) is impossible by construction — there is nothing to
link to. The single-sensor block-gap/disappear-reappear issue noted above
is deferred to a possible Stage B/C revision, not part of this stage.
