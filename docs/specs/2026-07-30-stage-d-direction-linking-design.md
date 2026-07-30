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

---

## Post-implementation calibration results

*Added after implementing the above, reviewing it, fixing the review's
findings, and re-running the calibration. These are the real numbers.*

### Headline

**Direction detection does not work well enough to trust.** The Calibration
section above anticipated this outcome ("If held-out error is large or
unstable across folds, that is itself a reportable finding … it would mean
the feature set isn't sufficient"). That is where we landed. The matched-pair
*volume* is in the right ballpark; the `direction` label is only weakly
correlated with ground truth and is badly attenuated toward 50/50 on exactly
the dates where it would matter most.

An initial implementation was worse still, for a bug rather than a data
reason: `rssi_shape_score` was implemented as an unbounded **sum** of the two
anchors' rectified `rssi_trend` magnitudes rather than a bounded similarity.
With real `rssi_trend` reaching ±16 against a timing term capped at 1.0, and
a calibrated `rssi_weight` of 5.0, RSSI steepness alone decided every match
and timing was irrelevant; 18.2% of the shipped rows had a *negative*
`transit_gap_s` (median ~88 minutes backwards, worst case 11.2 hours). The
numbers below are from the corrected implementation: a bounded [0,1]
agreement score, a hard 300s maximum-transit gate, below-threshold edges
forbidden before rather than after the assignment is solved, and the manual
log's label orientation fitted inside each CV fold instead of once over all
8 dates.

### Calibrated parameters

Grid: 1512 points over `transit_center_s` ∈ {10…50}, `transit_spread_s` ∈
{8…60}, `rssi_weight` ∈ {0.0…2.0}, `min_match_score` ∈ {0.05…1.1};
`max_transit_s` fixed at 300s as a physical gate rather than a fitted value.

```
final_params = {'transit_center_s': 10.0, 'transit_spread_s': 60.0,
                'rssi_weight': 0.0, 'min_match_score': 0.05,
                'max_transit_s': 300.0}
label orientation: flip=True   (in-sample loss 12412 vs 23146 for flip=False;
                                chosen identically in all 8 CV folds)
in-sample total loss: 12412    mean held-out loss: 2205.5
```

**`rssi_weight` fitted to exactly 0.0.** The RSSI-shape feature — the whole
of §2's "signal fading at A while strengthening at B" idea — earns zero
weight when it is put on a fair, bounded footing against timing and allowed
to compete. The calibration's verdict is that it adds nothing.

All four fitted values sit on a grid edge, but that is **not** a truncated
grid: probing explicitly out-of-grid settings (centre 5s/spread 150s, centre
1s/spread 1000s, and the flat limit spread 1e6 with threshold 1e-9) gives
*worse* in-sample loss (13412, 13466, 13694) and predictions that converge to
a fixed point. The corner is a **saturation plateau** — beyond it the score
is effectively constant across the whole `(0, 300]s` gate and the matcher can
only "pair everyone it can", which it already does.

### Why the fit ran to that corner

Two of the eight GT dates have a manual total larger than the maximum number
of pairs a bipartite matching could possibly produce:

| date | n_A travelers | n_B travelers | max possible matches | manual total |
|---|---|---|---|---|
| 2026-05-19 | 124 | 140 | 124 | **138** |
| 2026-05-31 | 33 | 25 | 25 | **56** |

On those dates *no* parameter setting can reach the target, so their squared
error is irreducible and large, and it dominates the summed loss. The
optimizer's only lever is to match as many people as it can — which is
exactly the corner it chose. Stage C simply did not resolve enough distinct
travelers on those days for the manual counts to be reachable; that is a
Stage A–C recall issue surfacing here, not a Stage D matching issue.

There is also a flaw in the loss defined in §5 above. Writing `a` and `b` for
the two directional errors, the `total` error is `a + b`, so
`a² + b² + (a+b)² = 2a² + 2b² + 2ab`. A purely *directional* mistake
(`a = -e`, `b = +e`, total correct) costs `2e²`, while a purely *volumetric*
one (`a = b = e`) costs `6e²`. **The loss weights total-count error three
times more heavily than direction error**, so the calibration was always
going to buy volume at direction's expense. The `total` term should be
dropped, or the directional terms up-weighted, in any next iteration.

### Leave-one-date-out results (held-out, params + orientation fitted on the other 7)

| date | pred A→B | pred B→A | pred total | manual A→B | manual B→A | manual total | held-out loss |
|---|---|---|---|---|---|---|---|
| 2026-04-11 | 30 | 7 | 37 | 27 | 1 | 28 | 126 |
| 2026-05-02 | 36 | 39 | 75 | 20 | 36 | 56 | 626 |
| 2026-05-09 | 148 | 110 | 258 | 181 | 56 | 237 | 4446 |
| 2026-05-16 | 25 | 17 | 42 | 22 | 2 | 24 | 558 |
| 2026-05-19 | 61 | 48 | 109 | 136 | 2 | 138 | 8582 |
| 2026-05-23 | 18 | 29 | 47 | 30 | 13 | 43 | 416 |
| 2026-05-27 | 11 | 10 | 21 | 17 | 0 | 17 | 152 |
| 2026-05-31 | 15 | 8 | 23 | 55 | 1 | 56 | 2738 |

Mean held-out loss **2205.5** (previous, pre-fix run: 2781.2).

### Direction skew — the check that actually matters

Predicted vs. manual A→B *share* on each held-out date:

| date | predicted A→B share | manual A→B share | error |
|---|---|---|---|
| 2026-04-11 | 0.811 | 0.964 | −0.153 |
| 2026-05-02 | 0.480 | 0.357 | +0.123 |
| 2026-05-09 | 0.574 | 0.764 | −0.190 |
| 2026-05-16 | 0.595 | 0.917 | −0.322 |
| 2026-05-19 | 0.560 | 0.986 | −0.426 |
| 2026-05-23 | 0.383 | 0.698 | −0.315 |
| 2026-05-27 | 0.524 | 1.000 | −0.476 |
| 2026-05-31 | 0.652 | 0.982 | −0.330 |

- Mean |share error| **0.292**
- Always-predict-50/50 baseline **0.369**
- corr(predicted share, manual share) across the 8 dates **0.533**
  (n = 8, so not significant — p ≈ 0.17)

Read honestly: the fixed implementation is **better than a coin flip but not
by much, and not reliably**. It beats the 50/50 baseline by 0.077 in mean
absolute share error, and the rank ordering is loosely right (the one date
whose true traffic runs the other way, 2026-05-02, is also the date with the
lowest predicted A→B share). But every prediction is compressed into roughly
[0.38, 0.81] while the truth spans [0.36, 1.00], seven of the eight errors
point the same way (toward 50/50), and on 2026-05-19 — 98.6% one-way in the
manual log — the model predicts 56%. On 2026-05-27 (100% one-way) it predicts
52%.

This is not a tuning problem. Holding the *shape* of the matcher fixed and
sweeping to much tighter transit windows does not recover the skew: at
`spread=8s, min_match_score=0.5` the mean share error is 0.284; at
`spread=12s, min_match_score=0.8` it is 0.271; at `spread=8s,
min_match_score=1.1` it is 0.392 (worse, because too few pairs survive). The
attenuation is essentially flat across every operating point.

The mechanism is plain enough. For a *correct* pair the direction is
unambiguous — one direction's gap is positive and the other's is negative by
construction. The problem is that most pairs are not correct. On 2026-05-09
there are 265 Sensor-A and 268 Sensor-B travelers inside a 58-minute window,
i.e. ~4.5 arrivals per minute per sensor, so within any 300s gate there are
~20 equally plausible partners on each side. `(anchor timestamp, rssi_trend)`
does not distinguish between them, the assignment pairs people essentially by
temporal proximity, and whether a randomly-paired A traveler's last sighting
happens to precede or follow a randomly-paired B traveler's first sighting is
close to a coin toss. A weak true signal is diluted by a large majority of
arbitrary pairs.

### What this means for consumers of the output

- `DATA/travelers_matched_direction.csv` / `.parquet` now include unmatched
  single-sensor travelers with `direction = NaN` (§6), and every matched
  row's `transit_gap_s` is guaranteed to lie in (0, 300]s — the previous
  run's 18.2% of negative, multi-hour gaps are gone.
- The **`direction` column should not be used for per-person direction
  claims, and only cautiously for aggregate skew.** It carries a weak,
  attenuated signal that systematically understates one-way traffic.
- `match_score` is a plausibility score on the scale
  `[min_match_score, 1 + rssi_weight]`, not a probability. With the
  calibrated `rssi_weight = 0.0` it is exactly `transit_time_score`, i.e. a
  Gaussian in the transit gap.

### Suggested next steps, in priority order

1. **Fix the loss** (drop the redundant `total` term, or up-weight the
   directional terms) so the calibration is allowed to trade volume for
   direction. Cheapest change with the clearest rationale.
2. **Address the two unreachable dates.** The manual totals exceed Stage C's
   available traveler counts, so the fit is anchored to irreducible error.
   This points back at Stage A–C recall (and at the deferred single-sensor
   block-gap issue in "Out of scope" above), not at Stage D.
3. **Only then** consider the richer pairwise classifier mentioned in §5 —
   the escalation was explicitly conditioned on seeing this result first, and
   more features on top of a mis-specified loss and an unreachable target
   would be premature.
