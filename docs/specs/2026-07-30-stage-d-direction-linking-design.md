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

---

## Follow-up investigation 1: richer RSSI-derived features (2026-08-12)

*Motivated by suggested next step 3 above, but run before the loss/target fixes
as an independent empirical question: does `rssi_weight` calibrate to 0.0
because RSSI carries no directional signal, or because the single
whole-session `rssi_trend` feature is too simple to expose it?*

Both sensors' raw per-reading data are now true second-resolution (Stage A's
minute-resolution reconstruction caveat no longer applies — see the Stage A
notebook), which let this investigation recompute all of: RSSI at
first/last detection, max RSSI + its timestamp, local slope near the
start/end of a visit (last-`k`-reading OLS, as opposed to whole-session
OLS), whole-visit slope, strengthen/weaken rates, time-to-peak,
peak-to-last time, and per-sensor z-normalized versions, all rebuilt from
raw readings and verified byte-for-byte against `sessions_sensor_{A,B}.csv`.

**Structural confirmation of the hypothesis behind the test:** 89.8% of
traveler visits (n>2 readings) have an *interior* RSSI peak — the maximum
reading is neither the first nor the last reading of the visit. Local
slopes near a visit's boundary are ~4x steeper than the visit's whole-session
OLS trend at the median (0.4 dB/s vs 0.11 dB/s). A single linear trend over
an approach-then-recede shape genuinely does wash out real signal, exactly
as suspected. Separately: Sensor A and B's raw RSSI distributions are
nearly identical (mean −86.93 vs −86.16 dB, std 9.23 vs 9.20) — the
"different sensitivity/radius" concern motivating per-sensor normalization
doesn't show up as a global offset in this data, so normalization made no
measurable difference here (still good practice if hardware changes).

**But none of the richer features survive held-out validation.** Each was
wired into `direction_candidate_score` as a new optional weighted term
(same convention as `order_weight`) and evaluated LODO:

| feature family | mean held-out loss | Δ vs baseline | mean \|share error\| | corr(pred,manual) |
|---|---|---|---|---|
| baseline (this doc's calibrated matcher, pre-order-signal) | 1045.6 | — | 0.253 | 0.535 |
| peak-time order (A's peak before B's peak) | 1056.9 | +11 | 0.252 | 0.537 |
| local slope shape (start/end, vs. whole-session trend) | 1324.6 | +279 | 0.266 | 0.466 |
| strengthen/weaken rate | 1324.6 | +279 | 0.266 | 0.466 |
| combined (peak + local shape) | 1056.9 | +11 | 0.252 | 0.537 |

(Local-slope and rate scores land on *identical* numbers — they're
redundant, not a bug: both derive from the same underlying RSSI sequence.)
A paired Wilcoxon test on the per-fold share-error deltas gives p=0.875 —
no significant difference from baseline, though n=8 gives this very low
power regardless.

Mechanistically, adding a peak-order term causes large churn in *which* B
traveler gets matched to a given A traveler on busy dates (124/265 pairs
get a different partner on 2026-05-09), but for pairs that stay matched to
the same partner, direction never flips (0/8 dates) — it re-ranks among an
already-large candidate pool rather than supplying new directional
information. A concrete audit on 2026-05-27 (ground truth 100% one-way)
found peak-time order agreed with the baseline's 9 *wrong* B→A calls in
all 9 cases, not the true direction. On the busiest date (2026-05-09) a
fixed peak-order weight did improve that date's loss (730→520), but the
same weight worsened 2026-05-19 (4877→4969) — the signal is real but
sign-inconsistent across dates, which is why a single LODO-fit weight
nets out close to zero.

**Verdict: `rssi_weight = 0.0` is not a feature-poverty artifact.** Three
different summarizations of the same raw signal (whole-session trend,
local slope/rate, peak timing) all fail the same way. Peak-time order is
the least-bad of the three and the only one with an identifiable, if
inconsistent, genuine effect. **Not recommended for production as tested.**
No per-pair ground truth exists to validate cross-sensor *matching*
quality separately from the aggregate directional counts — this limits
every RSSI finding to the directional-count question, not matching
correctness.

---

## Follow-up investigation 2: order-preservation ("no-overtaking") constraint (2026-08-12)

*Tests the assumption that, on this ~20m single-file trail segment,
travelers moving the same direction usually keep their relative order
between sensors A and B.*

**The assumption is strongly visible in the data.** Measuring Kendall's τ
between departure order (source sensor) and arrival order (destination
sensor) for the current matcher's own matched pairs: τ = 0.82–1.00 across
every date and direction, pooled inversion rate 0.9% (239 of 26,447
pairwise comparisons). One caveat: this measures order *within* matches
the timing-based matcher already chose, so it isn't fully independent
confirmation — but the failure-mode analysis below shows it isn't pure
circularity either.

Two matchers were built on top of the existing scoring function: a **hard**
monotonic-alignment DP (Needleman-Wunsch style, free skips, zero crossings
by construction, run separately per direction and merged by
greedy highest-score-first conflict resolution — a subsequence of a
monotonic sequence stays monotonic, so this preserves no-crossing), and a
**soft** crossing-penalty term (rank-mismatch based, additive, plugged into
the existing bipartite solver). LODO results:

| variant | MAE(total) | held-out loss | share error | corr |
|---|---|---|---|---|
| baseline | 17.88 | 1045.6 | 0.253 | 0.535 |
| HARD monotonic-DP | 17.50 | **2141.5** | 0.296 | **0.627** |
| SOFT crossing-penalty | 18.25 | 1682.5 | 0.277 | 0.378 |

**Hard-DP drives inversions to ~0 but costs real matched volume**
(207/265 matched on 2026-05-09, 119 unmatched vs. baseline's 3) — the
count-based loss punishes the volume loss more than it rewards the order
gain. **Soft crossing-penalty adds essentially nothing**: `crossing_weight`
LODO-fits to 0.0 in 7 of 8 folds; confirmed not a landmine-date artifact
by re-evaluating with 2026-05-19 excluded from scoring — soft stays
*exactly* identical to baseline in all 7 remaining folds.

**Root cause, quantified:** near-simultaneous arrivals defeat strict
per-traveler ordering. The gap to the next Sensor-A departure is under 2
seconds for 17.8% of consecutive travelers on 2026-05-09, and 26.0% on
2026-05-19. Concrete example: `A-80-23` and `A-80-25` share an identical
last-seen timestamp — no orderable information between them at all. A
group-level relaxation (cluster travelers within 1–5s, enforce order only
*between* groups, free matching *within* a matched group-pair) was tested
as the natural fix and **made this worse, not better**: best window (5s)
gives held-out loss 2269.6, matching fewer total pairs (460) than
individual hard-DP (517), because forcing all members of a fixed-window
cluster to align with a single opposing cluster strands members whose true
partner sits in a *different* neighboring group — see follow-up
investigation 3 for the full group-level results.

**Verdict: real structural property, not production-ready as a
constraint.** The one genuinely useful piece — a soft preference for
order-consistent matches — is already in production via `order_weight`
(adopted earlier in this doc). Going further (hard gate, explicit rank
penalty, or group-level relaxation) doesn't add anything beyond what's
already there, and actively hurts the count-based loss. **Not recommended
for production beyond the existing `order_weight` term.**

---

## Follow-up investigation 3: four alternative directional-inference approaches (2026-08-12)

*Motivated by the crowded-date bottleneck identified in both follow-ups
above: many equally-plausible candidates per traveler on busy dates. Tests
four structurally different ideas, each against the same LODO framework.*

| method | MAE(total) | held-out loss | share err | corr | crowded-date mean loss* | quiet-date mean loss* |
|---|---|---|---|---|---|---|
| baseline (production) | 17.88 | **1045.6** | **0.253** | 0.535 | **2803.5** | **459.7** |
| Log-normal transit likelihood | 17.88 | 1228.9 | 0.263 | 0.585 | 3534.5 | 460.3 |
| RSSI full trajectory shape (best variant) | 17.63 | 1057.1 | 0.260 | 0.542 | 2840.0 | 462.8 |
| Group-level order (best: 5s window) | 19.63 | 2269.6 | 0.314 | 0.655 | 7655.5 | 474.3 |
| Aggregate flow deconvolution (best: 2s bin) | 34.67 | 2921.6 | 0.280 | 0.612 | 9693.7 | 664.2 |

*crowded = {2026-05-09, 2026-05-19}, quiet = other 6 dates

**1. Aggregate flow without identity matching** — build binned
departure/arrival count series, estimate directional counts via 1-D
deconvolution against a learned transit kernel, entirely without solving
per-traveler identity. **Clearly worse**: MAE(total)=34.7 (~2x baseline),
systematically over-predicts volume (2026-05-09: 365 vs. manual 237)
because without identity resolution the regression conflates "A and B are
both busy at once" with "these are the same people transiting" — a
confound that *worsens* with density. Cannot outperform person-level
matching here.

**2. Group-level order preservation** — clusters travelers into local
temporal groups (1–5s windows), enforces order between groups only, free
matching within a matched group-pair. **Worse than individual hard-DP**
(loss 2269.6 vs. 2141.5, fewer total matches: 460 vs. 517). Fixed-window
clustering doesn't reliably correspond to real physical groups, so it
strands travelers whose true partner is in a neighboring group rather than
their own cluster's match. Does not fix the near-tie volume problem from
follow-up investigation 2.

**3. Empirical transit-time distribution** — a genuinely unambiguous
subset barely exists in this data (median top1-vs-top2 candidate score gap
is only 0.018 on a ~1.25 scale, even on the quietest date); using the
least-ambiguous available quartile (324 pairs) as a low-circularity
estimation sample: **transit times are log-normal, not Gaussian** (5-fold
CV logL −278.0 vs. −299.9), and **A→B and B→A transit-time distributions
are statistically indistinguishable** (Mann-Whitney p=0.474) — transit
time itself isn't a direction cue. Substituting the learned likelihood for
production's Gaussian (fit per-fold from training dates only, no leakage):
held-out loss 1228.9, slightly worse than baseline despite the better
distributional fit — candidate density, not kernel shape, is the
bottleneck.

**4. Full RSSI trajectory shape** — resampled 1,314 traveler-visit RSSI
curves to a fixed 20-point grid, tested Pearson correlation and cosine
similarity (direct and time-reversed orientations) as candidate-pair shape
scores; a DTW spot-check confirmed directional agreement with correlation
(ρ=−0.59) without diverging enough to justify a full DTW LODO sweep. All
four variants land within noise of baseline (loss 1057–1148). **Confirms,
does not overturn, follow-up investigation 1's finding** — the full curve
contains no more usable directional information than the earlier summary
features did. No supervised classifier was trained: the aggregate-only
ground truth (8 dates, no per-pair labels) makes a standard classifier
infeasible, and the calibrated-weight LODO approach used throughout is the
correct available analog.

**On crowded dates specifically, baseline wins outright** — every
structural/aggregate alternative is 2–4x worse there; only the two
RSSI-based methods stay within noise of baseline. None of the four
approaches reduces the candidate-density bottleneck; several make it
substantially worse.

### Cross-investigation conclusion

Three independent investigations (RSSI features x2, structural order x2,
transit-distribution shape, aggregate flow) now converge on the same
diagnosis: **the bottleneck is candidate density on 2–3 crowded dates, and
no feature-, distribution-, or structure-level refinement tested has moved
it.** Two things stand out as the actual next steps, in priority order:

1. **Fix the loss function.** Every alternative tested here trades matched
   volume for order/direction consistency and is punished for it by
   squared-error-on-raw-counts (the same 3x volume-weighting flaw already
   diagnosed and partially fixed earlier in this doc). The correlation
   metric improves in almost every alternative tested (0.54–0.66 vs.
   baseline's 0.535) even as held-out loss gets worse — a loss that
   credits reduced ambiguity separately from volume might reveal real
   value the current metric is hiding.
2. **Get better ground truth.** All three investigations converge on "the
   aggregate 8-date GT can't discriminate these methods" as the limiting
   factor, not any specific feature or algorithm. A small controlled
   validation (even a handful of known-MAC walk-throughs) would let
   genuine progress be told apart from noise, which nothing tested so far
   reliably can.

**No method from any of these three follow-up investigations is
recommended for production integration as tested.** The only validated
win remains the `order_weight` boundary-order term already adopted
earlier in this doc.

---

## Follow-up investigation 4: does direction reduce to a trivial `last_seen` rule? (2026-08-12)

*Tests the simplest possible baseline — `last_seen_A` vs. `last_seen_B`,
with `first_seen` as a tie-break — against production's full
`direction_candidate_score` (Gaussian transit score + RSSI shape +
boundary order), on the SAME matched pairs the production matcher already
produces. This isolates the direction-classification question from
identity matching entirely.*

**Rule:**

```
if last_A < last_B: A_to_B
elif last_B < last_A: B_to_A
else (tie): first_A < first_B -> A_to_B, first_B < first_A -> B_to_A, else ambiguous
```

### Headline: the rule is not merely competitive — it is byte-for-byte the same classifier

Across all 626 matched pairs on the 8 GT dates, the simple rule's direction
label matches production's `direction_candidate_score` output in **625/626
cases (99.8%)**, and the 8 dates' aggregate A→B/B→A counts are **identical
pair-for-pair**. Every downstream metric is therefore identical too, for
both a genuinely parameter-free direct evaluation (single global label
orientation across all 8 dates) and the LODO version:

| variant | MAE(total) | held-out loss | share error | corr |
|---|---|---|---|---|
| simple `last_seen`/`first_seen` rule — DIRECT | 17.88 | 1045.6 | 0.253 | 0.535 |
| simple `last_seen`/`first_seen` rule — LODO | 17.88 | 1045.6 | 0.253 | 0.535 |
| production baseline (full pipeline) | 17.88 | 1045.6 | 0.253 | 0.535 |

**Rule characterization** (on production's own matched pairs):
`last_seen` decides 99.8% of the time; `first_seen` tie-break is needed
exactly **once** (an exact-second tie, correctly resolved); zero
fully-ambiguous pairs. First-vs-last *ordering* agreement (not just
tie-break usage — does `sign(first_B−first_A)` match `sign(last_B−last_A)`
independently of which one "wins"?) is **99.8%**; the sole exception is
the same tie case, not a genuine conflict. The hypothesis that first/last
disagreement explains direction failures is **not supported** — that
population essentially doesn't exist here.

**Reception-zone overlap does not break the rule.** Overlap (`A`'s and
`B`'s `[first,last]` intervals intersect) occurs in 4.8% of matched pairs
(30/626), and in **100%** of those, `last_seen` alone still points the
same direction production's full scoring independently assigned.

**Confidence-magnitude thresholding on `|last_B − last_A|` has almost no
effect** (coverage 99.8%→98.4%, metrics flat, across thresholds
0/1/2/5/10s) — because accepted matches are already pre-filtered to large
gaps by the existing transit gate (5th percentile of `|last_B−last_A|` is
17s, median 86.5s; only 2/626 pairs are within 2s). The candidate-level
ambiguity this project has repeatedly diagnosed lives in the *rejected*
candidate pool, not in confirmed matches.

**Concrete failure examples, and why confidence doesn't help:** on
2026-05-27 (ground truth 100% A→B), the rule mislabels the same 9 pairs
production does — same pairs, because it's the same classifier. All 9 are
non-overlapping with large, *confident* `last_B−last_A` gaps (−16s to
−214s):

```
A-355-1  <-> B-477-7   Δlast=-214s
A-355-3  <-> B-477-9   Δlast=-209s
A-355-4  <-> B-477-1   Δlast=-174s
A-355-2  <-> B-477-8   Δlast=-162s
A-355-11 <-> B-477-4   Δlast= -68s
A-355-9  <-> B-477-16  Δlast= -35s
A-355-21 <-> B-477-10  Δlast= -28s
A-355-20 <-> B-477-13  Δlast= -19s
A-355-8  <-> B-477-14  Δlast= -16s
```

High confidence does not protect against this failure mode. Consistent
with every other finding across all four follow-up investigations, this
points at **wrong partner (identity matching), not ambiguous direction
labeling**, as the source of residual error.

### Comparison against every method tested so far

| method | held-out loss | share err | corr |
|---|---|---|---|
| **simple `last_seen`/`first_seen` rule** | **1045.6** | **0.253** | **0.535** |
| **production baseline (full pipeline)** | **1045.6** | **0.253** | **0.535** |
| RSSI peak-order (follow-up 1) | 1056.9 | 0.252 | 0.537 |
| RSSI full trajectory shape (follow-up 3) | 1057.1 | 0.260 | 0.542 |
| Log-normal transit likelihood (follow-up 3) | 1228.9 | 0.263 | 0.585 |
| Soft crossing-penalty (follow-up 2) | 1682.5 | 0.277 | 0.378 |
| Individual hard-DP order (follow-up 2) | 2141.5 | 0.296 | 0.627 |
| Group-level order (follow-up 3) | 2269.6 | 0.314 | 0.655 |
| Aggregate flow deconvolution (follow-up 3) | 2921.6 | 0.280 | 0.612 |

The trivial rule ties the full pipeline exactly and beats every
alternative tried across all four follow-up investigations on the primary
loss metric.

### Verdict: the direction half of the scoring function can be simplified; the matching half should not be

`rssi_weight` calibrates to 0.0 (follow-up 1) and `order_weight`'s
`boundary_order_score` never flips the argmax away from what `last_seen`
alone already says, on this data. The complexity in
`direction_candidate_score` earns its keep entirely in *which pairs get
matched* (the bipartite assignment's candidate selection and scoring), not
in the *direction label* assigned to whichever pairs end up matched — for
that, it reduces exactly to comparing two timestamps. **Recommendation:**
Stage D's direction-labeling step could be simplified to the trivial
`last_seen`/`first_seen` rule without losing any measured performance;
this is a legitimate, evidence-backed simplification worth adopting for
clarity and maintainability. The matching/candidate-selection machinery
(`transit_time_score`, `min_match_score`, `order_weight`'s role in
selecting *which* pair, the bipartite solver) should **not** be
simplified on this evidence — that is where the real, unresolved
candidate-density problem (diagnosed across all four follow-ups) actually
lives.

---

## Follow-up investigation 5: improving cross-sensor identity matching (2026-08-12)

*Follow-up 4 showed direction labeling is not the bottleneck — the simple
`last_seen_A` vs. `last_seen_B` rule is byte-for-byte equivalent to
production's full direction logic. This investigation targets the actual
remaining lever: **which A traveler gets paired with which B traveler**,
keeping the simple direction rule fixed throughout.*

### Headline: a real, reproducible improvement — the first one across five follow-up investigations

**Neighborhood-consistency re-scoring** — a bonus added to each candidate
pair's score for how well a traveler's ~2 nearest temporal neighbors'
*own* best candidates agree in relative order — beats production on every
primary metric, at **zero matched-volume cost**, with the winning
parameter (`λ=2.0`, 1 neighbor each side) chosen identically in **all 8
LODO folds** — unlike every previous new term tested in this project,
which only ever picked up a non-zero weight by overfitting to one
unstable date (2026-05-19).

| | held-out loss | share err | corr | matched |
|---|---|---|---|---|
| baseline | 1045.6 | 0.253 | 0.535 | 626 |
| **neighborhood-consistency (λ=2, 1 neighbor)** | **997.4 (−4.6%)** | **0.240** | **0.564** | 626 |

Robustness check: with `n_neighbors` fixed at 1, the in-sample loss curve
across `λ ∈ {0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 16, 32}` dips at `λ=2`
(7979) and plateaus above it (~8059) — a genuine interior optimum, not a
truncated grid edge. Letting `n_neighbors` float too (up to 4) chases a
less stable, less consistent fit across folds; the validated, robust
configuration is `λ=2.0` with exactly 1 neighbor on each side.

### Part 1: candidate density and ambiguity, quantified

90–94% of A travelers on the busiest dates (2026-05-09, 2026-05-19) have
a "near-tie" (2nd-best candidate scoring ≥90% of the best). **Not purely a
density effect**: even the quietest date (2026-05-31, mean 4.1
candidates/traveler) shows a 77.4% near-tie rate; corr(mean candidates,
near-tie fraction) = 0.79 across dates — density matters, but the
scoring function's resolving power is poor everywhere, not only when
crowded.

### Critical diagnostic: the 9 wrong pairs on 2026-05-27, traced individually

For each of the 9 A travelers production mislabels B→A (ground truth:
100% A→B):

- **Only 2/9 chosen partners were mutual-best** — 7/9 pair a traveler
  with someone who didn't prefer them back. Direct evidence the global
  Hungarian objective creates cascading wrong matches: it sacrifices
  locally-better pairings for global score.
- **5/9 have an alternative candidate in the same gate that would flip
  the pair to correct A→B** — the fix exists in the pool but the solver
  doesn't reach it, because that better B candidate got claimed by a
  competing A traveler with an even higher score.
- **4/9 (`A-355-1/2/3/4`) have no fixing alternative in their top 5** —
  clustered in time, all competing for the same 2–3 popular B candidates
  (`B-477-8`, `B-477-1`, `B-477-17`). Possibly a Stage C recall gap (true
  partner never resolved as a separate `Traveler_ID`) rather than a
  matching-algorithm fix at all.

This split (5 fixable-by-better-matching vs. 4 possibly unfixable) is the
clearest evidence yet that **identity matching, not direction labeling,
is where the recoverable error lives.**

### Part 2: confidence tiers do NOT reliably predict correctness

Bucketing matches by `best − 2nd-best` margin (tertiles): directional
share error is **0.275 (high-confidence only) → 0.257 (high+medium) →
0.253 (all)** — accuracy gets *worse*, not better, as low-confidence
matches are excluded. Rejecting ambiguous matches does not improve share
accuracy enough to justify the volume loss — **it doesn't improve it at
all.** Only correlation improves with stricter tiers (0.511→0.604), the
same pattern seen throughout this project.

### Parts 3–9: every other method tested, against baseline (LODO)

| method | held-out loss | share err | corr | matched | crowded loss | quiet loss |
|---|---|---|---|---|---|---|
| baseline | 1045.6 | 0.253 | 0.535 | 626 | 2803.5 | 459.7 |
| **neighborhood-consistency** | **997.4** | **0.240** | **0.564** | 626 | **2650.5** | 446.3 |
| hybrid (mutual-best + Hungarian on rest) | 1047.8 | 0.260 | 0.486 | 621 | 2811.5 | 459.8 |
| ambiguity-penalty only (crowdedness, no order info) | 1045.6 | 0.253 | 0.535 | 626 | 2803.5 | 459.7 |
| top-K candidate pruning | 1603.4 | 0.278 | 0.517 | 616 | 5019.5 | 464.7 |
| score-margin candidate pruning | 1687.9 | 0.288 | 0.544 | 596 | 5400.5 | 450.3 |
| anchor-window (local monotonic) | 2014.6 | 0.273 | 0.588 | 532 | 6687.0 | 457.2 |
| mutual-best only (strict) | 5332.0 | 0.314 | 0.470 | 173 | 19306.0 | 674.0 |

- **Mutual-best (strict)**: collapses (173/626 matched) — far too
  restrictive to use alone.
- **Hybrid** (lock mutual-best, Hungarian on the rest): ties baseline,
  doesn't help.
- **All candidate-pruning variants**: hurt, worst exactly on the crowded
  dates they were meant to help (loss 5000–6700 vs. baseline's 2803.5).
- **Anchor-window** (soft local monotonic windows around high-confidence
  anchors): still substantially worse — same starvation failure mode as
  the hard monotonic-DP from follow-up 2, just triggered by sparse
  anchors instead of a global constraint.
- **Ambiguity-penalty alone** (penalize crowded rows with no directional
  information): `beta=0` fits in every fold — inert. This isolates the
  mechanism: it is specifically *local order information*, not generic
  de-ambiguation, that neighborhood-consistency's win comes from.
- **Multi-hypothesis check** (Part 8): under trivial score perturbation
  (noise ≪ typical margins), the solved A→B/B→A split varies by ±2–4
  units on busy dates while total matched count stays perfectly fixed —
  direct, quantitative confirmation that the Hungarian optimum is only
  marginally better than nearby alternatives on this data.

### Verdict and recommendation

The Hungarian max-total-score objective (Part 9) is not inherently
wrong — a pure crowdedness/ambiguity penalty added on top of it is inert
— but it was missing a genuinely informative **local-order feature**.
Every hard-constraint approach to supplying that (mutual-best-only,
pruning, anchor windows) fails the same way earlier hard/soft
order-preservation attempts did in follow-up 2: they trade away matched
volume faster than they buy directional correctness, and hurt worst on
the crowded dates they target. The one method that isn't a hard
constraint — a small, bounded, per-pair **bonus** for local
neighbor-order agreement, solved with the exact same global assignment —
is the one that wins, cleanly and reproducibly across every fold.

**Recommendation: integrate neighborhood-consistency re-scoring
(`λ=2.0`, ±1 temporal neighbor) into `direction_candidate_score`, and
keep the simple `last_seen`/`first_seen` direction rule from follow-up 4.**
The term only changes *which* pairs get matched (626 matched either way),
so it composes cleanly with the direction-rule simplification already
validated — this is the first change across five follow-up investigations
with evidence to justify shipping it. Do not adopt mutual-best filtering,
candidate pruning, or anchor-windowing in any form tested here — all three
underperform baseline, most severely on exactly the crowded dates they
were meant to fix.

---

## Follow-up investigation 6: same-MAC cross-sensor overlap as an identity signal (2026-08-12)

*Tests whether the subset of matched pairs that share a raw MAC address
across sensors — despite MAC randomization preventing this for most
travelers — is a useful high-confidence signal, layered on top of
follow-up 5's neighborhood-consistency matcher (not a reversion to raw-MAC
matching, which the original design doc already ruled out as an
identity-resolution method on its own).*

### Headline: a second real win, stacking on top of neighborhood-consistency

| | held-out loss | share err | corr | matched | crowded loss |
|---|---|---|---|---|---|
| current best (neighborhood-consistency alone) | 997.4 | 0.240 | 0.564 | 626 | 2650.5 |
| **MAC-anchor hybrid** | **944.8 (−5.3%)** | **0.199** | **0.676** | 623 | **2484.5** |
| **MAC bonus, binary** | **942.5 (−5.5%)** | 0.204 | 0.603 | 631 | 2414.0 |
| MAC bonus, jaccard | 947.8 | 0.214 | 0.621 | 631 | 2492.0 |
| MAC bonus, prop_a / prop_b | 961.5 / 951.5 | 0.210 / 0.213 | 0.605 / 0.649 | 631 | 2505.0 / 2520.0 |

Every MAC-based variant improves on all three primary metrics
simultaneously, at essentially zero matched-volume cost.

### Prevalence and structure

**Only 4.0% of matched pairs (25/626) share ≥1 MAC** — always exactly one
shared MAC, never more. Rarer on crowded dates (1.9%) than quiet ones
(7.2%) — consistent with longer/busier transits giving MAC randomization
more opportunity to fire. **96% coverage loss if only same-MAC pairs were
trusted** — far too sparse to use as a primary matcher.

Same-MAC pairs are structurally cleaner on metrics the current scoring
function doesn't touch: **mutual-best rate 20.0% vs. 8.3%** (2.4x), **mean
transit gap 13.7s vs. 39.3s**, fewer competing candidates (17.7 vs. 23.1).
Existing `match_score`/margin show *no* difference (0.072 vs 0.069) —
confirming this is genuinely new information, not something already
captured indirectly by timing or RSSI.

**Directional alignment with manual GT**: same-MAC subset alone gives
share error 0.180 and corr 0.848 (n=25, thin but striking) vs.
different-MAC subset's 0.257 / 0.473 (n=601).

### Critical diagnostic: 2026-05-27, closing the loop from follow-up 5

Under the current-best matcher, **all 8 remaining wrong-direction pairs
are in the different-MAC subset — 0 of 8 share a MAC.** Tracing
alternatives: **3 of 8 have a same-gate alternative that both shares a MAC
and gives the correct direction** (`A-355-20`, `A-355-21`, `A-355-8`; for
`A-355-8` the MAC-sharing alternative scored almost identically to the
chosen one, 1.042 vs 1.050). The other 5 have no MAC-sharing alternative
at all — consistent with follow-up 5's diagnosis that they may be a
genuine Stage C recall gap, not fixable by any matching change. **Both MAC
methods fix exactly the predicted 3/8**: `{A_to_B: 13, B_to_A: 8}` →
`{A_to_B: 16, B_to_A: 5}` (true count: 17/0).

### Anchor vs. soft bonus, and overlap strength

**MAC-anchor hybrid** (lock only *mutually unambiguous* same-MAC pairs —
if the same MAC appears in multiple viable candidates on either side,
lock nothing — then run neighborhood-consistency on the rest): best on
share error, correlation, and crowded-date loss, at almost no volume cost
(623 vs. 626).

**Soft MAC bonus**, robustness-checked over bonus ∈ {0…32}: in-sample loss
dips at **bonus≈3** and plateaus for bonus≥4 — genuine saturation, not a
truncated grid (same "saturation not truncation" shape as
`RSSI_TREND_SCALE`'s original calibration). LODO-fitted bonus was 4.0 in
6/8 folds, 2.0 in the other 2 — consistent, not a single-date artifact
like most earlier failed terms in this project.

**Binary equality beat every graded-overlap variant** (Jaccard, proportion
of A-side/B-side MACs shared) on held-out loss. Since no pair in this
dataset ever shares more than one MAC, graded "strength" is degenerate by
construction here — there's no data to test whether *more* shared MACs
would be stronger evidence.

### Leakage check: none found

Verified directly in Stage C's clustering code
(`notebooks/03_stage_c_entity_resolution.ipynb`): `cluster_block()` merges
same-sensor sessions using only `start_time`/`end_time` overlap-or-gap,
`rssi_trend`, and `mean_rssi` — **the raw MAC value is never a clustering
feature**, only the initial grouping key for raw readings before
clustering. Stage C is also per-sensor only (per this doc's own Context
section) — it never compares MACs *across* sensors. Stage D's existing
scoring (timing, RSSI shape, boundary order, neighborhood-consistency)
also never touches raw MAC identity. Cross-sensor same-MAC overlap is
independent evidence, not double-counted anywhere in the pipeline.

### Recommendation

**Integrate same-MAC information into production as both an anchor and a
scoring bonus** — the evidence supports more than validation-only use.
Concretely: lock mutually-unambiguous same-MAC pairs as anchors before the
main solve, and add a bounded `mac_bonus≈3` (binary equality, no need for
graded overlap) to any remaining same-MAC candidate pair, layered on top
of follow-up 5's neighborhood-consistency term. Do not raise same-MAC's
coverage or use it as a standalone matcher — 96% of matches will never
share a MAC, so it must stay supplementary to the existing timing/order
machinery.

### Per-date predictions vs. manual ground truth (MAC-anchor hybrid, LODO held-out)

| date | GT window | pred A→B | pred B→A | pred total | manual A→B | manual B→A | manual total | total err | pred A-share | manual A-share | held-out loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-04-11 | 09:03:54–09:36:57 | 34 | 3 | 37 | 27 | 1 | 28 | +9 | 0.919 | 0.964 | 53 |
| 2026-05-02 | 12:36:26–13:15:03 | 36 | 39 | 75 | 20 | 36 | 56 | +19 | 0.480 | 0.357 | 265 |
| 2026-05-09 | 11:00:35–11:58:39 | 189 | 76 | 265 | 181 | 56 | 237 | +28 | 0.713 | 0.764 | 464 |
| 2026-05-16 | 08:20:51–09:04:17 | 32 | 11 | 43 | 22 | 2 | 24 | +19 | 0.744 | 0.917 | 181 |
| 2026-05-19 | 09:27:20–10:43:42 | 77 | 34 | 111 | 136 | 2 | 138 | −27 | 0.694 | 0.986 | 4505 |
| 2026-05-23 | 09:04:50–09:57:14 | 18 | 29 | 47 | 30 | 13 | 43 | +4 | 0.383 | 0.698 | 400 |
| 2026-05-27 | 10:29:41–10:51:16 | 16 | 5 | 21 | 17 | 0 | 17 | +4 | 0.762 | 1.000 | 26 |
| 2026-05-31 | 10:09:30–10:51:04 | 15 | 9 | 24 | 55 | 1 | 56 | −32 | 0.625 | 0.982 | 1664 |

**MAE(total) = 17.75, mean held-out loss = 944.8, mean \|share error\| =
0.199, corr = 0.676.** Every row is genuinely held-out (that date's own
manual counts were never used to fit the parameters or label orientation
applied to it). 2026-05-19 and 2026-05-31 remain the two structurally
unreachable dates flagged earlier in this doc (Stage C traveler counts
below the manual total) — their large held-out losses are a Stage A–C
recall ceiling, not a Stage D matching failure, and dominate the mean loss
the same way they always have across every follow-up in this doc. The
other six dates are within 4–28 people of the true total and the
directional share is now visibly closer to manual truth than the
production-baseline table at the top of this document (e.g. 2026-05-27:
predicted A-share 0.762 vs. the original baseline's ~0.52, against a true
1.000).

---

## Follow-up investigation 7: Stage C recall / identity-resolution quality (2026-08-13)

*Every follow-up so far improved Stage D's matching or direction logic.
This investigation asks whether the two structurally-unreachable GT dates
(2026-05-19, 2026-05-31 — `structural_ceiling = min(n_A, n_B)` below the
manual total no matter how Stage D scores candidates) are actually a
Stage C problem, one stage upstream.*

### Headline: the two dates have different root causes — only one is fixable here

| date | A raw MACs | B raw MACs | structural ceiling | manual total | ceiling − manual |
|---|---|---|---|---|---|
| **2026-05-19** | 306 | 357 | 124 | 138 | **−14** |
| **2026-05-31** | 47 | **40** | 25 | 56 | **−31** |

**2026-05-31 is not a Stage C problem at all.** Sensor B's raw MAC count
(40) is already below the manual total (56) *before any clustering runs*
— no un-merging can manufacture travelers that were never detected. This
is a Stage A/B raw-sensing ceiling (or a fundamental BLE-counting limit:
not everyone carries an advertising device), confirmed unfixable across
every Stage C variant tested below (ceiling stays 20–34 regardless of
threshold).

**2026-05-19 shows genuine over-merging.** Cluster-size distribution is
anomalous only on this date: mean MACs/traveler 2.36 vs. 1.36–2.06 on
every other GT date, p99 11.9 vs. 4.0–9.0 elsewhere, clusters running up
to 681–805s. Concrete example: Sensor-A `Traveler_ID` `A-198-47` merges
**17 different MAC addresses**, up to **8 advertising concurrently**
within a 2-minute span, all sharing a similar `rssi_trend` (≈ −0.02 to
−0.03 dB/s) — a crowd whose members merely decayed at similar rates near
the sensor together, which Stage C's trend-difference clustering can't
distinguish from one person's MAC rotating.

**The cross-sensor same-MAC diagnostic does NOT cleanly implicate
2026-05-19 specifically** — worth stating plainly since it's an intuitive
check that turned out not to discriminate. Testing whether a Sensor-A
cluster's constituent MACs map to multiple distinct Sensor-B travelers:
100% of large (≥5 MAC) A-clusters hit ≥2 different B travelers on
2026-05-19 — but the same is true on 2026-05-09 (100%) and 2026-05-27
(100%), and even 2-MAC clusters hit this 71–78% of the time on *every*
date tested. This diagnostic flags nearly everything everywhere; it isn't
noise-free enough to serve as an over-merge detector on its own. Cluster
*size* remains the only signal that actually discriminates 2026-05-19.

### Threshold sensitivity: two Stage C levers, two very different risk profiles

Re-implemented Stage C's clustering faithfully (verified against
production output: identical on Sensor A, 3/19,359 sessions differ on
Sensor B from a floating-point tie at the exact cutoff boundary —
negligible) and swept both free parameters, propagated through the
current-best Stage D matcher, LODO over the 8 GT dates:

| variant | held-out loss | share err | corr | ceiling₀₅₋₁₉ (Δ manual) |
|---|---|---|---|---|
| default (gap=60s, trend-percentile=10) | 901.0 | 0.211 | 0.632 | 124 (−14) |
| percentile=5 (stricter RSSI-trend cutoff) | 1628.5 | 0.178 | 0.695 | 156 (**+18**) |
| percentile=2 | 2377.1 | 0.168 | 0.811 | 196 (**+58**) |
| **gap=30s (temporal gap only)** | **881.6 (best)** | **0.202** | **0.676** | 133 (−5) |
| gap=20s | 994.5 | 0.199 | 0.662 | 138 (**0**, exactly closed) |

`THRESHOLD_PERCENTILE` (the RSSI-trend-difference cutoff) is the
aggressive lever: tightening it fully closes 2026-05-19's ceiling but
**more than doubles overall held-out loss** — it over-splits everywhere,
not just the genuinely over-merged crowd clusters, creating duplicate
travelers broadly. `BLOCK_GAP_THRESHOLD_S` (max temporal gap, currently
60s) is the surgical lever: tightening to 30s is the *only* variant in the
whole sweep that improves overall held-out loss versus the current
default, while partially narrowing the ceiling gap. It's a redistribution,
not a clean win: per-date, default→gap=30s, 2026-05-19 improves sharply
(loss 3986→3328) but **2026-05-09 — the other crowded date — gets
noticeably worse (554→1130)**. Net crowded-date mean loss still improves
(2270→2229) but this is not "problem solved."

Splitting the worst offender directly: tracing `A-198-47` through
gap=30s, it only splits into 2 sub-clusters, not the 8+ concurrent people
it actually contains — a single global gap threshold is too blunt to
resolve genuinely concurrent, overlapping sessions. Fixing that properly
would need a different clustering feature entirely, not a stricter
version of the same rule.

**Under-merging risk, checked honestly:** 1,115 of 7,961 Sensor-A
travelers (14%) that were one `Traveler_ID` at gap=60s split into
multiple at gap=30s. One inspected 2-session split, at exactly a 56s gap,
shows a 13 dB mean-RSSI jump and a trend-sign flip between the two halves
— plausibly two different people, but genuinely ambiguous without
MAC-level ground truth. Some fraction of these splits are probably wrong;
this cost can't be precisely quantified from data available here.

### Full 30-day validation (not just the 8 GT dates, per the design doc's own standing caution about Stage C changes)

Every threshold change is fit **globally per sensor across all 30
collection days**, not per-date, so any recommendation has to be checked
against the 22 days with no ground truth, not just the two target dates.

**Per-day traveler-count impact of gap=30s, all 30 days, both sensors (44
sensor-days):** mean **+5.2%**, std 2.6%, range 0–11.5%, **no day exceeds
15%**. The increase is a smooth, proportionate function of how busy the
date is (busiest dates like 2026-05-19/05-09 see the largest relative
increases, 8.8%/11.5% — consistent with them having the most over-merged
crowd clusters to split), not an isolated blow-up anywhere. Cluster-size
statistics (mean/p99 MACs per traveler) shrink modestly and consistently
across every one of the 30 days, GT or not — no sign of localized
instability.

**Full Stage D output, all 14 dual-sensor dates** (current-best matcher:
neighborhood-consistency + same-MAC bonus + simple direction rule),
default vs. gap=30s:

| | total matched | A→B share range across dates |
|---|---|---|
| gap=60s (current production) | 6,221 | 0.449 – 0.800 |
| gap=30s | 6,645 (+6.8%) | 0.435 – 0.806 |

The six dual-sensor dates with **no ground truth at all**
(2026-05-03/17/18/20/24/28) move by +6–10% in matched volume, each with a
direction share within a few points of its gap=60s value — no date
swings to an implausible extreme, and the shift is proportionate to the
rest of the dataset. This is not proof of *improved accuracy* on those
six dates (no ground truth exists to check that), but it is evidence the
change behaves consistently rather than pathologically outside the GT
sample.

### Verdict

1. **Is Stage C over-merging the main reason both dates are unreachable?**
   No — only for 2026-05-19. 2026-05-31 is a raw-detection ceiling,
   unfixable at Stage C.
2. **Which rule causes the most damage?** The RSSI-trend-similarity
   criterion during genuinely crowded, temporally-overlapping periods —
   it can't distinguish "one MAC rotating" from "several people with
   similar decay trends passing together."
3. **Can the ceiling be removed without overcounting?** Partially.
   `gap_threshold_s: 60→30s` gets a real, full-dataset-validated
   improvement without pathological overcounting; fully closing the
   ceiling requires either the riskier percentile lever or accepting a
   modest overall cost (gap=20s).
4. **Best held-out Stage C variant?** `gap_threshold_s=30,
   threshold_percentile=10` (percentile unchanged) — held-out loss
   901.0→881.6, share err 0.211→0.202, corr 0.632→0.676, validated as
   proportionate and non-pathological across all 30 collection days.

**Recommendation: adopt `BLOCK_GAP_THRESHOLD_S: 60→30s` in Stage C.**
Validated on both axes the design doc's own standing caution asked for —
held-out accuracy on the 8 GT dates, and non-pathological behavior on the
22 days with no ground truth. It is a redistribution more than a clean
win (2026-05-09 gets worse while 2026-05-19 improves), and it does not
fully close either structurally-unreachable date, so it should be framed
to stakeholders as a modest, evidence-backed refinement, not a fix for
2026-05-19 or 2026-05-31. Do not change `THRESHOLD_PERCENTILE` — every
tightening tested there traded away more overall accuracy than it
recovered. 2026-05-31 remains genuinely out of reach for any Stage
A–D change.
