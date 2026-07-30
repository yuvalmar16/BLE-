# Stage C: Entity Resolution / Clustering — Design

## Update (2026-07-29, after Stage A/B rebuild): rerun on corrected data

Stage A's dedup was corrected after the user obtained the full
second-resolution Sensor A log (see Stage A's design doc), which changed
Stage B's session counts for Sensor A (14,661 -> 15,093 sessions). This
stage was rerun on the corrected Stage B output; the algorithm itself
(blocking, complete-linkage clustering) is unchanged. Sensor A: 7,604 ->
**7,838** travelers. Sensor B is unaffected (its own data didn't change)
and reproduces identically: 10,516 travelers. The largest-cluster sizes
stayed in the same range as before (17 for A, 22 for B), confirming the
chaining fix still holds on the corrected data. Numbers below reflect the
rerun.

## Context

This is Stage C of the 4-stage hiker-monitoring ML pipeline described in
`hiker_monitoring_project_summary.md`. It merges multiple MAC addresses
that likely belong to the same physical hiker (multi-device carrying +
mid-visit MAC rotation) into a single `Traveler_ID`, fixing the
overcounting problem described in the project background. This is the
stage that turns Stage B's device-level session counts into hiker-level
traveler counts.

## Scope

Runs **independently per sensor** — Sensor A's sessions are clustered
among themselves, and Sensor B's sessions are clustered among themselves,
each producing its own `Traveler_ID` space. Cross-sensor matching (linking
a traveler's Sensor A presence to their Sensor B presence) is Stage D's
job, not this stage's.

In scope: `DATA/sessions_sensor_A.parquet`, `DATA/sessions_sensor_B.parquet`
(Stage B output).

Out of scope: Stage D (directionality via ΔRSSI) — planned separately once
Stage C's traveler tables are available.

## Key finding driving the design

Sessions that literally overlap in time chain into blocks of up to ~500
concurrent sessions during busy periods (many devices/hikers present at
once). Temporal overlap alone therefore cannot decide "same traveler" —
RSSI-trend similarity must be an independent, second condition, not folded
into overlap.

Combining "time overlap" (seconds) and "RSSI-trend difference" (dBm/s)
into one blended DBSCAN distance would require an arbitrary weighting
between two unrelated units, with no ground truth available to validate
that weight. Instead, this stage uses **two independent threshold
conditions**.

**Update after first implementation attempt:** the initial design took
connected components of the match graph (equivalent to DBSCAN with
`min_samples=1`, i.e. single-linkage). This chains transitively — testing
on real data produced a 419-session "traveler" whose member sessions'
`rssi_trend` ranged from -2.0 to +6.0 dBm/s, despite every individual
pairwise edge being under the 0.02 dBm/s cutoff (session 1 resembled
session 2, which resembled session 3, etc., but session 1 and session 419
were nothing alike). A real same-traveler group should be *mutually*
similar, not just chained through neighbors. Fixed by switching to
**complete-linkage hierarchical clustering** per block (see step 5 below),
which requires every pair within a merged group — not just a path — to be
under threshold.

## Pipeline steps (per sensor)

1. **Blocking** (a computational optimization, not a decision step): sort
   sessions by `start_time`. Chain sessions into blocks using the same
   running-max-end-time + gap logic as Stage B's own sessionization, with a
   **60-second** gap allowance (same threshold as Stage B — chosen for
   consistency, and because it also covers a device rotating its MAC
   mid-visit: the old MAC's session ends right as the new MAC's session
   starts a few seconds later). This bounds pairwise comparison to
   within-block sessions only (blocks observed up to ~1,300 sessions at
   this threshold on Sensor A, not the full 14–19K).

2. **Candidate pairs within each block**: for every session pair `(i, j)`
   in a block, compute:
   - `time_ok`: `[start_i, end_i]` and `[start_j, end_j]` overlap, OR the
     gap between them is ≤ 60s.
   - `trend_diff = |rssi_trend_i - rssi_trend_j|` (dBm/s). If either
     session has `n_readings == 1` (trend is `NaN`, undefined for a single
     point), fall back to `|mean_rssi_i - mean_rssi_j|` instead.

3. **Threshold selection**: plot the distribution of `trend_diff` (and the
   `mean_rssi` fallback) across all candidate pairs and pick a cutoff at
   the natural break in that distribution — the same empirical,
   elbow-style approach used to justify the 60s gap threshold in Stages A
   and B. There's no ground truth to tune against, so this is documented
   as an inspected, chosen value rather than a formula.

4. **Normalize distances**: each pair's `diff_value` is divided by its own
   metric's threshold (`trend` or `fallback`), giving a unitless
   `normalized_distance` where `<= 1.0` means "under threshold" regardless
   of which metric produced it. Pairs that fail `time_ok` get a large
   sentinel distance so they can never merge.

5. **Traveler_ID = complete-linkage cluster**, computed per block: build
   the full pairwise `normalized_distance` matrix for the block's
   sessions, run hierarchical clustering with `method="complete"`, and cut
   at `distance = 1.0`. Complete linkage requires every pair within a
   resulting cluster to be under threshold, not just a chained path,
   eliminating the chaining failure mode described above. A block of size
   1 trivially becomes its own singleton cluster. `Traveler_ID` combines
   `sensor + block_id + cluster_label` for global uniqueness per sensor.

6. **Traveler summary table**: one row per `(sensor, Traveler_ID)`:
   `n_devices` (distinct MACs merged), `MACs` (list), `start_time` /
   `end_time` (min/max across merged sessions), `total_duration_s`. This
   is the output that actually addresses the project's overcounting
   problem — traveler counts instead of MAC/device counts.

## Output

- `DATA/sessions_sensor_A_with_traveler.csv` / `.parquet`
- `DATA/sessions_sensor_B_with_traveler.csv` / `.parquet`
  (Stage B's session tables with an added `Traveler_ID` column)
- `DATA/travelers_sensor_A.csv` / `.parquet`
- `DATA/travelers_sensor_B.csv` / `.parquet`
  (one row per traveler, per sensor)
- `DATA/travelers_per_day.csv` / `.parquet`
  (one row per `(date, sensor)`, with `n_travelers` — a traveler's day is
  the calendar date of their `start_time`; verified that no session in
  either sensor's data crosses midnight, so this assignment is unambiguous)

## Validation / sanity checks (in-notebook)

No ground truth exists, so validation here is inspection-based rather than
a metric:
- Distribution of `n_devices` per traveler (expect mostly 1, a meaningful
  minority >1).
- Spot-check a few multi-device travelers: print their merged sessions
  side by side (MACs, time windows, trends) to sanity-check the merge is
  physically plausible.
- Report: total sessions in vs. total travelers out, per sensor.
- Flag the largest few connected components explicitly — an oversized
  component (e.g. merging a whole busy-period block into one "traveler")
  is a red flag that the trend-difference threshold is too loose, and
  should prompt tightening it.

**Daily coverage isn't identical between sensors:** of 29 distinct
collection days across both sensors (after Stage A's rebuild added
Sensor A data back to 2026-02-26), 6 days (2026-02-26, 02-27, 02-28,
03-02, 03-03, 05-25) have Sensor A data only, and 9 days have Sensor B
data only (2026-04-09, 04-10, 05-04, 05-05, 05-06, 05-21, 05-22, 05-29,
05-30) — reflecting real differences in when each sensor was actively
deployed/logging, not a pipeline bug. On the 14 days both sensors have
data, daily traveler counts track closely (e.g. 2026-05-18: 902 vs. 901;
2026-05-19: 809 vs. 834), which is a useful independent sanity check that
the two per-sensor pipelines behave consistently despite being computed
completely separately.

**Known limitation (accepted, not fixed):** even after the complete-linkage
fix, the largest clusters (up to ~17-22 devices) are more plausibly hiking
groups walking together than one person's devices — people on the same
trail, at the same pace, produce genuinely similar overlapping dwell time
and RSSI trend. Dwell-time overlap + trend similarity cannot distinguish
"one person, many devices" from "several people with similar devices,
walking together" from RSSI alone, and there's no ground truth to resolve
this. Decision: leave these clusters as-is, surfaced via the largest-
components report for manual/domain-expert review, rather than guessing
further (e.g. an arbitrary max-devices-per-traveler cap was considered and
rejected, since it would trade this error for a different, equally
unvalidated one).

## Deliverable

`notebooks/03_stage_c_entity_resolution.ipynb`, structured like Stages A
and B:
1. Load Stage B session tables (both sensors)
2. Block sessions per sensor (gap-based chaining)
3. Compute candidate pairs + trend/RSSI differences within blocks
4. Threshold selection (distribution plot + chosen cutoff, documented)
5. Cluster via complete linkage -> Traveler_ID
6. Build traveler summary table
7. Validation/sanity-check cells
8. Travelers per day, per sensor (table + bar chart), plus a coverage
   check for days seen by only one sensor
9. Write output CSV + Parquet (sessions-with-traveler, traveler summary,
   and travelers-per-day, all per sensor)

## Out of scope for this design

Stage D (directionality via ΔRSSI, and cross-sensor traveler linking) is
not designed here and will get its own spec once Stage C's traveler tables
are available to build against.
