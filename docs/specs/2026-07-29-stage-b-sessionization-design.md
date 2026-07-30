# Stage B: Sessionization & Feature Engineering — Design

## Update (2026-07-29, after Stage A rebuild): rerun on corrected Sensor A data

Stage A originally interpolated seconds into a minute-resolution Sensor A
export. The user has since obtained the full second-resolution log
(`Sensor_A_combined_file.xlsx`), and rebuilding Stage A on it revealed the
old dedup had undercounted (minute-resolution timestamps let distinct
readings collide) — see Stage A's design doc for the full story. Stage A
now outputs `sensor_A_deduped.parquet` with a plain `timestamp` column
(638,254 rows, vs. the old 347,148), no `timestamp_reconstructed`. This
stage is rebuilt against that corrected input; the 60s gap threshold was
re-verified against the new data (99.9% of gaps ≤60s, consistent with the
99.8% measured before) and still holds. Numbers below reflect the rerun.

## Context

This is Stage B of the 4-stage hiker-monitoring ML pipeline described in
`hiker_monitoring_project_summary.md`. It groups each MAC address's raw RSSI
readings — on each sensor independently — into continuous "visits"
(sessions), and extracts per-session features. These sessions/features feed
Stage C (entity resolution/clustering across MACs) and Stage D
(directionality via ΔRSSI between sensors).

## Scope

In scope:
- `DATA/sensor_A_deduped.parquet` (Stage A output — already deduped, plain
  `timestamp` column).
- `DATA/combined_file_sensor_B_part_1.xlsx` and `_part_2.xlsx` (raw Sensor
  B — deduped as a prep step inside this stage; no time interpolation
  needed since Sensor B already has second resolution).

Out of scope: Stage C (clustering/entity resolution), Stage D
(directionality) — planned separately once Stage B's output is available.

## Input data characteristics

- Sensor B has the same cumulative-re-export duplication issue found in
  Sensor A: 1,217,216 raw rows across 7 source files reduce to 815,093 after
  deduplicating on `(MAC, RSSI, timestamp)`. Sensor B's combined data is
  also already globally sorted by timestamp (confirmed, same as Sensor A).
- Per-MAC consecutive-reading gaps (measured on Stage A's corrected,
  deduped output): median 1s, 99.9% of gaps ≤60s, with a long tail of much
  larger gaps (different days, or a device leaving and returning later).

## Pipeline steps

1. **Load Sensor A**: `sensor_A_deduped.parquet`, use `timestamp` as the
   time field directly.
2. **Load & dedup Sensor B**: concatenate `_part_1.xlsx` + `_part_2.xlsx`,
   dedup on `(MAC, RSSI, timestamp)` (same method as Stage A, folded in here
   as a prep step rather than a separate notebook since no interpolation is
   needed). Use `timestamp` as the time field directly.
3. **Sessionize** (per sensor, independently): sort rows by `(MAC, time)`.
   Within each MAC's readings, start a new session whenever the gap since
   the previous reading exceeds **60 seconds** (chosen because 99.8% of
   observed same-MAC gaps fall within this threshold — a natural break
   point between "still nearby" and "gone").
4. **Extract per-session features**:
   - `sensor` (`A` or `B`)
   - `MAC`
   - `session_id`
   - `start_time`, `end_time`
   - `duration_s` = `end_time - start_time`
   - `n_readings`
   - `mean_rssi`
   - `rssi_trend`: least-squares linear regression slope of RSSI against
     seconds-since-session-start (dBm/second). `NaN` when `n_readings < 2`
     (slope is undefined with a single point).
5. Sessions with `n_readings == 1` are **kept** (not dropped) — they're
   still potentially informative (e.g. a device briefly at the edge of
   range) and it's premature to discard them before Stage C's clustering
   is designed. `duration_s` is `0` and `rssi_trend` is `NaN` for these.

## Output

Two table pairs (CSV + Parquet, same convention as Stage A):
- `DATA/sessions_sensor_A.csv` / `.parquet`
- `DATA/sessions_sensor_B.csv` / `.parquet`

Columns: `sensor, MAC, session_id, start_time, end_time, duration_s,
n_readings, mean_rssi, rssi_trend`.

## Validation / sanity checks (in-notebook)

- Sum of `n_readings` across all sessions equals the deduped input row
  count, per sensor (no readings lost or double-counted).
- `duration_s` matches `end_time - start_time` for every session.
- Spot-check a MAC with multiple sessions (confirms gap-splitting works)
  and a MAC with one long session.
- Histograms of `duration_s` and `n_readings` per session, per sensor.
- Report totals: sessions per sensor, and how many have `n_readings == 1`
  (undefined trend) vs. multi-reading sessions.

## Deliverable

`notebooks/02_stage_b_sessionization.ipynb`, structured like Stage A's
notebook:
1. Load Sensor A (Stage A output) & Sensor B (raw, dedup inline)
2. Sessionize both sensors (gap-based splitting)
3. Extract per-session features (incl. RSSI trend via linear regression)
4. Validation/sanity-check cells
5. Write output CSV + Parquet for both sensors

## Out of scope for this design

Stage C (entity resolution/clustering) and Stage D (directionality) are not
designed here and will get their own spec once Stage B's session tables are
available to build against.
