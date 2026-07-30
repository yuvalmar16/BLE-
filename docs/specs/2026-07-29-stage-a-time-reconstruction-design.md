# Stage A: Sensor A Data Preparation — Design

## Update (2026-07-29, after initial implementation): superseded by full data

The original version of this stage reconstructed approximate per-reading
seconds from `combined_file_sensor_A.csv`, which only logged minute
resolution. **The user has since obtained the full device log,
`DATA/Sensor_A_combined_file.xlsx`, which has true second-resolution
timestamps for every row.** This makes the interpolation step obsolete —
there's nothing to reconstruct anymore.

This also exposed a real problem with the original approach, not just an
improvement opportunity: **the original dedup undercounted.** With
minute-resolution timestamps, two genuinely different readings from the
same MAC with the same RSSI that happened to land in the same *minute*
were indistinguishable from true duplicates, so deduplicating on
`(MAC, RSSI, timestamp)` collapsed them together incorrectly. Rerunning
the identical dedup logic on the new file (with real seconds) gives
638,254 rows instead of 347,148 — a large fraction of what looked like
"duplicates" before were actually distinct readings sharing a minute.
Since this row count fed directly into Stage B and Stage C, **both of
those stages were rebuilt** after this fix (see their own design docs for
updated numbers).

Row-count comparison, per source file (old minute-resolution file vs. new
second-resolution file — 6 of 7 files are otherwise identical in content,
see below):

| source file | old rows | new rows |
|---|---|---|
| `dataA_110426_1_good.csv` | 3,837 | 34,125 |
| all other 6 files | (unchanged) | (unchanged) |

The new file's `dataA_110426_1_good.csv` also turned out to span **26 Feb
– 11 Apr 2026** (46 days), whereas the old extract only covered a 75-minute
window on 11 Apr. This adds six weeks of Sensor A history that predates
Sensor B's earliest data (9 Apr) — that whole stretch has no Sensor B
counterpart, which matters for Stage D later.

The sections below are rewritten to describe the current (corrected)
pipeline. The original interpolation design is no longer implemented;
it's not preserved in detail here since the new data makes it moot rather
than wrong-but-historically-interesting (unlike Stage C's clustering bug,
which is worth keeping as a cautionary example).

## Context

This is Stage A of the 4-stage hiker-monitoring ML pipeline described in
`hiker_monitoring_project_summary.md`. It prepares Sensor A's raw BLE scan
log for the stages that follow: deduplicating cumulative re-export
overlap, the same issue already known from Sensor B (handled inline in
Stage B).

## Scope

In scope: `DATA/Sensor_A_combined_file.xlsx` only.

Out of scope / excluded:
- `data_A_030126.xlsx` — a separate, earlier test capture (3 Jan 2026, one
  hour) with no matching Sensor B data for that day. Not needed for this
  stage.
- Sensor B processing, sessionization, clustering, directionality — later
  stages.

## Input data characteristics

- 945,642 raw rows across 7 source files, spanning **26 Feb – 31 May
  2026**. Timestamps already have second resolution for every row.
- Same duplicate-export issue found before: e.g. `data230526a.csv`
  overlaps `data_A_260526.csv`. Deduplicating on `(MAC, RSSI, timestamp)`
  globally across all 7 files reduces the dataset from 945,642 to
  **638,254** rows.
- Data is already globally sorted by timestamp (confirmed, same property
  as before and as Sensor B).

## Pipeline steps

1. **Load** `Sensor_A_combined_file.xlsx`.
2. **Global dedup** on `(MAC, RSSI, timestamp)` across the full
   concatenated dataset — removes redundant rows from cumulative
   re-exports, independent of which file is "extra."
3. No interpolation needed — `timestamp` is already second-resolution and
   used directly downstream.

## Output

Two files, same data:
- `DATA/sensor_A_deduped.csv` — for manual inspection / Excel.
- `DATA/sensor_A_deduped.parquet` — canonical typed input for Stage B.

Both contain all original columns (`MAC, RSSI, timestamp, name,
manufacturer, serviceUUID, source_file`). No `timestamp_reconstructed`
column — Stage B and downstream use `timestamp` directly for Sensor A now,
the same as they already did for Sensor B.

## Validation / sanity checks (in-notebook)

- Row counts before/after dedup.
- `timestamp` is non-decreasing within each source file.
- Per-source-file row counts pre/post dedup, reported explicitly so the
  overlap issue is visible and documented rather than silently fixed.

## Deliverable

`notebooks/01_stage_a_time_reconstruction.ipynb` (filename kept for
continuity even though it no longer reconstructs anything), structured as:
1. Load & inspect raw data (schema, date ranges, per-file counts)
2. Dedup + report on duplicates removed per file
3. Validation/sanity-check cells
4. Write output CSV + Parquet

## Out of scope for this design

Stages B–D (sessionization, entity resolution/clustering, directionality)
are not designed here. Stages B and C have their own design docs, both
updated after this data correction.
