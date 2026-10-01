# Hiker Monitoring via BLE Sensors

Estimating pedestrian/hiker traffic volume and direction of travel on a trail
from two cheap Arduino-based Bluetooth Low Energy (BLE) sniffers, using
unsupervised machine learning to de-duplicate devices and link detections
across sensors. See [`hiker_monitoring_project_summary.md`](hiker_monitoring_project_summary.md)
(Hebrew) for the original project brief.

## Problem

Two BLE sensors (**Sensor A** and **Sensor B**) are placed a short distance
apart on a trail, with an overlapping detection radius. Counting raw MAC
addresses overcounts badly because:

- **One hiker carries multiple BLE devices** (phone, watch, earbuds, ...).
- **MAC randomization**: modern devices rotate their advertised MAC address
  every few minutes for privacy.

There is no ground truth labelling which MAC belongs to which person, so the
pipeline is unsupervised, with a small manually-labelled subset
(`DATA/combined_file_manual.csv`) used only to calibrate and validate Stage D.

## Pipeline

Four stages, each a notebook under `notebooks/`, run in order. Every stage
reads the previous stage's output from `DATA/` and writes its own output back
to `DATA/` (both `.csv` and `.parquet`).

| Stage | Notebook | Design doc | Reads | Writes |
|---|---|---|---|---|
| A — Time reconstruction & dedup | `01_stage_a_time_reconstruction.ipynb` | [spec](docs/specs/2026-07-29-stage-a-time-reconstruction-design.md) | `Sensor_A_combined_file.xlsx` | `sensor_A_deduped.{csv,parquet}` |
| B — Sessionization | `02_stage_b_sessionization.ipynb` | [spec](docs/specs/2026-07-29-stage-b-sessionization-design.md) | `sensor_A_deduped.parquet`, `combined_file_sensor_B_part_{1,2}.xlsx` | `sessions_sensor_{A,B}.{csv,parquet}` |
| C — Entity resolution | `03_stage_c_entity_resolution.ipynb` | [spec](docs/specs/2026-07-29-stage-c-entity-resolution-design.md) | `sessions_sensor_{A,B}.parquet` | `sessions_sensor_{A,B}_with_traveler.{csv,parquet}`, `travelers_sensor_{A,B}.{csv,parquet}`, `travelers_per_day.{csv,parquet}` |
| D — Cross-sensor linking & direction | `04_stage_d_direction_linking.ipynb` | [spec](docs/specs/2026-07-30-stage-d-direction-linking-design.md) | `sessions_sensor_{A,B}_with_traveler.csv`, `combined_file_manual.csv` | `travelers_matched_direction.{csv,parquet}` |

- **Stage A**: deduplicates Sensor A's raw export (several cumulative
  re-exports of the same on-device log overlap). Sensor A now logs true
  second-resolution timestamps, so the stage no longer needs to interpolate
  minute-resolution timestamps (see the notebook's "Update" note).
- **Stage B**: dedups Sensor B the same way, then groups each MAC's readings
  into "visit" sessions (new session after a >60s gap) and extracts per-session
  features (duration, mean RSSI, `rssi_trend`).
- **Stage C**: merges MAC addresses likely belonging to the same physical
  hiker (multi-device + mid-visit MAC rotation) into one `Traveler_ID` per
  sensor, via blocking + complete-linkage clustering on session overlap and
  RSSI-trend similarity. No ground truth here, so the merge threshold is an
  inspected, documented choice (10th percentile of candidate-pair
  dissimilarity), not a fit.
- **Stage D**: links each sensor's `Traveler_ID`s across sensors into
  cross-sensor pairs (bipartite matching on transit-time + RSSI-shape +
  boundary-order + neighborhood-consistency + same-MAC score) and assigns a
  direction (`A_to_B` / `B_to_A`) to each pair. Calibrated and validated with
  leave-one-date-out cross-validation against the 8 manually-labelled dates in
  `combined_file_manual.csv`, then applied to all 14 dual-sensor dates. See
  the design doc's follow-up investigations and
  [`docs/reports/2026-08-13-stage-d-best-method-summary.html`](docs/reports/2026-08-13-stage-d-best-method-summary.html)
  for the honest, current take on how well direction detection actually
  works — it is **not** reliable enough for per-person claims.

Final output schema (`DATA/travelers_matched_direction.csv`): one row per
matched cross-sensor pair or per sensor-only traveler, with columns `date`,
`match_status` (`matched` / `unmatched_a_only` / `unmatched_b_only`),
`traveler_id_a`, `traveler_id_b`, `direction`, `transit_gap_s`, `match_score`.

## Setup

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.11+.

## Running the pipeline

All four stages are checked-in notebooks; `DATA/` already contains every
stage's output from the last run, so you can open any notebook directly to
inspect results. To regenerate everything from the raw sensor exports, run
the notebooks **in order** from the `notebooks/` directory, since each stage
reads the previous stage's output:

```bash
cd notebooks
python -m nbconvert --to notebook --execute --inplace 01_stage_a_time_reconstruction.ipynb
python -m nbconvert --to notebook --execute --inplace 02_stage_b_sessionization.ipynb
python -m nbconvert --to notebook --execute --inplace 03_stage_c_entity_resolution.ipynb
python -m nbconvert --to notebook --execute --inplace 04_stage_d_direction_linking.ipynb
```

`notebooks/04_stage_d_direction_linking.ipynb` is itself generated by
`notebooks/_build_stage_d_notebook.py` (kept so the notebook's cell-by-cell
structure is reproducible from source rather than hand-edited). To rebuild it
from scratch before executing:

```bash
cd notebooks
python _build_stage_d_notebook.py
python -m nbconvert --to notebook --execute --inplace 04_stage_d_direction_linking.ipynb
```

Stages A-D are all self-contained within their notebook (no shared library
module) — this was a deliberate convention, including for Stage D, whose
matching/scoring/calibration logic previously lived in a separate
`stage_d_lib.py` module and has since been folded into the notebook itself.

## Data

`DATA/` holds both the raw sensor exports and every intermediate/final
pipeline output (checked in so the repo is runnable without the original
field hardware). Raw inputs:

- `Sensor_A_combined_file.xlsx` — Sensor A's full device log (second
  resolution).
- `combined_file_sensor_B_part_1.xlsx`, `combined_file_sensor_B_part_2.xlsx`
  — Sensor B's device log, split across two export files.
- `combined_file_manual.csv` — manually labelled ground truth (8 dates) used
  only to calibrate/validate Stage D.

`combined_file_sensor_A.csv`, `data_A_030126.xlsx`, and
`sensor_A_reconstructed.{csv,parquet}` are leftovers from an earlier version
of Stage A (back when Sensor A only had minute-resolution timestamps and
needed interpolation); no current notebook reads them.

## Reports

Generated write-ups under `docs/reports/`, most recent first:

- [`2026-08-13-stage-d-best-method-summary.html`](docs/reports/2026-08-13-stage-d-best-method-summary.html)
  — current best-method summary for Stage D.
- [`2026-08-01-stage-d-direction-linking-report.html`](docs/reports/2026-08-01-stage-d-direction-linking-report.html)
  (and `-he.html` Hebrew translation) — full Stage D report.
- [`2026-07-30-current-status-he.html`](docs/reports/2026-07-30-current-status-he.html),
  [`2026-07-29-recap-he.html`](docs/reports/2026-07-29-recap-he.html) — earlier
  status recaps (Hebrew). PDF versions sit alongside each HTML report.

## Repo layout

```
DATA/               raw sensor exports + all pipeline intermediate/final outputs
notebooks/          01-04 stage notebooks, + Stage D's notebook-builder script
docs/specs/          per-stage design docs
docs/reports/        generated HTML/PDF reports and their figures
docs/superpowers/    implementation plan used to build Stage D
hiker_monitoring_project_summary.md   original project brief (Hebrew)
```
