"""One-off script: assembles notebooks/04_stage_d_direction_linking.ipynb.
Run once with `python _build_stage_d_notebook.py`, then execute the
resulting notebook with nbconvert. Not part of the pipeline itself.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


md("""
# Stage D: Cross-Sensor Identity Linking & Direction

See `docs/specs/2026-07-30-stage-d-direction-linking-design.md` for the
full design. This notebook: (1) reproduces the three ground-truth
findings that motivated the design, (2) calibrates the matcher against
`DATA/combined_file_manual.csv`, (3) applies it to all dual-sensor dates,
(4) writes the matched-pair output table.
""")

code("""
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from stage_d_lib import (
    first_session_anchor, last_session_anchor, best_direction_for_pair,
    match_travelers, predicted_counts, calibration_loss, manual_to_internal,
    date_predictions, grid_search, grid_search_orientation, leave_one_date_out,
    DEFAULT_MAX_TRANSIT_S, RSSI_TREND_SCALE,
)

DATA_DIR = "../DATA"
""")

md("## Load Stage C output and the manual ground truth")

code("""
sess_a = pd.read_csv(f"{DATA_DIR}/sessions_sensor_A_with_traveler.csv")
sess_b = pd.read_csv(f"{DATA_DIR}/sessions_sensor_B_with_traveler.csv")
for df in (sess_a, sess_b):
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

manual = pd.read_csv(f"{DATA_DIR}/combined_file_manual.csv")
manual["dt"] = pd.to_datetime(manual["Date"] + " " + manual["Time"], format="%d.%m.%Y %H:%M:%S")
manual["date"] = manual["dt"].dt.date
gt_windows = manual.groupby("date").agg(window_start=("dt", "min"), window_end=("dt", "max")).reset_index()
gt_dirs = manual.groupby(["date", "direction"]).size().unstack(fill_value=0).reset_index()
gt_windows = gt_windows.merge(gt_dirs, on="date")
print(f"Sensor A sessions: {len(sess_a)}, Sensor B sessions: {len(sess_b)}")
print(f"GT dates: {len(gt_windows)}")
gt_windows
""")

md("""
## Ground-truth finding 1: sensors see mostly the same people

Summing each sensor's independent traveler count double-counts, because
most travelers show up on both sensors within the same short window.
""")

code("""
def travelers_in_window(sess_df, ws, we):
    win = sess_df[(sess_df["start_time"] >= ws) & (sess_df["start_time"] <= we)]
    return win.groupby("Traveler_ID")["MAC"].apply(set)

overlap_rows = []
for _, r in gt_windows.iterrows():
    ta = travelers_in_window(sess_a, r["window_start"], r["window_end"])
    tb = travelers_in_window(sess_b, r["window_start"], r["window_end"])
    shared_macs = set().union(*ta.values) & set().union(*tb.values) if len(ta) and len(tb) else set()
    overlap_rows.append({
        "date": r["date"], "trav_A": len(ta), "trav_B": len(tb),
        "manual_total": int(r["A"] + r["B"]),
    })
overlap_df = pd.DataFrame(overlap_rows)
print("Sum of A+B traveler counts vs. manual total (should NOT match if double-counting):")
overlap_df
""")

md("""
## Ground-truth finding 2: raw MAC-address matching over-counts

Matching Sensor A and B by literal MAC equality (no per-sensor entity
resolution first) produces a much larger union than the manual total.
""")

code("""
a_raw = pd.read_parquet(f"{DATA_DIR}/sensor_A_deduped.parquet")[["MAC", "timestamp"]]
b1 = pd.read_excel(f"{DATA_DIR}/combined_file_sensor_B_part_1.xlsx")
b2 = pd.read_excel(f"{DATA_DIR}/combined_file_sensor_B_part_2.xlsx")
b_raw = pd.concat([b1, b2], ignore_index=True).drop_duplicates(subset=["MAC", "RSSI", "timestamp"])[["MAC", "timestamp"]]

raw_union_rows = []
for _, r in gt_windows.iterrows():
    ws, we = r["window_start"], r["window_end"]
    a_macs = set(a_raw[(a_raw["timestamp"] >= ws) & (a_raw["timestamp"] <= we)]["MAC"])
    b_macs = set(b_raw[(b_raw["timestamp"] >= ws) & (b_raw["timestamp"] <= we)]["MAC"])
    union_total = len(a_macs | b_macs)
    manual_total = int(r["A"] + r["B"])
    raw_union_rows.append({
        "date": r["date"], "raw_mac_union": union_total, "manual_total": manual_total,
        "ratio": round(union_total / manual_total, 2) if manual_total else None,
    })
pd.DataFrame(raw_union_rows)
""")

md("""
## Ground-truth finding 3: naive "first-seen" order is not a direction signal

Comparing which sensor first detected a shared MAC, against the manual
direction split, shows the opposite skew on some dates -- consistent with
Sensor A having a longer detection range, not true arrival order.
""")

code("""
first_seen_rows = []
for _, r in gt_windows.iterrows():
    ws, we = r["window_start"], r["window_end"]
    a_first = a_raw[(a_raw["timestamp"] >= ws) & (a_raw["timestamp"] <= we)].groupby("MAC")["timestamp"].min()
    b_first = b_raw[(b_raw["timestamp"] >= ws) & (b_raw["timestamp"] <= we)].groupby("MAC")["timestamp"].min()
    shared = set(a_first.index) & set(b_first.index)
    a_seen_first = sum(1 for m in shared if a_first[m] < b_first[m])
    b_seen_first = sum(1 for m in shared if b_first[m] < a_first[m])
    first_seen_rows.append({
        "date": r["date"], "A_seen_first": a_seen_first, "B_seen_first": b_seen_first,
        "manual_A": int(r["A"]), "manual_B": int(r["B"]),
    })
pd.DataFrame(first_seen_rows)
""")

md("## Build per-traveler boundary anchors for each GT window")

code("""
def build_anchors_for_window(sess_df, ws, we):
    win = sess_df[(sess_df["start_time"] >= ws) & (sess_df["start_time"] <= we)]
    anchors = []
    for tid, grp in win.groupby("Traveler_ID"):
        anchors.append({
            "traveler_id": tid,
            "first": first_session_anchor(grp),
            "last": last_session_anchor(grp),
        })
    return anchors

gt_dates_data = {}
for _, r in gt_windows.iterrows():
    ws, we = r["window_start"], r["window_end"]
    gt_dates_data[str(r["date"])] = {
        "a_anchors": build_anchors_for_window(sess_a, ws, we),
        "b_anchors": build_anchors_for_window(sess_b, ws, we),
        "manual_A": int(r["A"]), "manual_B": int(r["B"]),
    }

for date, d in gt_dates_data.items():
    print(f"{date}: {len(d['a_anchors'])} A travelers, {len(d['b_anchors'])} B travelers, "
          f"manual A={d['manual_A']} B={d['manual_B']}")
""")

md("""
### Structural ceiling: can the matcher even reach the manual totals?

A bipartite matching can produce at most `min(n_A, n_B)` pairs. If that
ceiling is below the manual ground-truth total for a date, **no** parameter
setting can hit that date's target, and the squared-error loss will reward
"match as many people as possible" on that date no matter what. Worth
checking before reading anything into the calibrated parameters.
""")

code("""
cap_rows = []
for date, d in gt_dates_data.items():
    cap = min(len(d["a_anchors"]), len(d["b_anchors"]))
    total = d["manual_A"] + d["manual_B"]
    cap_rows.append({"date": date, "n_A": len(d["a_anchors"]), "n_B": len(d["b_anchors"]),
                     "max_possible_matches": cap, "manual_total": total,
                     "reachable": cap >= total})
cap_df = pd.DataFrame(cap_rows)
print(f"{(~cap_df['reachable']).sum()} of {len(cap_df)} GT dates have a manual total "
      f"the matcher cannot reach even by pairing every available traveler.")
cap_df
""")

md("""
## Calibration: leave-one-date-out fit (params AND label orientation)

Free parameters (design doc §5): the transit-time window's centre/spread,
the `rssi_shape_score` weight relative to timing, and the minimum
match-score threshold. `max_transit_s` is *not* fitted — it is a hard
physical gate (see below), not a tuning knob.

Two things are fitted jointly, **both inside every CV fold**:

1. the four grid parameters, and
2. the label orientation `flip` — the manual CSV's A/B convention is
   undocumented, so which physical direction the observer's "A" means is
   itself an inference from the ground truth.

Choosing the orientation once over all 8 dates and then reusing it inside
each fold would leak the held-out date's own ground truth into its
evaluation, so `leave_one_date_out` re-fits it per fold from that fold's
7 training dates only.

### Score scale

`match_score = transit_time_score + rssi_weight * rssi_shape_score`, where
both terms are bounded to [0, 1]. So the maximum attainable score is
`1.0 + rssi_weight`, and `min_match_score` must be read against that.
Both terms are 0 for any candidate whose transit gap is negative or longer
than `max_transit_s` (300s), so such a pair can never be matched.
""")

code("""
import itertools

# Grid ranges are sized for the *bounded* [0,1] rssi_shape_score. The
# original grid (rssi_weight up to 5.0, min_match_score up to 0.6) was
# tuned against the old unbounded shape score, whose typical magnitude was
# an order of magnitude larger; carrying those ranges over would make the
# RSSI term dominate again. Max attainable score is 1.0 + rssi_weight.
#
# These ranges were then widened once, after a first fit put
# transit_spread_s, rssi_weight and min_match_score all on a grid edge:
# spread now reaches 60s, rssi_weight reaches down to 0.0 (an explicit
# "does the RSSI shape term help at all?" hypothesis, not a filler value)
# and min_match_score down to 0.05. Note min_match_score must stay > 0 --
# an out-of-gate candidate scores exactly 0 and is rejected only because
# the threshold is strictly positive.
PARAM_GRID = [
    {"transit_center_s": c, "transit_spread_s": s, "rssi_weight": w,
     "min_match_score": m, "max_transit_s": DEFAULT_MAX_TRANSIT_S}
    for c, s, w, m in itertools.product(
        [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0],
        [8.0, 12.0, 20.0, 30.0, 45.0, 60.0],
        [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
        [0.05, 0.15, 0.3, 0.5, 0.8, 1.1],
    )
]
print(f"Grid size: {len(PARAM_GRID)}")
print(f"rssi_shape_score saturation scale: {RSSI_TREND_SCALE} dB/s; "
      f"max transit gate: {DEFAULT_MAX_TRANSIT_S:.0f}s")

# Matching is independent of the label orientation, so every (date, params)
# match is computed exactly once and reused across both orientations and
# all 8 CV folds.
PREDS = date_predictions(gt_dates_data, PARAM_GRID)

for flip in (False, True):
    _, loss = grid_search(gt_dates_data, PARAM_GRID, flip, preds=PREDS)
    print(f"in-sample best total loss with flip={flip}: {loss:.1f}")
""")

code("""
loo_rows = leave_one_date_out(gt_dates_data, PARAM_GRID, preds=PREDS)
loo_report = pd.DataFrame([
    {
        "date": r["date"], "fold_flip": r["flip"],
        "predicted_A_to_B": r["predicted"]["A_to_B"], "predicted_B_to_A": r["predicted"]["B_to_A"],
        "predicted_total": r["predicted"]["total"],
        "manual_A_to_B": r["manual"]["A_to_B"], "manual_B_to_A": r["manual"]["B_to_A"],
        "manual_total": r["manual"]["total"],
        "held_out_loss": r["held_out_loss"],
    }
    for r in loo_rows
])
print("Leave-one-date-out held-out predictions vs. ground truth")
print("(params and orientation fitted on the other 7 dates only):")
for r in loo_rows:
    print(f"  {r['date']}: flip={r['flip']} params={r['params']}")
loo_report
""")

code("""
final_params, best_orientation, final_loss = grid_search_orientation(
    gt_dates_data, PARAM_GRID, preds=PREDS)
print(f"Final production params (fit on all 8 GT dates): {final_params}")
print(f"Chosen label orientation: flip={best_orientation}")
print(f"In-sample total loss: {final_loss:.1f}  |  mean held-out loss: {loo_report['held_out_loss'].mean():.1f}")

# Boundary check: a fitted value sitting at the edge of its searched range
# means the true optimum is probably outside the grid.
for key in ("transit_center_s", "transit_spread_s", "rssi_weight", "min_match_score"):
    vals = sorted({p[key] for p in PARAM_GRID})
    v = final_params[key]
    edge = " <-- ON GRID BOUNDARY" if v in (vals[0], vals[-1]) else ""
    print(f"  {key}: {v} in {vals}{edge}")
""")

md("""
### Is a boundary hit a truncated grid, or a saturation plateau?

A fitted value on the edge of its searched range usually means the true
optimum lies outside the grid. Not necessarily here: the loosest corner of
this grid (small centre, huge spread, zero RSSI weight, near-zero
threshold) makes the score nearly constant across the whole
`(0, max_transit_s]` gate, so the matcher saturates at "pair everyone you
can within the gate" and *cannot* go further. The check below evaluates
several explicitly out-of-grid parameter sets: if their loss is no better
than the fitted corner's, the boundary is a plateau, not a truncation, and
widening the grid again would buy nothing.
""")

code("""
beyond_grid = [
    ("fitted optimum (grid corner)", final_params),
    ("centre 5s, spread 150s", {**final_params, "transit_center_s": 5.0, "transit_spread_s": 150.0}),
    ("centre 1s, spread 1000s, min 0.001", {**final_params, "transit_center_s": 1.0,
                                            "transit_spread_s": 1000.0, "min_match_score": 0.001}),
    ("flat limit (spread 1e6, min 1e-9)", {**final_params, "transit_center_s": 0.0,
                                           "transit_spread_s": 1e6, "min_match_score": 1e-9}),
]
probe_rows = []
for label, p in beyond_grid:
    loss, totals = 0.0, []
    for date, d in gt_dates_data.items():
        pr = predicted_counts(match_travelers(d["a_anchors"], d["b_anchors"], p))
        loss += calibration_loss(pr, manual_to_internal(d["manual_A"], d["manual_B"], best_orientation))
        totals.append(pr["total"])
    probe_rows.append({"params": label, "in_sample_loss": loss, "per_date_totals": totals})
probe_df = pd.DataFrame(probe_rows)
print("If no out-of-grid row beats the fitted corner, the boundary is a plateau:")
probe_df
""")

md("""
### Does the direction split actually track ground truth?

The aggregate loss can look acceptable while the *direction* column is
worthless (a near-50/50 prediction on a date whose true traffic is
overwhelmingly one-way still gets the total right). This table checks the
directional skew explicitly: predicted A→B share vs. manual A→B share,
per GT date, using the held-out (leave-one-date-out) predictions.
""")

code("""
skew = loo_report.copy()
skew["pred_A_share"] = (skew["predicted_A_to_B"] /
                        skew["predicted_total"].replace(0, np.nan)).round(3)
skew["manual_A_share"] = (skew["manual_A_to_B"] / skew["manual_total"]).round(3)
skew["share_error"] = (skew["pred_A_share"] - skew["manual_A_share"]).round(3)
skew_report = skew[["date", "fold_flip", "predicted_A_to_B", "predicted_B_to_A",
                    "manual_A_to_B", "manual_B_to_A",
                    "pred_A_share", "manual_A_share", "share_error"]]
mean_abs_err = skew_report["share_error"].abs().mean()
# Baseline: a model that always predicts a 50/50 split, i.e. no direction
# signal whatsoever. If our error is not clearly below this, the direction
# column is not carrying information.
coin_flip_err = (0.5 - skew["manual_A_share"]).abs().mean()
corr = skew["pred_A_share"].corr(skew["manual_A_share"])
print(f"Mean |share error|:            {mean_abs_err:.3f}")
print(f"Always-50/50 baseline error:   {coin_flip_err:.3f}")
print(f"corr(predicted share, manual share) across the 8 dates: {corr:.3f}")
print("(0.0 error = direction perfectly recovered. An error at or above the "
      "baseline, or a correlation near/below 0, means the direction column "
      "is a coin flip and should not be trusted downstream.)")
skew_report
""")

md("""
## Apply to all 14 dual-sensor dates

Not just the 8 GT dates -- the calibrated matcher is applied to every
date where both sensors have data.
""")

code("""
a_dates = set(sess_a["start_time"].dt.date)
b_dates = set(sess_b["start_time"].dt.date)
dual_dates = sorted(a_dates & b_dates)
print(f"{len(dual_dates)} dual-sensor dates: {dual_dates}")

all_matches = []
unmatched_rows = []
for date in dual_dates:
    ws = pd.Timestamp(date)
    we = ws + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    a_anchors = build_anchors_for_window(sess_a, ws, we)
    b_anchors = build_anchors_for_window(sess_b, ws, we)
    matches = match_travelers(a_anchors, b_anchors, final_params)
    for m in matches:
        m["date"] = date
    all_matches.extend(matches)

    matched_a = {m["traveler_id_a"] for m in matches}
    matched_b = {m["traveler_id_b"] for m in matches}
    for a in a_anchors:
        if a["traveler_id"] not in matched_a:
            unmatched_rows.append({"date": date, "sensor": "A", "traveler_id": a["traveler_id"]})
    for b in b_anchors:
        if b["traveler_id"] not in matched_b:
            unmatched_rows.append({"date": date, "sensor": "B", "traveler_id": b["traveler_id"]})

matched_df = pd.DataFrame(all_matches)
unmatched_df = pd.DataFrame(unmatched_rows)
print(f"Total matched pairs across all dual-sensor dates: {len(matched_df)}")
print(f"Total unmatched (single-sensor-only) travelers: {len(unmatched_df)}")
print()
print("Per-date direction split of the matched pairs:")
per_date = matched_df.groupby(["date", "direction"]).size().unstack(fill_value=0)
per_date["total"] = per_date.sum(axis=1)
if "A_to_B" in per_date:
    per_date["A_to_B_share"] = (per_date["A_to_B"] / per_date["total"]).round(3)
print(per_date.to_string())
""")

md("## Validation: spot-check a few matched pairs")

code("""
sample = matched_df.sort_values("match_score", ascending=False).head(5)
for _, row in sample.iterrows():
    print(f"{row['date']}  {row['traveler_id_a']} <-> {row['traveler_id_b']}  "
          f"direction={row['direction']}  gap={row['transit_gap_s']:.1f}s  score={row['match_score']:.2f}")
""")

md("""
## Write output

Design doc §6 requires the output to carry **both** matched pairs and the
travelers only one sensor ever saw, the latter with a null direction.

### Schema of `DATA/travelers_matched_direction.csv` / `.parquet`

| column | matched row | unmatched row |
|---|---|---|
| `date` | collection date | collection date |
| `match_status` | `matched` | `unmatched_a_only` / `unmatched_b_only` |
| `traveler_id_a` | Stage-C Sensor-A `Traveler_ID` | set for `unmatched_a_only`, else NaN |
| `traveler_id_b` | Stage-C Sensor-B `Traveler_ID` | set for `unmatched_b_only`, else NaN |
| `direction` | `A_to_B` or `B_to_A` | **NaN** — no direction is inferable |
| `transit_gap_s` | seconds between the two boundary anchors; always in `(0, max_transit_s]` | NaN |
| `match_score` | see scale below | NaN |

`match_score = transit_time_score + rssi_weight * rssi_shape_score`, both
terms bounded to [0, 1], so the range is `[min_match_score, 1 + rssi_weight]`
with the calibrated `rssi_weight`. It is a plausibility score, not a
probability. `transit_gap_s` can no longer be negative or absurdly large:
candidates outside `(0, max_transit_s]` score exactly 0 and are rejected.
""")

code("""
out_cols = ["date", "match_status", "traveler_id_a", "traveler_id_b",
            "direction", "transit_gap_s", "match_score"]

matched_out = matched_df.copy()
matched_out["match_status"] = "matched"

unmatched_out = unmatched_df.copy()
if len(unmatched_out):
    unmatched_out["match_status"] = unmatched_out["sensor"].map(
        {"A": "unmatched_a_only", "B": "unmatched_b_only"})
    unmatched_out["traveler_id_a"] = unmatched_out["traveler_id"].where(unmatched_out["sensor"] == "A")
    unmatched_out["traveler_id_b"] = unmatched_out["traveler_id"].where(unmatched_out["sensor"] == "B")
    for c in ("direction", "transit_gap_s", "match_score"):
        unmatched_out[c] = np.nan
    unmatched_out = unmatched_out[out_cols]
else:
    unmatched_out = pd.DataFrame(columns=out_cols)

out = pd.concat([matched_out[out_cols], unmatched_out], ignore_index=True)
out = out.sort_values(["date", "match_status"]).reset_index(drop=True)
out.to_csv(f"{DATA_DIR}/travelers_matched_direction.csv", index=False)
out.to_parquet(f"{DATA_DIR}/travelers_matched_direction.parquet", index=False)
print(f"wrote {len(out)} rows to travelers_matched_direction.csv / .parquet")
print(out["match_status"].value_counts().to_string())
print("\\ndirection value counts (NaN = unmatched, by construction):")
print(out["direction"].value_counts(dropna=False).to_string())
print("\\ntransit_gap_s of matched rows:")
print(out["transit_gap_s"].describe().round(2).to_string())
print(f"negative transit_gap_s rows: {(out['transit_gap_s'] < 0).sum()} (must be 0)")
""")

nb["cells"] = cells
with open("04_stage_d_direction_linking.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote 04_stage_d_direction_linking.ipynb")
