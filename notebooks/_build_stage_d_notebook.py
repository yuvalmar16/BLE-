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
    grid_search, leave_one_date_out,
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
## Calibration: determine label orientation, then leave-one-date-out fit

The manual CSV's A/B convention isn't documented, so both orientations
are tried; whichever gives the lower total in-sample loss (across all 8
GT dates, using the best grid params for that orientation) is kept.
""")

code("""
import itertools

PARAM_GRID = [
    {"transit_center_s": c, "transit_spread_s": s, "rssi_weight": w, "min_match_score": m}
    for c, s, w, m in itertools.product(
        [15.0, 20.0, 25.0, 30.0, 40.0],
        [10.0, 15.0, 20.0, 30.0],
        [0.5, 1.0, 2.0, 5.0],
        [0.2, 0.4, 0.6],
    )
]
print(f"Grid size: {len(PARAM_GRID)}")

best_orientation = None
best_orientation_loss = float("inf")
for flip in (False, True):
    _, loss = grid_search(gt_dates_data, PARAM_GRID, flip)
    print(f"flip={flip}: best in-sample total loss = {loss:.1f}")
    if loss < best_orientation_loss:
        best_orientation_loss, best_orientation = loss, flip

print(f"\\nChosen orientation: flip={best_orientation}")
""")

code("""
loo_rows = leave_one_date_out(gt_dates_data, PARAM_GRID, best_orientation)
loo_report = pd.DataFrame([
    {
        "date": r["date"],
        "predicted_A_to_B": r["predicted"]["A_to_B"], "predicted_B_to_A": r["predicted"]["B_to_A"],
        "predicted_total": r["predicted"]["total"],
        "manual_A_to_B": r["manual"]["A_to_B"], "manual_B_to_A": r["manual"]["B_to_A"],
        "manual_total": r["manual"]["total"],
        "held_out_loss": r["held_out_loss"],
    }
    for r in loo_rows
])
print("Leave-one-date-out held-out predictions vs. ground truth:")
loo_report
""")

code("""
final_params, final_loss = grid_search(gt_dates_data, PARAM_GRID, best_orientation)
print(f"Final production params (fit on all 8 GT dates): {final_params}")
print(f"In-sample total loss: {final_loss:.1f}  |  mean held-out loss: {loo_report['held_out_loss'].mean():.1f}")
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
""")

md("## Validation: spot-check a few matched pairs")

code("""
sample = matched_df.sort_values("match_score", ascending=False).head(5)
for _, row in sample.iterrows():
    print(f"{row['date']}  {row['traveler_id_a']} <-> {row['traveler_id_b']}  "
          f"direction={row['direction']}  gap={row['transit_gap_s']:.1f}s  score={row['match_score']:.2f}")
""")

md("## Write output")

code("""
out_cols = ["date", "traveler_id_a", "traveler_id_b", "direction", "transit_gap_s", "match_score"]
matched_out = matched_df[out_cols]
matched_out.to_csv(f"{DATA_DIR}/travelers_matched_direction.csv", index=False)
matched_out.to_parquet(f"{DATA_DIR}/travelers_matched_direction.parquet", index=False)
print(f"wrote {len(matched_out)} matched pairs to travelers_matched_direction.csv / .parquet")
""")

nb["cells"] = cells
with open("04_stage_d_direction_linking.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote 04_stage_d_direction_linking.ipynb")
