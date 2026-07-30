"""Stage D: cross-sensor traveler matching, direction, and calibration.

Pure logic only -- no file IO. See
docs/specs/2026-07-30-stage-d-direction-linking-design.md for the design
this implements.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def last_session_anchor(sessions_df: pd.DataFrame) -> dict:
    """Anchor derived from a traveler's LAST session -- used when they are
    about to leave a sensor's view (the departure side of a transit)."""
    row = sessions_df.sort_values("end_time").iloc[-1]
    return {"time": row["end_time"], "mean_rssi": row["mean_rssi"], "rssi_trend": row["rssi_trend"]}


def first_session_anchor(sessions_df: pd.DataFrame) -> dict:
    """Anchor derived from a traveler's FIRST session -- used when they are
    just entering a sensor's view (the arrival side of a transit)."""
    row = sessions_df.sort_values("start_time").iloc[0]
    return {"time": row["start_time"], "mean_rssi": row["mean_rssi"], "rssi_trend": row["rssi_trend"]}


def rssi_shape_score(departure_trend: float, arrival_trend: float) -> float:
    """Higher = more consistent with 'fading out at the departure sensor,
    strengthening at the arrival sensor'. NaN trends (single-reading
    sessions) are treated as neutral (0). Only rewards the expected sign,
    never penalizes the opposite -- avoids double-penalizing noise."""
    d = 0.0 if pd.isna(departure_trend) else departure_trend
    a = 0.0 if pd.isna(arrival_trend) else arrival_trend
    return max(0.0, -d) + max(0.0, a)


def transit_time_score(gap_s: float, center: float, spread: float) -> float:
    """Gaussian-shaped plausibility score for a transit gap, peaked at
    `center` seconds with standard deviation `spread`. Negative gaps
    (arrival anchor before departure anchor) are physically implausible
    for that direction and score 0."""
    if gap_s < 0:
        return 0.0
    z = (gap_s - center) / spread
    return float(np.exp(-0.5 * z * z))


def direction_candidate_score(departure_time, arrival_time, departure_trend, arrival_trend,
                               center: float, spread: float, rssi_weight: float):
    """Score one directional candidate (departure sensor -> arrival
    sensor). Returns (score, gap_s)."""
    gap_s = (arrival_time - departure_time).total_seconds()
    t_score = transit_time_score(gap_s, center, spread)
    r_score = rssi_shape_score(departure_trend, arrival_trend)
    return t_score + rssi_weight * r_score, gap_s


def best_direction_for_pair(a_first: dict, a_last: dict, b_first: dict, b_last: dict, params: dict):
    """Given one Sensor-A traveler's anchors and one Sensor-B traveler's
    anchors, score both directional candidates (A->B and B->A) and return
    the better one as (direction, gap_s, score)."""
    center = params["transit_center_s"]
    spread = params["transit_spread_s"]
    rssi_weight = params["rssi_weight"]
    score_ab, gap_ab = direction_candidate_score(
        a_last["time"], b_first["time"], a_last["rssi_trend"], b_first["rssi_trend"],
        center, spread, rssi_weight,
    )
    score_ba, gap_ba = direction_candidate_score(
        b_last["time"], a_first["time"], b_last["rssi_trend"], a_first["rssi_trend"],
        center, spread, rssi_weight,
    )
    if score_ab >= score_ba:
        return "A_to_B", gap_ab, score_ab
    return "B_to_A", gap_ba, score_ba


def match_travelers(a_anchors: list, b_anchors: list, params: dict) -> list:
    """Optimal bipartite matching between Sensor-A and Sensor-B travelers
    in the same window. Returns accepted matches only; pairs scoring below
    `params["min_match_score"]` are left unmatched."""
    n_a, n_b = len(a_anchors), len(b_anchors)
    if n_a == 0 or n_b == 0:
        return []

    score = np.zeros((n_a, n_b))
    direction = [["" for _ in range(n_b)] for _ in range(n_a)]
    gap = np.zeros((n_a, n_b))
    for i, a in enumerate(a_anchors):
        for j, b in enumerate(b_anchors):
            d, g, s = best_direction_for_pair(a["first"], a["last"], b["first"], b["last"], params)
            score[i, j] = s
            direction[i][j] = d
            gap[i, j] = g

    row_idx, col_idx = linear_sum_assignment(-score)

    min_score = params["min_match_score"]
    matches = []
    for i, j in zip(row_idx, col_idx):
        if score[i, j] < min_score:
            continue
        matches.append({
            "traveler_id_a": a_anchors[i]["traveler_id"],
            "traveler_id_b": b_anchors[j]["traveler_id"],
            "direction": direction[i][j],
            "transit_gap_s": float(gap[i, j]),
            "match_score": float(score[i, j]),
        })
    return matches
