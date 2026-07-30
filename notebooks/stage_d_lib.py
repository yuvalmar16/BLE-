"""Stage D: cross-sensor traveler matching, direction, and calibration.

Pure logic only -- no file IO. See
docs/specs/2026-07-30-stage-d-direction-linking-design.md for the design
this implements.
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Characteristic magnitude of a session's `rssi_trend` (dB per second).
# Chosen from the real Stage-C session tables: |rssi_trend| has median
# ~0.085, p75 ~0.21, p90 ~0.52, p95 ~1.0 (both sensors). 0.5 therefore sits
# around the 90th percentile -- a trend that steep is already a clear,
# unambiguous fade/rise, so the shape score saturates there instead of
# letting the handful of |trend| up to 16 dominate everything else.
RSSI_TREND_SCALE = 0.5

# Physically implausible transits are rejected outright (see
# `direction_candidate_score`). The sensors are ~20m apart with an expected
# ~20-30s walking transit; 300s is a walking speed of 0.067 m/s, i.e. an
# order of magnitude of slack over "someone dawdled". Anything beyond this
# is a different trip, not a slow one.
DEFAULT_MAX_TRANSIT_S = 300.0

# Finite sentinel used to forbid below-threshold edges in the assignment
# problem. Deliberately finite rather than -inf: scipy's
# `linear_sum_assignment` raises on an infeasible matrix, and in real data
# the vast majority of candidate edges ARE below threshold, so whole rows
# would be -inf. A large constant penalty makes the optimizer minimise the
# number of forbidden edges it is forced to use first, and maximise the
# real score among valid edges second -- which is the behaviour we want.
FORBIDDEN_EDGE_SCORE = -1e6


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


def rssi_shape_score(departure_trend: float, arrival_trend: float,
                     trend_scale: float = RSSI_TREND_SCALE) -> float:
    """Bounded [0, 1] *agreement* score between "fading out at the departure
    sensor" and "strengthening at the arrival sensor" (design doc §2).

    Formula: each side is rectified (only the expected sign counts), scaled
    by `trend_scale` and clipped to [0, 1]; the score is their geometric
    mean::

        fade = clip(max(0, -departure_trend) / scale, 0, 1)
        rise = clip(max(0,  arrival_trend)   / scale, 0, 1)
        score = sqrt(fade * rise)

    Properties this is designed for:

    - **Bounded.** Max possible value is exactly 1.0, so this term is on the
      same scale as `transit_time_score` and `rssi_weight` genuinely trades
      the two off.
    - **Requires agreement.** The score is 0 unless BOTH anchors behave as a
      real transit would. A single steep trend on one side alone scores
      nothing.
    - **Saturating, not accumulating.** A trend twice as steep as
      `trend_scale` is not twice as convincing.
    - NaN trends (single-reading sessions) are substituted with 0 and never
      penalised below 0, as before. Note the consequence of requiring
      agreement: a NaN on either side yields 0, i.e. no RSSI support, so
      such pairs must be carried by the timing term alone. That is the
      honest reading of "we have no shape evidence here".

    This replaces the original ``max(0, -d) + max(0, a)``, which a review of
    the completed branch identified as the root cause of the direction
    column carrying no signal (finding C1). That form was an unbounded SUM
    of the two magnitudes: with real `rssi_trend` values reaching +/-16 and
    a calibrated `rssi_weight` of 5.0, a single steep trend on one side
    could contribute ~80 while the timing term is capped at 1.0 -- so
    timing, and any requirement that the two anchors actually agree, became
    irrelevant to the match.
    """
    d = 0.0 if pd.isna(departure_trend) else float(departure_trend)
    a = 0.0 if pd.isna(arrival_trend) else float(arrival_trend)
    fade = min(1.0, max(0.0, -d) / trend_scale)
    rise = min(1.0, max(0.0, a) / trend_scale)
    return math.sqrt(fade * rise)


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
                              center: float, spread: float, rssi_weight: float,
                              max_transit_s: float = DEFAULT_MAX_TRANSIT_S):
    """Score one directional candidate (departure sensor -> arrival
    sensor). Returns (score, gap_s).

    A hard physical gate is applied first: a gap that is negative (the
    "arrival" happened before the "departure") or longer than
    `max_transit_s` is not a transit at all, and scores exactly 0 -- the
    RSSI term is not even consulted. Without this gate a pair separated by
    hours could still clear the match threshold on RSSI shape alone
    (review finding I2), which is also what produced the reported
    `transit_gap_s` values of many hours in the first calibration run
    (finding I3).
    """
    gap_s = (arrival_time - departure_time).total_seconds()
    if gap_s < 0 or gap_s > max_transit_s:
        return 0.0, gap_s
    t_score = transit_time_score(gap_s, center, spread)
    r_score = rssi_shape_score(departure_trend, arrival_trend)
    return t_score + rssi_weight * r_score, gap_s


def best_direction_for_pair(a_first: dict, a_last: dict, b_first: dict, b_last: dict, params: dict):
    """Given one Sensor-A traveler's anchors and one Sensor-B traveler's
    anchors, score both directional candidates (A->B and B->A) and return
    the better one as (direction, gap_s, score).

    Because both candidates are gated on a positive, bounded gap, at most
    one of them can be physically valid for a given pair (a_first <= a_last
    and b_first <= b_last make "both gaps positive" impossible), so the
    winner is automatically the design doc's "whichever is positive and
    smaller" gap. If neither is valid both score 0 and the pair will fall
    below any positive `min_match_score`.
    """
    center = params["transit_center_s"]
    spread = params["transit_spread_s"]
    rssi_weight = params["rssi_weight"]
    max_transit_s = params.get("max_transit_s", DEFAULT_MAX_TRANSIT_S)
    score_ab, gap_ab = direction_candidate_score(
        a_last["time"], b_first["time"], a_last["rssi_trend"], b_first["rssi_trend"],
        center, spread, rssi_weight, max_transit_s,
    )
    score_ba, gap_ba = direction_candidate_score(
        b_last["time"], a_first["time"], b_last["rssi_trend"], a_first["rssi_trend"],
        center, spread, rssi_weight, max_transit_s,
    )
    if score_ab >= score_ba:
        return "A_to_B", gap_ab, score_ab
    return "B_to_A", gap_ba, score_ba


def match_travelers(a_anchors: list, b_anchors: list, params: dict) -> list:
    """Optimal bipartite matching between Sensor-A and Sensor-B travelers
    in the same window. Returns accepted matches only; pairs scoring below
    `params["min_match_score"]` are left unmatched.

    Below-threshold edges are *forbidden before* the assignment is solved
    (their weight is replaced by `FORBIDDEN_EDGE_SCORE`), not merely
    filtered from the result afterwards. Filtering afterwards lets the
    optimizer spend a traveler on a pairing it will then discard, blocking a
    valid pairing that traveler could have had (review finding I4). The
    post-hoc filter is still applied as a safety net, because a
    square/rectangular assignment can be forced onto a forbidden edge when
    nothing valid is left for that row or column.
    """
    n_a, n_b = len(a_anchors), len(b_anchors)
    if n_a == 0 or n_b == 0:
        return []

    min_score = params["min_match_score"]
    score = np.zeros((n_a, n_b))
    direction = [["" for _ in range(n_b)] for _ in range(n_a)]
    gap = np.zeros((n_a, n_b))
    for i, a in enumerate(a_anchors):
        for j, b in enumerate(b_anchors):
            d, g, s = best_direction_for_pair(a["first"], a["last"], b["first"], b["last"], params)
            score[i, j] = s
            direction[i][j] = d
            gap[i, j] = g

    solver_score = np.where(score >= min_score, score, FORBIDDEN_EDGE_SCORE)
    row_idx, col_idx = linear_sum_assignment(-solver_score)

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


def predicted_counts(matches: list) -> dict:
    """Aggregate matches into directional counts, same shape as the manual
    ground truth (via manual_to_internal)."""
    a_to_b = sum(1 for m in matches if m["direction"] == "A_to_B")
    b_to_a = sum(1 for m in matches if m["direction"] == "B_to_A")
    return {"A_to_B": a_to_b, "B_to_A": b_to_a, "total": a_to_b + b_to_a}


def calibration_loss(predicted: dict, manual: dict) -> float:
    """Squared error between predicted and manual {'A_to_B','B_to_A','total'}
    counts for one date."""
    return (
        (predicted["A_to_B"] - manual["A_to_B"]) ** 2
        + (predicted["B_to_A"] - manual["B_to_A"]) ** 2
        + (predicted["total"] - manual["total"]) ** 2
    )


def manual_to_internal(manual_A: int, manual_B: int, flip: bool) -> dict:
    """Map the manual CSV's raw direction counts onto our internal
    A_to_B/B_to_A vocabulary. The human observer's A/B convention isn't
    documented anywhere, so which physical direction manual 'A' means is
    unknown a priori; `flip` lets the calibration step try both
    orientations and keep whichever fits the ground truth better."""
    if flip:
        return {"A_to_B": manual_B, "B_to_A": manual_A, "total": manual_A + manual_B}
    return {"A_to_B": manual_A, "B_to_A": manual_B, "total": manual_A + manual_B}


def date_predictions(dates_data: dict, param_grid: list) -> dict:
    """Precompute `predicted_counts` for every (date, param) combination.

    Matching does not depend on the label orientation (`flip`) at all --
    only the loss comparison does -- so this cache is computed once and
    reused across both orientations and across every leave-one-date-out
    fold. That is what makes searching the orientation *inside* each fold
    (review finding I1) affordable.

    Returns {date: [counts_for_param_grid[0], counts_for_param_grid[1], ...]}.
    """
    return {
        date: [
            predicted_counts(match_travelers(d["a_anchors"], d["b_anchors"], params))
            for params in param_grid
        ]
        for date, d in dates_data.items()
    }


def _grid_search_index(dates_subset: dict, param_grid: list, flip: bool, preds: dict):
    """Core grid search; returns (best_param_index, best_total_loss)."""
    manual = {
        date: manual_to_internal(d["manual_A"], d["manual_B"], flip)
        for date, d in dates_subset.items()
    }
    best_k, best_loss = None, float("inf")
    for k in range(len(param_grid)):
        total = 0.0
        for date in dates_subset:
            total += calibration_loss(preds[date][k], manual[date])
        if total < best_loss:
            best_loss, best_k = total, k
    return best_k, best_loss


def grid_search(dates_subset: dict, param_grid: list, flip: bool, preds: dict = None):
    """Returns (best_params, best_total_loss): the params from
    `param_grid` minimizing summed calibration_loss across every date in
    `dates_subset`, under the given label orientation. `preds` is an
    optional `date_predictions` cache."""
    if preds is None:
        preds = date_predictions(dates_subset, param_grid)
    k, loss = _grid_search_index(dates_subset, param_grid, flip, preds)
    return (param_grid[k] if k is not None else None), loss


def grid_search_orientation(dates_subset: dict, param_grid: list, preds: dict = None):
    """Search grid params AND label orientation jointly.

    Returns (best_params, best_flip, best_total_loss). Used both for the
    final production fit and, crucially, *inside* each leave-one-date-out
    fold: the manual CSV's A/B convention is not documented, so choosing it
    is a fitted decision like any other and must be made from training
    dates only.
    """
    if preds is None:
        preds = date_predictions(dates_subset, param_grid)
    k, flip, loss = _grid_search_orientation_index(dates_subset, param_grid, preds)
    return (param_grid[k] if k is not None else None), flip, loss


def _grid_search_orientation_index(dates_subset: dict, param_grid: list, preds: dict):
    """Core joint param+orientation search; returns (index, flip, loss)."""
    best = (None, None, float("inf"))
    for flip in (False, True):
        k, loss = _grid_search_index(dates_subset, param_grid, flip, preds)
        if loss < best[2]:
            best = (k, flip, loss)
    return best


def leave_one_date_out(dates_data: dict, param_grid: list, preds: dict = None) -> list:
    """Leave-one-date-out CV: for each held-out date, fit params AND label
    orientation via `grid_search_orientation` on the rest, then evaluate the
    held-out date's own loss under that fit. One report row per date.

    The orientation is deliberately re-fitted per fold rather than chosen
    once over all 8 dates: choosing it globally and reusing it inside every
    fold leaks the held-out date's own ground truth into its evaluation
    (review finding I1).
    """
    if preds is None:
        preds = date_predictions(dates_data, param_grid)
    rows = []
    for held_out in dates_data:
        train = {d: v for d, v in dates_data.items() if d != held_out}
        k, flip, _ = _grid_search_orientation_index(train, param_grid, preds)
        d = dates_data[held_out]
        pred = preds[held_out][k]
        manual = manual_to_internal(d["manual_A"], d["manual_B"], flip)
        rows.append({
            "date": held_out, "params": param_grid[k], "flip": flip, "predicted": pred,
            "manual": manual, "held_out_loss": calibration_loss(pred, manual),
        })
    return rows
