import pytest
import pandas as pd
from stage_d_lib import (
    RSSI_TREND_SCALE, DEFAULT_MAX_TRANSIT_S,
    last_session_anchor, first_session_anchor,
    rssi_shape_score, transit_time_score, direction_candidate_score, best_direction_for_pair,
    match_travelers,
    predicted_counts, calibration_loss, manual_to_internal,
    date_predictions, grid_search, grid_search_orientation, leave_one_date_out,
)


def _sessions(rows):
    return pd.DataFrame(rows)


def test_last_session_anchor_picks_latest_end_time():
    df = _sessions([
        {"start_time": pd.Timestamp("2026-05-02 12:00:00"), "end_time": pd.Timestamp("2026-05-02 12:00:30"), "mean_rssi": -80.0, "rssi_trend": -0.1},
        {"start_time": pd.Timestamp("2026-05-02 12:05:00"), "end_time": pd.Timestamp("2026-05-02 12:05:20"), "mean_rssi": -70.0, "rssi_trend": 0.2},
    ])
    anchor = last_session_anchor(df)
    assert anchor["time"] == pd.Timestamp("2026-05-02 12:05:20")
    assert anchor["mean_rssi"] == -70.0
    assert anchor["rssi_trend"] == 0.2


def test_first_session_anchor_picks_earliest_start_time():
    df = _sessions([
        {"start_time": pd.Timestamp("2026-05-02 12:05:00"), "end_time": pd.Timestamp("2026-05-02 12:05:20"), "mean_rssi": -70.0, "rssi_trend": 0.2},
        {"start_time": pd.Timestamp("2026-05-02 12:00:00"), "end_time": pd.Timestamp("2026-05-02 12:00:30"), "mean_rssi": -80.0, "rssi_trend": -0.1},
    ])
    anchor = first_session_anchor(df)
    assert anchor["time"] == pd.Timestamp("2026-05-02 12:00:00")
    assert anchor["mean_rssi"] == -80.0
    assert anchor["rssi_trend"] == -0.1


def test_single_session_traveler_anchors_use_its_own_bounds():
    df = _sessions([
        {"start_time": pd.Timestamp("2026-05-02 12:00:00"), "end_time": pd.Timestamp("2026-05-02 12:00:05"), "mean_rssi": -90.0, "rssi_trend": float("nan")},
    ])
    first = first_session_anchor(df)
    last = last_session_anchor(df)
    assert first["time"] == pd.Timestamp("2026-05-02 12:00:00")
    assert last["time"] == pd.Timestamp("2026-05-02 12:00:05")


def test_rssi_shape_score_rewards_fade_out_fade_in():
    # fade = 0.3/0.5 = 0.6, rise = 0.6, geometric mean = 0.6
    assert rssi_shape_score(-0.3, 0.3) == pytest.approx(0.6)


def test_rssi_shape_score_nan_scores_zero_but_never_negative():
    # NaN is substituted with 0 (neutral, never a negative penalty), but the
    # score requires agreement from BOTH anchors, so a missing trend leaves
    # no RSSI evidence at all and the pair must rely on timing.
    assert rssi_shape_score(float("nan"), 0.3) == 0.0
    assert rssi_shape_score(-0.3, float("nan")) == 0.0
    assert rssi_shape_score(float("nan"), float("nan")) == 0.0


def test_rssi_shape_score_never_negative():
    assert rssi_shape_score(0.5, -0.5) == 0.0


def test_rssi_shape_score_is_bounded_at_one():
    # C1: the old unbounded sum-of-magnitudes form returned 32.0 here.
    assert rssi_shape_score(-16.0, 16.0) == 1.0
    assert rssi_shape_score(-1e6, 1e6) == 1.0
    for d, a in [(-16.0, 16.0), (-0.3, 0.3), (-5.0, 0.01), (0.0, 0.0), (2.0, -2.0)]:
        assert 0.0 <= rssi_shape_score(d, a) <= 1.0


def test_rssi_shape_score_requires_both_anchors_to_agree():
    # A very steep fade at the departure anchor is worth nothing on its own
    # if the arrival anchor is not strengthening -- the whole point of C1.
    assert rssi_shape_score(-16.0, -16.0) == 0.0
    assert rssi_shape_score(-16.0, 0.0) == 0.0
    assert rssi_shape_score(0.0, 16.0) == 0.0
    # ...and it must not beat a modest but genuinely agreeing pair.
    assert rssi_shape_score(-16.0, -16.0) < rssi_shape_score(-0.05, 0.05)


def test_steep_mismatched_trends_do_not_outscore_well_timed_pair():
    """C1 regression: under the old unbounded score, a pair with steep but
    physically inconsistent trends beat a well-timed pair with modest ones."""
    params = {"transit_center_s": 25.0, "transit_spread_s": 15.0, "rssi_weight": 2.0}
    t0 = pd.Timestamp("2026-05-02 12:00:00")
    # Well-timed (25s, dead centre) with modest but agreeing trends.
    good, _ = direction_candidate_score(t0, t0 + pd.Timedelta(seconds=25),
                                        -0.1, 0.1, 25.0, 15.0, 2.0)
    # Badly timed (250s) with enormous but disagreeing trends.
    bad, _ = direction_candidate_score(t0, t0 + pd.Timedelta(seconds=250),
                                       -16.0, -16.0, 25.0, 15.0, 2.0)
    assert good > bad
    assert params["rssi_weight"] * rssi_shape_score(-16.0, -16.0) == 0.0


def test_direction_candidate_score_rejects_over_long_transit():
    """I2: no gate previously existed, so an hours-apart pair could still
    clear the threshold on RSSI shape alone."""
    t0 = pd.Timestamp("2026-05-02 12:00:00")
    score, gap = direction_candidate_score(
        t0, t0 + pd.Timedelta(hours=2), -16.0, 16.0, 25.0, 15.0, 5.0,
    )
    assert score == 0.0
    assert gap == pytest.approx(7200.0)


def test_direction_candidate_score_rejects_negative_gap_outright():
    t0 = pd.Timestamp("2026-05-02 12:00:00")
    score, gap = direction_candidate_score(
        t0, t0 - pd.Timedelta(seconds=30), -16.0, 16.0, 25.0, 15.0, 5.0,
    )
    assert score == 0.0
    assert gap == pytest.approx(-30.0)


def test_direction_candidate_score_honours_custom_max_transit():
    t0 = pd.Timestamp("2026-05-02 12:00:00")
    args = (t0, t0 + pd.Timedelta(seconds=100), -0.5, 0.5, 100.0, 30.0, 1.0)
    assert direction_candidate_score(*args, 300.0)[0] > 0
    assert direction_candidate_score(*args, 60.0)[0] == 0.0
    assert DEFAULT_MAX_TRANSIT_S == 300.0
    assert RSSI_TREND_SCALE == 0.5


def test_transit_time_score_peaks_at_center():
    assert transit_time_score(25.0, center=25.0, spread=15.0) == pytest.approx(1.0)


def test_transit_time_score_decays_away_from_center():
    near = transit_time_score(25.0, center=25.0, spread=15.0)
    far = transit_time_score(200.0, center=25.0, spread=15.0)
    assert far < near


def test_transit_time_score_zero_for_negative_gap():
    assert transit_time_score(-5.0, center=25.0, spread=15.0) == 0.0


def test_best_direction_for_pair_picks_a_to_b_when_plausible():
    a_first = {"time": pd.Timestamp("2026-05-02 12:00:00"), "mean_rssi": -90.0, "rssi_trend": 0.1}
    a_last = {"time": pd.Timestamp("2026-05-02 12:00:40"), "mean_rssi": -70.0, "rssi_trend": -0.3}
    b_first = {"time": pd.Timestamp("2026-05-02 12:01:05"), "mean_rssi": -85.0, "rssi_trend": 0.3}
    b_last = {"time": pd.Timestamp("2026-05-02 12:01:40"), "mean_rssi": -65.0, "rssi_trend": 0.1}
    params = {"transit_center_s": 25.0, "transit_spread_s": 15.0, "rssi_weight": 1.0}
    direction, gap_s, score = best_direction_for_pair(a_first, a_last, b_first, b_last, params)
    assert direction == "A_to_B"
    assert gap_s == pytest.approx(25.0)
    assert score > 0


def test_best_direction_for_pair_picks_b_to_a_when_b_leaves_first():
    """M4: the mirrored path -- B's last session precedes A's first."""
    b_first = {"time": pd.Timestamp("2026-05-02 12:00:00"), "mean_rssi": -90.0, "rssi_trend": 0.1}
    b_last = {"time": pd.Timestamp("2026-05-02 12:00:40"), "mean_rssi": -70.0, "rssi_trend": -0.3}
    a_first = {"time": pd.Timestamp("2026-05-02 12:01:05"), "mean_rssi": -85.0, "rssi_trend": 0.3}
    a_last = {"time": pd.Timestamp("2026-05-02 12:01:40"), "mean_rssi": -65.0, "rssi_trend": 0.1}
    params = {"transit_center_s": 25.0, "transit_spread_s": 15.0, "rssi_weight": 1.0}
    direction, gap_s, score = best_direction_for_pair(a_first, a_last, b_first, b_last, params)
    assert direction == "B_to_A"
    assert gap_s == pytest.approx(25.0)
    assert score == pytest.approx(1.0 + 0.6)


def test_best_direction_for_pair_scores_zero_when_both_directions_implausible():
    """I3: with the transit gate in place, a pair whose two anchors are
    hours apart in *either* direction can no longer be rescued by a large
    RSSI term -- both candidates score 0 and the pair is rejected."""
    a_first = {"time": pd.Timestamp("2026-05-02 12:00:00"), "mean_rssi": -90.0, "rssi_trend": -16.0}
    a_last = {"time": pd.Timestamp("2026-05-02 12:00:40"), "mean_rssi": -70.0, "rssi_trend": -16.0}
    b_first = {"time": pd.Timestamp("2026-05-02 15:00:00"), "mean_rssi": -85.0, "rssi_trend": 16.0}
    b_last = {"time": pd.Timestamp("2026-05-02 15:00:40"), "mean_rssi": -65.0, "rssi_trend": 16.0}
    params = {"transit_center_s": 25.0, "transit_spread_s": 15.0, "rssi_weight": 5.0}
    _, _, score = best_direction_for_pair(a_first, a_last, b_first, b_last, params)
    assert score == 0.0


PARAMS = {"transit_center_s": 25.0, "transit_spread_s": 15.0, "rssi_weight": 1.0, "min_match_score": 0.3}


def _anchor(t, rssi, trend):
    return {"time": pd.Timestamp(t), "mean_rssi": rssi, "rssi_trend": trend}


def test_match_travelers_picks_correct_optimal_pairing():
    a_anchors = [
        {"traveler_id": "A1", "first": _anchor("2026-05-02 12:00:00", -90, 0.1), "last": _anchor("2026-05-02 12:00:40", -70, -0.3)},
        {"traveler_id": "A2", "first": _anchor("2026-05-02 12:10:00", -90, 0.1), "last": _anchor("2026-05-02 12:10:40", -70, -0.3)},
    ]
    b_anchors = [
        {"traveler_id": "B1", "first": _anchor("2026-05-02 12:01:05", -85, 0.3), "last": _anchor("2026-05-02 12:01:40", -65, 0.1)},
        {"traveler_id": "B2", "first": _anchor("2026-05-02 12:11:05", -85, 0.3), "last": _anchor("2026-05-02 12:11:40", -65, 0.1)},
    ]
    matches = match_travelers(a_anchors, b_anchors, PARAMS)
    pairs = {(m["traveler_id_a"], m["traveler_id_b"]): m["direction"] for m in matches}
    assert pairs == {("A1", "B1"): "A_to_B", ("A2", "B2"): "A_to_B"}


def test_match_travelers_handles_rectangular_input():
    """M4: more B travelers than A -- every A should still get its best
    valid partner and the surplus B traveler is simply left unmatched."""
    a_anchors = [
        {"traveler_id": "A1", "first": _anchor("2026-05-02 12:00:00", -90, 0.1), "last": _anchor("2026-05-02 12:00:40", -70, -0.3)},
    ]
    b_anchors = [
        {"traveler_id": "B_far", "first": _anchor("2026-05-02 12:04:00", -85, 0.3), "last": _anchor("2026-05-02 12:04:30", -65, 0.1)},
        {"traveler_id": "B_near", "first": _anchor("2026-05-02 12:01:05", -85, 0.3), "last": _anchor("2026-05-02 12:01:40", -65, 0.1)},
    ]
    matches = match_travelers(a_anchors, b_anchors, PARAMS)
    assert len(matches) == 1
    assert matches[0]["traveler_id_b"] == "B_near"


def _filter_only_match(a_anchors, b_anchors, params):
    """Reimplementation of the ORIGINAL (buggy) match_travelers: solve the
    assignment on raw scores, then drop below-threshold pairs afterwards.
    Used only to demonstrate that finding I4 is a real behavioural
    difference, not a cosmetic one."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from stage_d_lib import best_direction_for_pair as _bd
    n_a, n_b = len(a_anchors), len(b_anchors)
    score = np.zeros((n_a, n_b))
    for i, a in enumerate(a_anchors):
        for j, b in enumerate(b_anchors):
            score[i, j] = _bd(a["first"], a["last"], b["first"], b["last"], params)[2]
    rows, cols = linear_sum_assignment(-score)
    return [
        (a_anchors[i]["traveler_id"], b_anchors[j]["traveler_id"], float(score[i, j]))
        for i, j in zip(rows, cols) if score[i, j] >= params["min_match_score"]
    ]


def test_below_threshold_edges_are_forbidden_before_assignment():
    """I4: crafted 2x2 where the raw (filter-afterwards) optimum burns the
    only above-threshold edge on a pairing it then discards.

    Score matrix under these params (all trends NaN, so RSSI contributes 0
    and score == transit_time_score):

                  B1(gap 25s)   B2(gap 12s)
        A1            1.000         0.687
        A2            0.687         invalid (negative gap)

    min_match_score = 0.9, so A1-B1 is the only valid edge. Raw assignment
    maximises 0.687 + 0.687 = 1.374 over 1.000 + 0.0 and then filters both
    -> zero matches. Forbidding first keeps A1-B1.
    """
    nan = float("nan")
    params = {"transit_center_s": 25.0, "transit_spread_s": 15.0,
              "rssi_weight": 1.0, "min_match_score": 0.9}
    a_anchors = [
        {"traveler_id": "A1", "first": _anchor("2026-05-02 11:50:00", -90, nan), "last": _anchor("2026-05-02 12:00:00", -70, nan)},
        {"traveler_id": "A2", "first": _anchor("2026-05-02 11:50:00", -90, nan), "last": _anchor("2026-05-02 12:00:13", -70, nan)},
    ]
    b_anchors = [
        {"traveler_id": "B1", "first": _anchor("2026-05-02 12:00:25", -85, nan), "last": _anchor("2026-05-02 12:01:00", -65, nan)},
        {"traveler_id": "B2", "first": _anchor("2026-05-02 12:00:12", -85, nan), "last": _anchor("2026-05-02 12:00:40", -65, nan)},
    ]

    old = _filter_only_match(a_anchors, b_anchors, params)
    assert old == []  # the old behaviour loses the one good match entirely

    matches = match_travelers(a_anchors, b_anchors, params)
    assert [(m["traveler_id_a"], m["traveler_id_b"]) for m in matches] == [("A1", "B1")]
    assert matches[0]["match_score"] == pytest.approx(1.0)
    assert sum(m["match_score"] for m in matches) > sum(s for _, _, s in old)


def test_match_travelers_rejects_below_threshold():
    a_anchors = [{"traveler_id": "A1", "first": _anchor("2026-05-02 12:00:00", -90, 0.1), "last": _anchor("2026-05-02 12:00:40", -70, 0.1)}]
    b_anchors = [{"traveler_id": "B1", "first": _anchor("2026-05-02 13:00:00", -85, -0.1), "last": _anchor("2026-05-02 13:00:40", -65, 0.1)}]
    matches = match_travelers(a_anchors, b_anchors, PARAMS)
    assert matches == []


def test_match_travelers_empty_inputs():
    assert match_travelers([], [], PARAMS) == []
    non_empty_a = [{"traveler_id": "A1", "first": _anchor("2026-05-02 12:00:00", -90, 0.1), "last": _anchor("2026-05-02 12:00:40", -70, -0.3)}]
    assert match_travelers(non_empty_a, [], PARAMS) == []


def test_predicted_counts_tallies_directions():
    matches = [{"direction": "A_to_B"}, {"direction": "A_to_B"}, {"direction": "B_to_A"}]
    assert predicted_counts(matches) == {"A_to_B": 2, "B_to_A": 1, "total": 3}


def test_predicted_counts_empty():
    assert predicted_counts([]) == {"A_to_B": 0, "B_to_A": 0, "total": 0}


def test_calibration_loss_zero_when_exact_match():
    pred = {"A_to_B": 5, "B_to_A": 3, "total": 8}
    assert calibration_loss(pred, pred) == 0


def test_calibration_loss_positive_on_mismatch():
    pred = {"A_to_B": 5, "B_to_A": 3, "total": 8}
    manual = {"A_to_B": 4, "B_to_A": 3, "total": 7}
    assert calibration_loss(pred, manual) == 1 + 0 + 1


def test_manual_to_internal_no_flip():
    assert manual_to_internal(10, 4, flip=False) == {"A_to_B": 10, "B_to_A": 4, "total": 14}


def test_manual_to_internal_flip():
    assert manual_to_internal(10, 4, flip=True) == {"A_to_B": 4, "B_to_A": 10, "total": 14}


GRID = [
    {"transit_center_s": 25.0, "transit_spread_s": 15.0, "rssi_weight": 1.0, "min_match_score": 0.3},
    {"transit_center_s": 200.0, "transit_spread_s": 15.0, "rssi_weight": 1.0, "min_match_score": 0.3},
]


def _one_pair_date(gap_s, manual_A, manual_B):
    # Trends are neutral (0.0) on purpose: rssi_shape_score(0.0, 0.0) == 0,
    # so match_score reduces to transit_time_score alone. This makes the
    # grid's two candidate `transit_center_s` values genuinely
    # distinguishable by timing fit -- with a nonzero trend, both 25s and
    # 200s centers can score above min_match_score (RSSI similarity alone
    # clears the bar), and grid_search would "pick" 25.0 only by list-order
    # tie-breaking rather than real discrimination.
    a_first = _anchor("2026-01-01 10:00:00", -90, 0.0)
    a_last = _anchor("2026-01-01 10:00:00", -70, 0.0)
    b_time = pd.Timestamp("2026-01-01 10:00:00") + pd.Timedelta(seconds=gap_s)
    b_anchor = _anchor(b_time, -85, 0.0)
    return {
        "a_anchors": [{"traveler_id": "A1", "first": a_first, "last": a_last}],
        "b_anchors": [{"traveler_id": "B1", "first": b_anchor, "last": b_anchor}],
        "manual_A": manual_A, "manual_B": manual_B,
    }


def test_grid_search_picks_param_matching_true_transit_time():
    dates = {"2026-01-01": _one_pair_date(25.0, manual_A=1, manual_B=0)}
    best_params, best_loss = grid_search(dates, GRID, flip=False)
    assert best_params["transit_center_s"] == 25.0
    assert best_loss == 0.0


def test_date_predictions_cache_matches_uncached_grid_search():
    dates = {
        "2026-01-01": _one_pair_date(25.0, manual_A=1, manual_B=0),
        "2026-01-02": _one_pair_date(25.0, manual_A=1, manual_B=0),
    }
    preds = date_predictions(dates, GRID)
    assert set(preds) == set(dates)
    assert all(len(v) == len(GRID) for v in preds.values())
    assert grid_search(dates, GRID, flip=False, preds=preds) == grid_search(dates, GRID, flip=False)


def test_grid_search_orientation_picks_the_fitting_label_convention():
    # A single matched pair predicted as A_to_B; the manual log records it
    # under its own 'B' column, so only flip=True can fit.
    dates = {"2026-01-01": _one_pair_date(25.0, manual_A=0, manual_B=1)}
    params, flip, loss = grid_search_orientation(dates, GRID)
    assert flip is True
    assert params["transit_center_s"] == 25.0
    assert loss == 0.0


def test_leave_one_date_out_reports_one_row_per_date():
    dates = {
        "2026-01-01": _one_pair_date(25.0, manual_A=1, manual_B=0),
        "2026-01-02": _one_pair_date(25.0, manual_A=1, manual_B=0),
        "2026-01-03": _one_pair_date(25.0, manual_A=1, manual_B=0),
    }
    rows = leave_one_date_out(dates, GRID)
    assert {r["date"] for r in rows} == set(dates.keys())
    assert all(r["held_out_loss"] == 0.0 for r in rows)


def test_leave_one_date_out_fits_orientation_inside_each_fold():
    """I1: orientation is a fitted choice, so it must come from the fold's
    training dates only -- never from a search over all dates including the
    held-out one.

    Here 3 of 4 dates are recorded under the manual log's 'B' column
    (flip=True fits them) and 2026-01-04 alone is recorded under 'A'. A
    global orientation search would pick flip=True and reuse it everywhere.
    Fitting per fold must still pick flip=True for the 2026-01-04 fold
    (whose 3 training dates all favour it) and expose a large held-out loss
    for that date, rather than hiding it.
    """
    dates = {
        "2026-01-01": _one_pair_date(25.0, manual_A=0, manual_B=1),
        "2026-01-02": _one_pair_date(25.0, manual_A=0, manual_B=1),
        "2026-01-03": _one_pair_date(25.0, manual_A=0, manual_B=1),
        "2026-01-04": _one_pair_date(25.0, manual_A=1, manual_B=0),
    }
    rows = {r["date"]: r for r in leave_one_date_out(dates, GRID)}
    assert all(r["flip"] is True for r in rows.values())
    assert rows["2026-01-04"]["held_out_loss"] == 2.0
    assert all(rows[d]["held_out_loss"] == 0.0 for d in
               ["2026-01-01", "2026-01-02", "2026-01-03"])
