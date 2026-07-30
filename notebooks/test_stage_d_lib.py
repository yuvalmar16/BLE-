import pytest
import pandas as pd
from stage_d_lib import (
    last_session_anchor, first_session_anchor,
    rssi_shape_score, transit_time_score, direction_candidate_score, best_direction_for_pair,
    match_travelers,
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
    assert rssi_shape_score(-0.3, 0.3) == pytest.approx(0.6)


def test_rssi_shape_score_nan_treated_as_neutral():
    assert rssi_shape_score(float("nan"), 0.3) == pytest.approx(0.3)
    assert rssi_shape_score(-0.3, float("nan")) == pytest.approx(0.3)


def test_rssi_shape_score_never_negative():
    assert rssi_shape_score(0.5, -0.5) == 0.0


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
        {"traveler_id": "B2", "first": _anchor("2026-05-02 12:10:25", -85, 0.3), "last": _anchor("2026-05-02 12:11:00", -65, 0.1)},
    ]
    matches = match_travelers(a_anchors, b_anchors, PARAMS)
    pairs = {(m["traveler_id_a"], m["traveler_id_b"]): m["direction"] for m in matches}
    assert pairs == {("A1", "B1"): "A_to_B", ("A2", "B2"): "A_to_B"}


def test_match_travelers_rejects_below_threshold():
    a_anchors = [{"traveler_id": "A1", "first": _anchor("2026-05-02 12:00:00", -90, 0.1), "last": _anchor("2026-05-02 12:00:40", -70, 0.1)}]
    b_anchors = [{"traveler_id": "B1", "first": _anchor("2026-05-02 13:00:00", -85, -0.1), "last": _anchor("2026-05-02 13:00:40", -65, 0.1)}]
    matches = match_travelers(a_anchors, b_anchors, PARAMS)
    assert matches == []


def test_match_travelers_empty_inputs():
    assert match_travelers([], [], PARAMS) == []
    non_empty_a = [{"traveler_id": "A1", "first": _anchor("2026-05-02 12:00:00", -90, 0.1), "last": _anchor("2026-05-02 12:00:40", -70, -0.3)}]
    assert match_travelers(non_empty_a, [], PARAMS) == []


from stage_d_lib import predicted_counts, calibration_loss, manual_to_internal


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


from stage_d_lib import grid_search, leave_one_date_out

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


def test_leave_one_date_out_reports_one_row_per_date():
    dates = {
        "2026-01-01": _one_pair_date(25.0, manual_A=1, manual_B=0),
        "2026-01-02": _one_pair_date(25.0, manual_A=1, manual_B=0),
        "2026-01-03": _one_pair_date(25.0, manual_A=1, manual_B=0),
    }
    rows = leave_one_date_out(dates, GRID, flip=False)
    assert {r["date"] for r in rows} == set(dates.keys())
    assert all(r["held_out_loss"] == 0.0 for r in rows)
