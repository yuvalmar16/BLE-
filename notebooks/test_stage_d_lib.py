import pandas as pd
from stage_d_lib import last_session_anchor, first_session_anchor


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
