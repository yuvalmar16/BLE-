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
