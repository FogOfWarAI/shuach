# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Predator-outcome prediction from pre-match fields and completed history.

These datasets do not identify the four prey players. This adapter therefore
supports an asymmetric outcome task, not reconstruction of 1v4 assignments.
"""

import heapq
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .heroes import file_hash

DBD_SOURCES = {
    "dbd-2022": ("raw/dbd-2022.csv", "player_id"),
    "dbd-2023": ("raw/dbd-2023.csv", "predator_id"),
}
NUMERIC_FEATURES = [
    "log1p_completed",
    "mean_previous_kills",
    "last_previous_kills",
    "log1p_days_since_feedback",
] + [f"previous_kill_fraction_{k}" for k in range(5)]
CATEGORICAL_FEATURES = ["environment_id", "avatar_id"]


@dataclass
class AsymmetricData:
    features: pd.DataFrame
    outcome: np.ndarray
    row_ids: np.ndarray
    player_ids: np.ndarray
    time: np.ndarray
    available_at: np.ndarray
    manifest: dict


def build_predator_features(frame, player_column, feedback_delay=0):
    """Strictly delayed cumulative features; source total_xp is never read."""
    if not np.isfinite(feedback_delay) or feedback_delay < 0:
        raise ValueError("Feedback delay must be finite and nonnegative")
    frame = frame.copy()
    parsed = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
    if parsed.isna().any():
        raise ValueError("Missing event timestamps")
    # pandas can use microsecond or nanosecond datetime storage. An integer
    # cast divided by 1e9 silently changes time semantics across versions.
    frame["_time"] = parsed.dt.as_unit("ns").astype("int64") / 1e9
    frame["_available"] = (
        frame["_time"] + frame["game_duration"].to_numpy(dtype=float) + feedback_delay
    )
    frame = frame.sort_values(["_time", "match_id"]).reset_index(drop=True)
    if frame.match_id.duplicated().any():
        raise ValueError("Duplicate match IDs require explicit resolution")
    if frame[[player_column, *CATEGORICAL_FEATURES]].isna().any().any():
        raise ValueError("Missing required pre-match entity identifiers")
    if (
        not np.isfinite(frame[["_time", "_available", "hunting_success"]]).all().all()
        or (frame["game_duration"] <= 0).any()
    ):
        raise ValueError("Invalid event times, durations or outcomes")
    y = frame["hunting_success"].to_numpy()
    if not np.isin(y, np.arange(5)).all():
        raise ValueError("Expected integer prey-kill count from 0 to 4")
    players = frame[player_column].astype(str).to_numpy()
    starts, ends = frame["_time"].to_numpy(), frame["_available"].to_numpy()
    stats = defaultdict(lambda: [np.zeros(5, dtype=float), 2.0, None])
    pending, values = [], np.empty((len(frame), len(NUMERIC_FEATURES)), dtype=np.float32)
    for i in range(len(frame)):
        while pending and pending[0][0] < starts[i]:
            _, j = heapq.heappop(pending)
            state = stats[players[j]]
            state[0][int(y[j])] += 1
            state[1], state[2] = float(y[j]), ends[j]
        counts, last, available = stats[players[i]]
        n = counts.sum()
        # Symmetric Dirichlet smoothing is fixed a priori, not fit to test.
        frequencies = (counts + 1) / (n + 5)
        values[i] = [
            np.log1p(n),
            float(frequencies @ np.arange(5)),
            last,
            np.log1p((starts[i] - available) / 86400) if available is not None else 0,
            *frequencies,
        ]
        heapq.heappush(pending, (ends[i], i))
    features = pd.DataFrame(values, columns=NUMERIC_FEATURES)
    for column in CATEGORICAL_FEATURES:
        features[column] = frame[column].astype(str)
    return frame, features


def load_asymmetric(dataset, root: Path, feedback_delay=0):
    if dataset not in DBD_SOURCES or feedback_delay < 0:
        raise ValueError("Unsupported asymmetric dataset or negative delay")
    relative, player_column = DBD_SOURCES[dataset]
    path = root / relative
    raw = pd.read_csv(path)
    frame, features = build_predator_features(raw, player_column, feedback_delay)
    manifest = {
        "dataset": dataset,
        "source": str(path),
        "sha256": file_hash(path),
        "rows": len(frame),
        "task": "predator_kill_count_prediction",
        "classes": [0, 1, 2, 3, 4],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feedback_delay_seconds": feedback_delay,
        "time_semantics": "2022 metadata explicitly defines launch timestamp; 2023 uses the corresponding start-time interpretation as an assumption",
        "excluded_columns": [
            c
            for c in raw.columns
            if c
            not in [
                player_column,
                "timestamp",
                "match_id",
                "game_duration",
                "hunting_success",
                *CATEGORICAL_FEATURES,
            ]
        ],
        "availability": "Map/avatar treated as known at roster-complete pre-match prediction; no current-match behavioral or result features",
        "limits": [
            "Four prey identities are absent; cannot evaluate reassigned 1v4 matches",
            "No current-match speed, points, duration or prey behavior used as predictive input",
            "total_xp/total_xp_pred span the observation period and are excluded",
            "Observed count is recomputed from completed records, not true lifetime experience",
            "Data licensed CC BY-NC 4.0; raw data are not part of software package",
        ],
    }
    return AsymmetricData(
        features,
        frame.hunting_success.to_numpy(dtype=np.int64),
        frame.match_id.to_numpy(dtype=str),
        frame[player_column].to_numpy(dtype=str),
        frame["_time"].to_numpy(),
        frame["_available"].to_numpy(),
        manifest,
    )
