# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Compact pre-event histories constructed only from delivered outcomes."""

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

HISTORY_FEATURES = [
    "previous_player_result",
    "log1p_previous_duration_minutes",
    "previous_competitive",
    "log1p_days_since_feedback",
    "same_domain_as_target",
]
STATIC_FEATURES = [
    f"{scope}_{feature}"
    for scope in ["global", "noncompetitive", "competitive"]
    for feature in [
        "log1p_completed",
        "smoothed_win_fraction",
        "smoothed_draw_fraction",
        "log1p_days_since_feedback",
    ]
] + ["target_competitive"]


@dataclass
class HistoryData:
    """Aligned event rows with history, static summaries, masks and outcome timing."""
    history: np.ndarray
    static: np.ndarray
    valid: np.ndarray
    outcome: np.ndarray
    domain: np.ndarray
    row_ids: np.ndarray
    time: np.ndarray
    available_at: np.ndarray

    def save(self, path):
        """Write the row-aligned arrays to a compressed NumPy archive."""
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        """Read an archive written by save using NumPy's non-pickle format."""
        with np.load(path, allow_pickle=False) as values:
            return cls(**dict(values))


def build_histories(events, history_length=10, same_domain_only=False):
    """Build features as they would have been observed at each prediction time.

    Args:
        events: Nonempty time-ordered MatchEvent sequence with fixed, equal
            team sizes and binary domain IDs.
        history_length: Maximum number of completed steps to retain per player.
        same_domain_only: Restrict sequences to the current domain. Static
            summaries still include both global and domain-specific statistics.

    Returns:
        HistoryData containing right-aligned history [N, 2, P, T, 5], static
        features [N, 2, P, 13] and validity flags [N, 2, P, T] for N events.
        Feedback tied with prediction time is excluded.
    """
    if history_length < 1 or not events:
        raise ValueError("Nonempty events and positive history length required")
    if any(event.domain not in (0, 1) for event in events):
        raise ValueError("This public history adapter supports binary competitive domains only")
    team_size = len(events[0].teams[0])
    shape = (len(events), 2, team_size)
    history = np.zeros((*shape, history_length, len(HISTORY_FEATURES)), dtype=np.float32)
    static = np.zeros((*shape, len(STATIC_FEATURES)), dtype=np.float32)
    valid = np.zeros((*shape, history_length), dtype=bool)
    # Values are (completed count, wins + half draws, draws, last available time).
    stats = defaultdict(lambda: [0, 0.0, 0, None])
    recent = defaultdict(lambda: deque(maxlen=history_length))
    pending, last_time = [], -float("inf")

    def update(event):
        for side, team in enumerate(event.teams):
            result = event.outcome / 2 if side == 0 else 1 - event.outcome / 2
            for player in team:
                for scope in (-1, event.domain):
                    s = stats[(player, scope)]
                    s[0] += 1
                    s[1] += result
                    s[2] += event.outcome == 1
                    s[3] = event.available_at
                key = (player, event.domain) if same_domain_only else (player, -1)
                duration = (
                    event.duration
                    if event.duration is not None
                    else event.available_at - event.time
                )
                recent[key].append(
                    (result, np.log1p(duration / 60), event.domain, event.available_at)
                )

    for index, event in enumerate(events):
        if event.time < last_time or any(len(team) != team_size for team in event.teams):
            raise ValueError("Consistent team sizes and chronological events are required")
        last_time = event.time
        while pending and pending[0][0] < event.time:
            update(heapq.heappop(pending)[2])
        for side, team in enumerate(event.teams):
            for position, player in enumerate(team):
                key = (player, event.domain) if same_domain_only else (player, -1)
                past = recent[key]
                # Right-align so the most recent completed event has the last
                # position; all absent history positions are explicitly masked.
                offset = history_length - len(past)
                for step, (result, duration, domain, available) in enumerate(past, offset):
                    history[index, side, position, step] = [
                        result,
                        duration,
                        domain,
                        np.log1p((event.time - available) / 86400),
                        float(domain == event.domain),
                    ]
                    valid[index, side, position, step] = True
                for j, scope in enumerate((-1, 0, 1)):
                    n, wins, draws, available = stats[(player, scope)]
                    static[index, side, position, j * 4 : (j + 1) * 4] = [
                        np.log1p(n),
                        (wins + 1) / (n + 2),
                        (draws + 0.5) / (n + 1),
                        np.log1p((event.time - available) / 86400) if available is not None else 0,
                    ]
                static[index, side, position, -1] = event.domain
        heapq.heappush(pending, (event.available_at, event.row_id, event))
    return HistoryData(
        history,
        static,
        valid,
        np.asarray([e.outcome for e in events], dtype=np.int64),
        np.asarray([e.domain for e in events], dtype=np.int64),
        np.asarray([e.row_id for e in events]),
        np.asarray([e.time for e in events]),
        np.asarray([e.available_at for e in events]),
    )


def history_split_masks(data, train_fraction=0.7, valid_fraction=0.15):
    """Return train/valid/test masks and two time boundaries, purging late labels."""
    if not (0 < train_fraction < train_fraction + valid_fraction < 1) or len(data.time) < 10:
        raise ValueError("Valid temporal fractions and at least ten events are required")
    ordered = np.sort(data.time)
    bounds = (
        ordered[int(len(ordered) * train_fraction)],
        ordered[int(len(ordered) * (train_fraction + valid_fraction))],
    )
    if bounds[0] >= bounds[1]:
        raise ValueError("Timestamp ties collapse temporal partitions")
    masks = {
        "train": (data.time < bounds[0]) & (data.available_at < bounds[0]),
        "valid": (data.time >= bounds[0])
        & (data.time < bounds[1])
        & (data.available_at < bounds[1]),
        "test": data.time >= bounds[1],
    }
    return masks, [float(b) for b in bounds]


def standardize_histories(data, train):
    """Mutate arrays using statistics fit on training features, never on test."""
    values = data.history[train][data.valid[train]]
    if len(values) == 0:
        raise ValueError("Training records have no completed player history")
    hmean, hscale = values.mean(0), values.std(0).clip(1e-6)
    svalues = data.static[train].reshape(-1, data.static.shape[-1])
    smean, sscale = svalues.mean(0), svalues.std(0).clip(1e-6)
    data.history = (data.history - hmean) / hscale
    data.history[~data.valid] = 0
    data.static = (data.static - smean) / sscale
    return {
        "history_mean": hmean.tolist(),
        "history_scale": hscale.tolist(),
        "static_mean": smean.tolist(),
        "static_scale": sscale.tolist(),
        "fitted_partition": "train",
    }
