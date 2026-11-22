# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Outcome-history enrichment, with a strict delivered-feedback boundary.

These are derived observations, not fabricated KDA/role/party measurements.
Elo is a declared fixed feature generator (1500 initial mean, K=32); its
expectation is captured before each match and its update occurs on delivery.
"""

import heapq
from collections import defaultdict, deque

import numpy as np

from .history import HISTORY_FEATURES, STATIC_FEATURES, HistoryData, build_histories

WINDOWS = (3, 5, 10, 20)
EXTRA_HISTORY_FEATURES = [
    "previous_pre_match_team_elo_centered",
    "previous_pre_match_opponent_elo_centered",
    "previous_result_minus_pre_match_expectation",
]
EXTRA_STATIC_FEATURES = [
    f"last{window}_{field}"
    for window in WINDOWS
    for field in ("log1p_count", "mean_result", "result_sd", "mean_log_duration")
] + [
    "ewma_result_alpha02",
    "signed_log1p_streak",
    "recent3_minus_previous7",
    "smoothed_result_entropy_last20",
    "elo_centered",
    "team_elo_centered",
    "opponent_elo_centered",
    "team_expected_score",
    "log1p_elo_updates",
    "teammate_seen_fraction",
    "teammate_mean_log1p_shared",
    "opponent_seen_fraction",
    "opponent_mean_log1p_encounters",
    "log1p_matches_1d",
    "log1p_matches_7d",
    "log1p_matches_30d",
    "has_completed_history",
]
RICH_HISTORY_FEATURES = HISTORY_FEATURES + EXTRA_HISTORY_FEATURES
RICH_STATIC_FEATURES = STATIC_FEATURES + EXTRA_STATIC_FEATURES


def build_rich_histories(events, history_length=10):
    """Return the HistoryData contract with 8 sequence and 46 static fields.

    Only feedback with available_at < target.time contributes. All aggregates
    use observed histories; missing history has explicit counts/indicators.
    Within-team order affects only tensor position, not aggregate values.
    """
    if len({e.row_id for e in events}) != len(events):
        raise ValueError("Unique event row IDs are required")
    base = build_histories(events, history_length=history_length)
    shape = base.static.shape[:-1]
    extra = np.zeros((*shape, len(EXTRA_STATIC_FEATURES)), dtype=np.float32)
    sequence = np.zeros((*base.valid.shape, len(EXTRA_HISTORY_FEATURES)), dtype=np.float32)
    recent = defaultdict(lambda: deque(maxlen=max(history_length, 20)))
    activity = defaultdict(deque)
    rating = defaultdict(lambda: 1500.0)
    updates = defaultdict(int)
    ewma = defaultdict(lambda: 0.5)
    streak = defaultdict(int)
    teammates, opponents = defaultdict(int), defaultdict(int)
    pending = []

    def pair(first, second):
        return tuple(sorted((first, second)))

    def expected(first, second):
        return 1 / (1 + 10 ** np.clip((second - first) / 400, -12, 12))

    def deliver(event, means):
        for side, team in enumerate(event.teams):
            score = event.outcome / 2 if side == 0 else 1 - event.outcome / 2
            expectation = expected(means[side], means[1 - side])
            duration = event.duration
            if duration is None:
                duration = event.available_at - event.time
            for player in team:
                recent[player].append(
                    (
                        score,
                        np.log1p(duration / 60),
                        means[side],
                        means[1 - side],
                        score - expectation,
                    )
                )
                activity[player].append(event.available_at)
                rating[player] += 32 * (score - expectation)
                updates[player] += 1
                ewma[player] = 0.8 * ewma[player] + 0.2 * score
                sign = int(np.sign(score - 0.5))
                streak[player] = (
                    streak[player] + sign if sign and np.sign(streak[player]) == sign else sign
                )
            for i, first in enumerate(team):
                for second in team[i + 1 :]:
                    teammates[pair(first, second)] += 1
        for first in event.teams[0]:
            for second in event.teams[1]:
                opponents[pair(first, second)] += 1

    for index, event in enumerate(events):
        while pending and pending[0][0] < event.time:
            _, _, past, means = heapq.heappop(pending)
            deliver(past, means)
        means = [float(np.mean([rating[p] for p in team])) for team in event.teams]
        for side, team in enumerate(event.teams):
            expectation = expected(means[side], means[1 - side])
            for position, player in enumerate(team):
                past = list(recent[player])
                values = []
                for window in WINDOWS:
                    selected = past[-window:]
                    scores = [r[0] for r in selected]
                    values.extend(
                        [
                            np.log1p(len(selected)),
                            np.mean(scores) if scores else 0.5,
                            np.std(scores) if scores else 0.0,
                            np.mean([r[1] for r in selected]) if selected else 0.0,
                        ]
                    )
                recent3, previous7 = past[-3:], past[-10:-3]
                trend = (
                    np.mean([r[0] for r in recent3]) - np.mean([r[0] for r in previous7])
                    if recent3 and previous7
                    else 0.0
                )
                frequencies = np.bincount([int(2 * r[0]) for r in past[-20:]], minlength=3) + 1
                frequencies = frequencies / frequencies.sum()
                shared = [
                    teammates.get(pair(player, other), 0) for other in team if other != player
                ]
                faced = [opponents.get(pair(player, other), 0) for other in event.teams[1 - side]]
                times = activity[player]
                while times and times[0] < event.time - 30 * 86400:
                    times.popleft()
                counts = [sum(t >= event.time - days * 86400 for t in times) for days in (1, 7, 30)]
                values.extend(
                    [
                        ewma[player],
                        np.sign(streak[player]) * np.log1p(abs(streak[player])),
                        trend,
                        -np.sum(frequencies * np.log(frequencies)),
                        (rating[player] - 1500) / 400,
                        (means[side] - 1500) / 400,
                        (means[1 - side] - 1500) / 400,
                        expectation,
                        np.log1p(updates[player]),
                        np.mean(np.asarray(shared) > 0) if shared else 0.0,
                        np.mean(np.log1p(shared)) if shared else 0.0,
                        np.mean(np.asarray(faced) > 0),
                        np.mean(np.log1p(faced)),
                        *np.log1p(counts),
                        float(updates[player] > 0),
                    ]
                )
                extra[index, side, position] = values
                observed = past[-history_length:]
                for step, row in enumerate(observed, history_length - len(observed)):
                    sequence[index, side, position, step] = [
                        (row[2] - 1500) / 400,
                        (row[3] - 1500) / 400,
                        row[4],
                    ]
        heapq.heappush(pending, (event.available_at, event.row_id, event, means))
    return HistoryData(
        np.concatenate([base.history, sequence], axis=-1),
        np.concatenate([base.static, extra], axis=-1),
        base.valid,
        base.outcome,
        base.domain,
        base.row_ids,
        base.time,
        base.available_at,
    )
