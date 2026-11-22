# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Typed, delayed-feedback team events and a conservative Overwatch adapter."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .heroes import file_hash


@dataclass(frozen=True)
class MatchEvent:
    """Represent ordered teams at prediction time and a separately timed outcome.

    The left side's outcome is encoded as 0 for loss, 1 for draw and 2 for win.
    Feedback becomes eligible for history updates strictly after available_at.
    """
    row_id: str
    time: float
    available_at: float
    teams: tuple[tuple[str, ...], tuple[str, ...]]
    outcome: int  # 0: left loss, 1: draw, 2: left win
    domain: int
    duration: float | None = None

    def __post_init__(self):
        if len(self.teams) != 2:
            raise ValueError("Exactly two teams are required")
        people = self.teams[0] + self.teams[1]
        if not self.teams[0] or not self.teams[1] or len(set(people)) != len(people):
            raise ValueError("Complete, disjoint teams with unique participants are required")
        if self.outcome not in (0, 1, 2):
            raise ValueError("Outcome must be loss, draw or win")
        if not np.isfinite([self.time, self.available_at]).all() or self.available_at < self.time:
            raise ValueError("Feedback cannot precede the prediction timestamp")
        if self.duration is not None and (not np.isfinite(self.duration) or self.duration <= 0):
            raise ValueError("Observed duration must be positive")


def _json_records(path):
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            yield line_no, json.loads(line)


def load_overwatch(path: Path, feedback_delay: float = 0):
    """Use reported time as start proxy and time+duration+delay for feedback.

    The export does not document an authoritative result-availability timestamp.
    This assumption is persisted in the manifest; extra-delay sensitivity runs
    must not be represented as a verification of the original timestamp meaning.
    """
    if not np.isfinite(feedback_delay) or feedback_delay < 0:
        raise ValueError("Feedback delay must be finite and nonnegative")
    counts = dict(
        raw=0,
        exact_duplicates=0,
        incomplete_rosters=0,
        repeated_players=0,
        invalid_time=0,
        invalid_outcome=0,
        invalid_domain=0,
    )
    seen, events = set(), []
    for line_no, item in _json_records(path):
        counts["raw"] += 1
        fingerprint = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).digest()
        if fingerprint in seen:
            counts["exact_duplicates"] += 1
            continue
        seen.add(fingerprint)
        teams = tuple(tuple(item.get("teams", {}).get(s, [])) for s in ("blue", "red"))
        players = teams[0] + teams[1]
        if any(len(t) != 6 for t in teams) or any(
            not isinstance(p, str) or not p.strip() for p in players
        ):
            counts["incomplete_rosters"] += 1
            continue
        if len(set(players)) != 12:
            counts["repeated_players"] += 1
            continue
        if item.get("result") not in {"LOSS", "DRAW", "WIN"}:
            counts["invalid_outcome"] += 1
            continue
        if item.get("competitive") not in (False, True):
            counts["invalid_domain"] += 1
            continue
        time, duration = item.get("time"), item.get("duration")
        if (
            not isinstance(time, (int, float))
            or not isinstance(duration, (int, float))
            or not np.isfinite([time, duration]).all()
            or duration <= 0
        ):
            counts["invalid_time"] += 1
            continue
        events.append(
            MatchEvent(
                f"overwatch:{line_no}",
                float(time),
                float(time + duration + feedback_delay),
                teams,
                {"LOSS": 0, "DRAW": 1, "WIN": 2}[item["result"]],
                int(item["competitive"]),
                float(duration),
            )
        )
    events.sort(key=lambda e: (e.time, e.row_id))
    counts["retained"] = len(events)
    manifest = {
        "dataset": "overwatch",
        "source": str(path),
        "sha256": file_hash(path),
        "counts": counts,
        "outcome_classes": ["blue_loss", "draw", "blue_win"],
        "outcome_evidence": "https://github.com/vivekjoshy/openskill.py/blob/main/benchmark/benchmark.py",
        "time_assumption": "reported time is a start proxy; feedback at time+duration+delay; exact original timestamp semantics not independently verified",
        "feedback_delay_seconds": feedback_delay,
        "entity_kind": "anonymized_player_token",
        "limitations": [
            "Token identity is not independently verified account identity",
            "Observation coverage is incomplete; first observed does not mean novice",
            "No queue, party or role-preference observations",
            "Blue-side sampling may reflect recorder-side selection",
        ],
    }
    return events, manifest


def temporal_boundaries(events, train_fraction=0.7, valid_fraction=0.15):
    if not (0 < train_fraction < train_fraction + valid_fraction < 1):
        raise ValueError("Temporal split fractions must leave three nonempty partitions")
    if len(events) < 10:
        raise ValueError("Too few events for temporal evaluation")
    ordered = np.sort([e.time for e in events])
    bounds = (
        float(ordered[int(len(ordered) * train_fraction)]),
        float(ordered[int(len(ordered) * (train_fraction + valid_fraction))]),
    )
    if bounds[0] >= bounds[1]:
        raise ValueError("Timestamp ties prevent distinct temporal partitions")
    return bounds
