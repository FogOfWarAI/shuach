# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Exact two-team partitioning for small fixed lobbies, without role constraints."""

from itertools import combinations
from numbers import Integral

import numpy as np


def partitions(players, team_size: int, exchangeable_sides: bool = False):
    """Yield complete, equal-size teams in input combination order.

    Args:
        players: Finite iterable of distinct, hashable player IDs.
        team_size: Positive integer number of players per side.
        exchangeable_sides: Keep the first input player on the left when
            swapping sides preserves the caller's objective.

    Yields:
        A pair of player tuples, each in the original relative order.

    Raises:
        ValueError: On iteration, if the size, player count or uniqueness is invalid.
    """
    players = tuple(players)
    if isinstance(team_size, bool) or not isinstance(team_size, Integral):
        raise ValueError("team_size must be a positive integer")
    if team_size < 1 or len(players) != 2 * team_size:
        raise ValueError("Expected exactly two equally sized complete teams")
    if len(set(players)) != len(players):
        raise ValueError("A player may occur only once in a lobby")
    for indices in combinations(range(len(players)), team_size):
        if exchangeable_sides and 0 not in indices:
            continue
        chosen = set(indices)
        yield (
            tuple(players[i] for i in indices),
            tuple(p for i, p in enumerate(players) if i not in chosen),
        )


def exact_balance(players, team_size: int, predict_win, exchangeable_sides: bool = False):
    """Return the first partition whose predicted probability is closest to 0.5.

    Args:
        players: Finite iterable of distinct, hashable player IDs.
        team_size: Positive size of each of the two teams.
        predict_win: Callable receiving one list of all team pairs in order.
            It must return a vector of left-team win probabilities in [0, 1].
        exchangeable_sides: Keep one orientation per pair if the absolute
            imbalance is invariant under a side swap.

    Returns:
        Dictionary with teams, predicted_win, predicted_imbalance, zero-based
        candidate_index, candidate_count and status.

    Raises:
        ValueError: If the lobby is invalid or the scorer returns the wrong
            shape, a nonfinite value or a probability outside [0, 1].
    """
    candidates = list(partitions(players, team_size, exchangeable_sides))
    scores = np.asarray(predict_win(candidates))
    if np.iscomplexobj(scores):
        raise ValueError("Scorer must return real win probabilities")
    p = np.asarray(scores, dtype=np.float64)
    if p.shape != (len(candidates),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Scorer must return one binary win probability per complete candidate")
    costs = np.abs(p - 0.5)
    best = int(np.argmin(costs))
    return {
        "teams": candidates[best],
        "predicted_win": float(p[best]),
        "predicted_imbalance": float(costs[best]),
        "candidate_count": len(candidates),
        "status": "optimal_over_enumerated_candidates",
        "candidate_index": best,
    }
