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


def local_swap_balance(
    players, team_size, predict_win, *, max_swaps=None, time_budget=None,
    seed=0, tolerance=1e-12,
):
    """Approximate balanced partitioning by pairwise cross-side swaps.

    ``exact_balance`` enumerates C(2n, n) partitions (184,756 at 10v10, and
    far more with role slots). This variant starts from a randomized split and
    evaluates all n^2 cross-side swap neighbours in ONE batched scorer call per
    round, stopping at a local optimum or a declared budget. The result is
    explicitly labelled approximate; it never claims an enumerated optimum.

    Returns the same keys as ``exact_balance`` plus swaps, elapsed_seconds and
    method, so callers can compare both solvers without branching on shape.
    """
    import time

    players = tuple(players)
    if isinstance(team_size, bool) or not isinstance(team_size, Integral) or team_size < 1:
        raise ValueError("team_size must be a positive integer")
    if len(players) != 2 * team_size:
        raise ValueError("Expected exactly two equally sized complete teams")
    if len(set(players)) != len(players):
        raise ValueError("A player may occur only once in a lobby")
    for key, value in (("max_swaps", max_swaps), ("time_budget", time_budget)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ValueError(f"{key} must be a positive number")
    if isinstance(max_swaps, float) and not max_swaps.is_integer():
        raise ValueError("max_swaps must be a whole number")
    swaps_limit = None if max_swaps is None else int(max_swaps)

    def score_pairs(pairs):
        candidates = [
            (tuple(players[i] for i in a), tuple(players[i] for i in b)) for a, b in pairs
        ]
        scores = np.asarray(predict_win(candidates))
        if np.iscomplexobj(scores):
            raise ValueError("Scorer must return real win probabilities")
        p = np.asarray(scores, dtype=np.float64)
        if p.shape != (len(candidates),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Scorer must return one binary win probability per candidate")
        return np.abs(p - 0.5), p

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(players)).tolist()
    left, right = order[:team_size], order[team_size:]
    initial_cost, initial_p = score_pairs([(left, right)])
    current_cost, current_p = float(initial_cost[0]), float(initial_p[0])
    start = time.monotonic()
    evaluated, swaps, status = 1, 0, "local_swap_converged"
    while True:
        if time_budget is not None and time.monotonic() - start >= time_budget:
            status = "time_budget_reached"
            break
        if swaps_limit is not None and swaps >= swaps_limit:
            status = "swap_budget_reached"
            break
        neighbours = []
        for i in range(team_size):
            for j in range(team_size):
                a, b = list(left), list(right)
                a[i], b[j] = b[j], a[i]
                neighbours.append((a, b))
        costs, probabilities = score_pairs(neighbours)
        evaluated += len(neighbours)
        best = int(np.argmin(costs))
        if costs[best] >= current_cost - tolerance:
            break
        left, right = neighbours[best]
        current_cost, current_p = float(costs[best]), float(probabilities[best])
        swaps += 1
    return {
        "teams": (tuple(players[i] for i in left), tuple(players[i] for i in right)),
        "predicted_win": current_p,
        "predicted_imbalance": current_cost,
        "candidate_count": evaluated,
        "status": status,
        "swaps": swaps,
        "elapsed_seconds": time.monotonic() - start,
        "time_budget": time_budget,
        "max_swaps": swaps_limit,
        "method": "pairwise cross-side swap local search; approximate, not an enumerated optimum",
    }
