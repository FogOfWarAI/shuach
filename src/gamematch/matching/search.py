# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Streaming exact search over caller-defined feasible assignments and costs."""

from itertools import islice

import numpy as np


def minimize_assignments(candidates, score_candidates, objective, batch_size=256):
    """Exhaust a finite iterable and return its first minimum-cost candidate.

    Args:
        candidates: Iterable of objects, typically from constrained_assignments.
        score_candidates: Callable receiving an ordered list of up to batch_size
            candidates. Its result passes directly to objective.
        objective: Callable converting a batch of scores to a one-dimensional
            vector of finite scalar costs in the same candidate order.
        batch_size: Positive integer maximum number of candidates per scorer call.

    Returns:
        Dictionary with the original assignment object, cost, zero-based
        candidate_index, total candidate_count, batch_size and status.
        Equal costs select the earliest candidate across all batches.

    Raises:
        ValueError: For an empty iterable, invalid batch size, wrong cost
            vector shape or nonfinite cost.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    iterator = iter(candidates)
    best, best_cost, best_index, count = None, float("inf"), -1, 0
    while batch := list(islice(iterator, batch_size)):
        values = np.asarray(objective(score_candidates(batch)))
        if np.iscomplexobj(values):
            raise ValueError("Objective must return real costs")
        costs = np.asarray(values, dtype=np.float64)
        if costs.shape != (len(batch),) or not np.isfinite(costs).all():
            raise ValueError("Objective must return one finite scalar cost per candidate")
        index = int(np.argmin(costs))
        if costs[index] < best_cost:
            best, best_cost, best_index = batch[index], float(costs[index]), count + index
        count += len(batch)
    if count == 0:
        raise ValueError("No feasible assignments were generated")
    return {
        "assignment": best,
        "cost": best_cost,
        "candidate_index": best_index,
        "candidate_count": count,
        "status": "optimal_over_enumerated_candidates",
        "batch_size": batch_size,
    }
