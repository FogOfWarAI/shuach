# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Streaming exact search over caller-defined feasible assignments and costs."""

import time
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


def minimize_assignments_budgeted(
    candidates, score_candidates, objective, *, batch_size=256,
    time_budget=None, max_candidates=None,
):
    """Anytime search: enumerate until a time or candidate budget is reached.

    Unlike ``minimize_assignments``, this returns the best assignment found so
    far instead of insisting on a complete scan. ``status`` distinguishes a
    proven optimum from a budget-limited result, so callers never mistake an
    approximate answer for an exact one.

    Args:
        candidates: Finite iterable of assignment objects.
        score_candidates: Callable receiving an ordered list of candidates.
        objective: Callable converting a batch of scores to finite scalar costs.
        batch_size: Maximum candidates per scorer call.
        time_budget: Optional positive wall-clock limit in seconds.
        max_candidates: Optional positive cap on evaluated candidates.

    Returns:
        Dictionary with assignment, cost, candidate_index, candidate_count,
        batch_size, status, exhausted and elapsed_seconds.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    for key, value in (("time_budget", time_budget), ("max_candidates", max_candidates)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ValueError(f"{key} must be a positive number")
    if isinstance(max_candidates, float) and not max_candidates.is_integer():
        raise ValueError("max_candidates must be a whole number")
    limit = None if max_candidates is None else int(max_candidates)
    start = time.monotonic()
    iterator = iter(candidates)
    best, best_cost, best_index, count = None, float("inf"), -1, 0
    exhausted = True
    while True:
        if time_budget is not None and time.monotonic() - start >= time_budget:
            exhausted = False
            break
        remaining = None if limit is None else limit - count
        if remaining is not None and remaining <= 0:
            exhausted = False
            break
        size = batch_size if remaining is None else min(batch_size, remaining)
        batch = list(islice(iterator, size))
        if not batch:
            break
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
        raise ValueError("No assignments were evaluated within the budget")
    return {
        "assignment": best,
        "cost": best_cost,
        "candidate_index": best_index,
        "candidate_count": count,
        "batch_size": batch_size,
        "status": "optimal_over_enumerated_candidates" if exhausted else "anytime_budget_reached",
        "exhausted": exhausted,
        "elapsed_seconds": time.monotonic() - start,
        "time_budget": time_budget,
        "max_candidates": limit,
    }
