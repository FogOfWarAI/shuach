# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Matched exact-search implementations: measured allocation and timing scope."""

import argparse
import gc
import hashlib
import json
import platform
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gamematch.matching.constraints import constrained_assignments
from gamematch.matching.search import minimize_assignments


def score(batch):
    return np.asarray([abs(sum(a.teams[0]) - sum(a.teams[1])) for a in batch], dtype=float)


def execute(size, streaming):
    candidates = constrained_assignments(range(2 * size), (size, size))
    if streaming:
        result = minimize_assignments(candidates, score, lambda x: x, batch_size=256)
        return (
            result["assignment"].teams,
            result["cost"],
            result["candidate_index"],
            result["candidate_count"],
        )
    candidates = list(candidates)
    costs = score(candidates)
    index = int(costs.argmin())
    return candidates[index].teams, float(costs[index]), index, len(candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--concurrent-training", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/search-performance.json"))
    args = parser.parse_args()
    rows = []
    for size in [5, 8, 10]:
        reference = execute(size, False)
        if execute(size, True) != reference:
            raise AssertionError("Streaming changed the optimum or tie selection")
        samples = {False: [], True: []}
        memory = {False: [], True: []}
        order_rng = np.random.default_rng(20261114 + size)
        timing_orders, tracing_orders = [], []
        for _ in range(args.replicates):
            order = [bool(value) for value in order_rng.permutation([False, True])]
            timing_orders.append(order)
            for streaming in order:
                gc.collect()
                start = time.perf_counter()
                answer = execute(size, streaming)
                samples[streaming].append(time.perf_counter() - start)
                if answer != reference:
                    raise AssertionError("Non-reproducible search result")
        # Traced runs are a separate phase, so allocation instrumentation never
        # contributes to the recorded wall-clock measurement.
        for _ in range(args.replicates):
            order = [bool(value) for value in order_rng.permutation([False, True])]
            tracing_orders.append(order)
            for streaming in order:
                gc.collect()
                tracemalloc.start()
                execute(size, streaming)
                memory[streaming].append(tracemalloc.get_traced_memory()[1])
                tracemalloc.stop()
        for streaming in [False, True]:
            rows.append(
                {
                    "team_size": size,
                    "candidate_count": reference[3],
                    "implementation": "streaming_256" if streaming else "materialized_reference",
                    "median_seconds": float(np.median(samples[streaming])),
                    "seconds": samples[streaming],
                    "median_traced_peak_bytes": int(np.median(memory[streaming])),
                    "traced_peak_bytes": memory[streaming],
                    "same_assignment_cost_index_count": True,
                    "timing_orders_streaming_flag": timing_orders,
                    "tracing_orders_streaming_flag": tracing_orders,
                }
            )
            print(
                size,
                rows[-1]["implementation"],
                rows[-1]["median_seconds"],
                rows[-1]["median_traced_peak_bytes"],
                flush=True,
            )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in [
                "src/gamematch/matching/constraints.py",
                "src/gamematch/matching/search.py",
            ]
        },
        "python": platform.python_version(),
        "machine": platform.machine(),
        "concurrent_training": args.concurrent_training,
        "order_policy": "Paired randomized implementation order within each size; separate timing and tracing phases; deterministic seed 20261114 + team size",
        "score": "absolute difference of participant-index sums; same exact candidate generator and objective",
        "memory_scope": "tracemalloc peak Python/registered allocation, separate from timing; not process RSS or GPU memory",
        "timing_scope": "CPU end-to-end candidate generation, simple scoring and selection; not a neural inference or production queue benchmark",
        "host_limit": "Single local Mac; background non-training host activity is not controlled",
        "measurements": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
