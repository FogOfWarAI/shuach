# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Paired bootstrap intervals for synthetic decision regret.

Kept free of PyTorch so the statistical contract can be tested from a base
installation, independently of the runner that produces the artifacts.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .comparison import paired_cluster_bootstrap


def build_decision_confidence(output, replicates=1000, seed=0):
    """Aggregate saved per-seed regret into paired bootstrap intervals.

    Args:
        output: Run directory containing ``<world>/<model>/decisions.npz``.
        replicates: Paired bootstrap draws, at least 100.
        seed: RNG seed for reproducible percentile intervals.

    Returns:
        A JSON-serializable payload with single-group intervals for every
        (mechanism, model, variant, solver) cell and exact-minus-random
        contrasts. Lobbies are evaluation clusters; hidden worlds are the
        paired resampling units.
    """
    output = Path(output)
    grouped = defaultdict(dict)
    for path in sorted(output.glob("*/*/decisions.npz")):
        parts = path.relative_to(output).parts
        if len(parts) != 3 or parts[2] != "decisions.npz" or "-seed-" not in parts[0]:
            continue
        kind, seed_text = parts[0].rsplit("-seed-", 1)
        try:
            world_seed = int(seed_text)
        except ValueError:
            continue
        with np.load(path, allow_pickle=False) as handle:
            for key in handle.files:
                if not key.endswith("_regret"):
                    continue
                variant, solver = key[: -len("_regret")].rsplit("_", 1)
                grouped[(kind, parts[1], variant, solver)][world_seed] = np.asarray(
                    handle[key], dtype=np.float64
                )

    def stacked(by_seed):
        seeds = sorted(by_seed)
        if len(seeds) < 2:
            return None, seeds
        lengths = {len(by_seed[s]) for s in seeds}
        if len(lengths) != 1 or min(lengths) < 2:
            return None, seeds
        return np.stack([by_seed[s] for s in seeds]), seeds

    intervals = []
    for identity, by_seed in sorted(grouped.items()):
        delta, seeds = stacked(by_seed)
        entry = {"identity": list(identity), "n_world_seeds": len(seeds), "world_seeds": seeds}
        if delta is None:
            entry["status"] = "insufficient_or_unaligned_seed_replicates"
            intervals.append(entry)
            continue
        entry.update(
            paired_cluster_bootstrap(
                delta, np.arange(delta.shape[1]), replicates=replicates, seed=seed,
                metric="decision_regret",
            )
        )
        entry["status"] = "estimated"
        entry["single_group_interval"] = True
        entry["estimand"] = (
            "mean decision regret; the delta itself is regret, so this is a "
            "single-group percentile interval compared against zero"
        )
        intervals.append(entry)

    contrasts = []
    for kind, model, variant in sorted({(k, m, v) for k, m, v, _ in grouped}):
        exact = grouped.get((kind, model, variant, "exact"), {})
        random = grouped.get((kind, model, variant, "random"), {})
        deltas = []
        for world_seed in sorted(set(exact) & set(random)):
            a, b = exact[world_seed], random[world_seed]
            if len(a) != len(b):
                deltas = []
                break
            deltas.append(a - b)
        if len(deltas) < 2:
            continue
        delta = np.stack(deltas)
        contrasts.append(
            {
                "identity": [kind, model, variant],
                "comparison": "exact_minus_random",
                "n_world_seeds": len(deltas),
                "status": "estimated",
                **paired_cluster_bootstrap(
                    delta, np.arange(delta.shape[1]), replicates=replicates, seed=seed,
                    metric="decision_regret",
                ),
            }
        )
    return {
        "status": "completed",
        "method": "paired training-seed / lobby-cluster percentile bootstrap (evaluation.comparison)",
        "interval_level": 0.95,
        "replicates": replicates,
        "bootstrap_seed": seed,
        "limitations": [
            "Synthetic known-mechanism evidence; not an online human trial",
            "Oracle regret uses the same hidden generator that produced the labels",
            "Exploratory marginal intervals; no multiple-comparison correction",
        ],
        "single_group_intervals": intervals,
        "solver_contrasts": contrasts,
    }
