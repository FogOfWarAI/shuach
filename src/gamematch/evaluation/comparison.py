# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Aligned paired comparisons with explicit sampling units and estimands."""

from pathlib import Path

import numpy as np


def load_predictions(path: Path):
    with np.load(path, allow_pickle=False) as handle:
        data = dict(handle)
    order = np.argsort(data["row_id"])
    if len(np.unique(data["row_id"])) != len(order):
        raise ValueError("Duplicate evaluation row IDs")
    return {k: v[order] if v.ndim and len(v) == len(order) else v for k, v in data.items()}


def per_example_loss(data, metric="log_loss"):
    y = np.asarray(data["y_true"])
    if y.ndim != 1 or not len(y):
        raise ValueError("Expected a nonempty one-dimensional target vector")
    if metric in {"absolute_error", "squared_error"}:
        prediction = np.asarray(data["prediction"], dtype=float)
        target = np.asarray(y, dtype=float)
        if prediction.shape != target.shape or not (
            np.isfinite(prediction).all() and np.isfinite(target).all()
        ):
            raise ValueError("Expected aligned finite regression targets and predictions")
        error = prediction - target
        return np.abs(error) if metric == "absolute_error" else np.square(error)
    if metric != "log_loss":
        raise ValueError("Expected log_loss, absolute_error or squared_error")
    if "p_positive" in data:
        p = np.asarray(data["p_positive"], dtype=float)
        if p.shape != y.shape or not np.isin(y, [0, 1]).all():
            raise ValueError("Expected aligned binary labels and probabilities")
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Expected finite binary probabilities in [0, 1]")
        eps = np.finfo(float).eps
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    p = np.asarray(data["probabilities"], dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or len(p) != len(y):
        raise ValueError("Expected aligned categorical probability rows")
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= p.shape[1])):
        raise ValueError("Invalid categorical labels")
    if (
        not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
        or not np.allclose(p.sum(1), 1, atol=1e-6)
    ):
        raise ValueError("Invalid categorical probability simplex")
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def paired_losses(first_paths, second_paths, metric="log_loss"):
    """Compare mean single-run loss, not the loss of an ensemble probability."""
    if len(first_paths) != len(second_paths) or not first_paths:
        raise ValueError("Pairs of matching seed runs are required")
    reference = None
    differences = []
    identity_fields = (
        "row_id",
        "y_true",
        "group_id",
        "player_id",
        "match_id",
        "time",
        "domain",
        "target_available_at",
    )
    for first, second in zip(first_paths, second_paths):
        a, b = load_predictions(first), load_predictions(second)
        for other in (b, reference):
            if other is None:
                continue
            for field in identity_fields:
                if field in a or field in other:
                    if (
                        field not in a
                        or field not in other
                        or not np.array_equal(a[field], other[field])
                    ):
                        raise ValueError(f"Prediction evaluation identities do not align: {field}")
        reference = a
        differences.append(per_example_loss(a, metric) - per_example_loss(b, metric))
    return np.stack(differences), reference


def paired_cluster_bootstrap(
    differences, clusters, replicates=2000, seed=0, alpha=0.05, metric="log_loss"
):
    """Two-stage paired bootstrap: training seeds, then evaluation clusters.

    Clusters are sampled uniformly with replacement, and their row sums/counts
    are reweighted to target mean per-example loss. Seeds are resampled as paired
    model runs. The method assumes independent clusters; time blocks and player
    IDs address different dependencies and should be reported as sensitivities.
    """
    delta = np.asarray(differences, dtype=float)
    if delta.ndim == 1:
        delta = delta[None]
    if (
        delta.ndim != 2
        or not all(delta.shape)
        or np.asarray(clusters).ndim != 1
        or delta.shape[1] != len(clusters)
        or not np.isfinite(delta).all()
    ):
        raise ValueError("Expected finite paired seed x example differences")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 100
        or not 0 < alpha < 1
    ):
        raise ValueError("Invalid bootstrap settings")
    _, indices = np.unique(np.asarray(clusters), return_inverse=True)
    groups = int(indices.max()) + 1
    if groups < 2:
        raise ValueError("At least two independent resampling units are required")
    counts = np.bincount(indices)
    sums = np.stack([np.bincount(indices, weights=d, minlength=groups) for d in delta])
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for i in range(replicates):
        sampled_seeds = rng.integers(len(delta), size=len(delta))
        sampled_groups = rng.integers(groups, size=groups)
        mean_sums = sums[sampled_seeds].mean(0)
        draws[i] = mean_sums[sampled_groups].sum() / counts[sampled_groups].sum()
    seed_means = delta.mean(1)
    return {
        "delta_first_minus_second": float(delta.mean()),
        "metric": metric,
        "interval_level": 1 - alpha,
        "percentile_interval": np.quantile(draws, [alpha / 2, 1 - alpha / 2]).tolist(),
        "n_examples": delta.shape[1],
        "n_seed_pairs": len(delta),
        "n_clusters": groups,
        "paired_seed_mean_differences": seed_means.tolist(),
        "paired_seed_sample_sd": float(seed_means.std(ddof=1)) if len(delta) > 1 else None,
        "replicates": replicates,
        "bootstrap_seed": seed,
        "estimand": f"mean single-run per-example {metric} difference; lower favors first model",
        "method": "paired training-seed and evaluation-cluster percentile bootstrap",
        "interpretation": "Exploratory marginal interval; no multiple-comparison or causal superiority claim",
    }
