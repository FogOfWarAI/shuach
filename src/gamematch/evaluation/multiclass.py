# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Proper scoring rules for declared categorical outcome probabilities."""

import numpy as np


def multiclass_metrics(labels, probabilities):
    """Evaluate predictions with one column per ordered outcome.

    Args:
        labels: Integer class indices [N].
        probabilities: Finite [N, C] probability matrix with each row summing to one.

    Returns:
        A dictionary including log loss, accuracy and class counts. Brier sums
        squared errors across classes per row. Probability MAE and RMSE average
        across both rows and class columns.
    """
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 2 or p.shape[1] < 2 or len(y) != len(p) or not len(y):
        raise ValueError("Expected nonempty aligned labels and class probabilities")
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= p.shape[1])):
        raise ValueError("Invalid categorical labels")
    if (
        not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
        or not np.allclose(p.sum(1), 1, atol=1e-6)
    ):
        raise ValueError("Invalid categorical probability simplex")
    onehot = np.eye(p.shape[1])[y]
    return {
        "n": len(y),
        "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "accuracy": float((p.argmax(1) == y).mean()),
        "brier_multiclass_sum": float(np.square(p - onehot).sum(1).mean()),
        "probability_mae": float(np.abs(p - onehot).mean()),
        "probability_rmse": float(np.sqrt(np.square(p - onehot).mean())),
        "class_counts": np.bincount(y, minlength=p.shape[1]).tolist(),
    }
