# SPDX-License-Identifier: Apache-2.0
"""Regression prediction metrics."""

import numpy as np
from scipy.stats import spearmanr


def regression_metrics(target, prediction):
    y, p = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or not len(y)
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Expected finite, aligned, nonempty regression vectors")
    error = p - y
    variance = np.square(y - y.mean()).sum()
    return {
        "n": len(y),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mean_error": float(error.mean()),
        "r2": float(1 - np.square(error).sum() / variance) if variance else None,
        "spearman": float(spearmanr(y, p).statistic) if np.std(y) > 0 and np.std(p) > 0 else None,
    }
