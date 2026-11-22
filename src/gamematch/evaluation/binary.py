# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def binary_metrics(labels, probabilities, bins: int = 15):
    """Evaluate one positive-class probability for each binary label.

    Args:
        labels: Nonempty one-dimensional array of zeros and ones.
        probabilities: Aligned, finite positive-class probabilities in [0, 1].
        bins: Number of equal-width probability bins for calibration error.

    Returns:
        A dictionary including log loss, Brier score, probability MAE and RMSE,
        accuracy at 0.5, ECE and AUC. AUC is None for a single-class sample.
    """
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("Calibration bin count must be a positive integer")
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 1 or len(p) == 0:
        raise ValueError("Expected equal non-empty one-dimensional label/probability arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Binary evaluation cannot silently discard draws or multiclass outcomes")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected finite probabilities in [0, 1], not logits or skill scores")
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(
        float((bucket == b).mean())
        * abs(float(p[bucket == b].mean()) - float(y[bucket == b].mean()))
        for b in range(bins)
        if (bucket == b).any()
    )
    return {
        "n": len(y),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "probability_mae": float(np.abs(p - y).mean()),
        "probability_rmse": float(np.sqrt(np.square(p - y).mean())),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        f"ece_{bins}_equal_width": ece,
    }
