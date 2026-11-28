# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Task-defined MAE/RMSE from saved predictions; no model refitting or selection."""

import numpy as np

from gamematch.evaluation.binary import binary_metrics
from gamematch.evaluation.multiclass import multiclass_metrics
from gamematch.evaluation.regression import regression_metrics


def extended_metrics(data):
    y = np.asarray(data["y_true"])
    if "prediction" in data:
        validated = regression_metrics(y, data["prediction"])
        return {key: validated[key] for key in ["mae", "rmse"]}
    if "p_positive" in data:
        validated = binary_metrics(y, data["p_positive"])
        return {key: validated[key] for key in ["probability_mae", "probability_rmse"]}
    probability = np.asarray(data["probabilities"], dtype=float)
    validated = multiclass_metrics(y, probability)
    classes = probability.shape[1]
    result = {key: validated[key] for key in ["probability_mae", "probability_rmse"]}
    if classes == 3:
        # The framework explicitly encodes [loss, draw, win], not arbitrary classes.
        scores = np.array([0.0, 0.5, 1.0])
        prefix = "expected_result_score"
        decisive = y != 1
        if decisive.any():
            q = probability[decisive]
            denominator = q[:, 0] + q[:, 2]
            win = np.divide(q[:, 2], denominator, out=np.full(len(q), 0.5), where=denominator > 0)
            target = (y[decisive] == 2).astype(int)
            metrics = binary_metrics(target, win)
            result.update(
                {"decisive_" + k: v for k, v in metrics.items() if isinstance(v, (float, int))}
            )
            result["decisive_probability_mae"] = float(np.abs(win - target).mean())
            result["decisive_probability_rmse"] = float(np.sqrt(np.square(win - target).mean()))
    elif classes == 5:
        scores = np.arange(5, dtype=float)
        prefix = "expected_count"
    else:
        return result
    residual = probability @ scores - scores[y.astype(int)]
    result[prefix + "_mae"] = float(np.abs(residual).mean())
    result[prefix + "_rmse"] = float(np.sqrt(np.square(residual).mean()))
    return result
