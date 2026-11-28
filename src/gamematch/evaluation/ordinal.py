# SPDX-License-Identifier: Apache-2.0
"""Ordinal prediction metrics."""

import numpy as np

from .multiclass import multiclass_metrics


def ordinal_metrics(y, probability):
    value = multiclass_metrics(y, probability)
    expected = probability @ np.arange(probability.shape[1])
    value.update(
        expected_count_mae=float(np.abs(expected - y).mean()),
        expected_count_rmse=float(np.sqrt(np.square(expected - y).mean())),
        ranked_probability_score=float(
            np.square(
                np.cumsum(probability, axis=1)[:, :-1]
                - (y[:, None] <= np.arange(probability.shape[1] - 1))
            )
            .sum(1)
            .mean()
        ),
    )
    return value
