# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Recover and verify history feedback availability times.

``build_histories`` stores ``log1p((prediction_time - available)/86400)`` in
history channel 3. Inverting it lets a validator check the strict
``available < prediction_time`` rule on real data rather than trusting the
builder's prose. NumPy only, so this holds in a base installation.
"""

from __future__ import annotations

import numpy as np

FEEDBACK_AGE_CHANNEL = 3
SECONDS_PER_DAY = 86400.0


def recover_availability(history, prediction_time):
    """Return ``(available, time)`` for a raw, unstandardized history tensor.

    Args:
        history: ``[N, 2, P, T, >=4]`` with a log1p-day feedback-age channel.
        prediction_time: ``[N]`` per-row prediction timestamps in seconds.

    Returns:
        ``available`` shaped like ``[N, 2, P, T]`` and the validated times.
    """
    history = np.asarray(history)
    time = np.asarray(prediction_time, dtype=np.float64)
    if history.ndim != 5 or history.shape[-1] <= FEEDBACK_AGE_CHANNEL:
        raise ValueError("Expected [N,2,P,T,>=4] history with a feedback-age channel")
    if not np.isfinite(history).all():
        raise ValueError("Nonfinite history features at the data boundary")
    if time.ndim != 1 or time.shape[0] != history.shape[0] or not np.isfinite(time).all():
        raise ValueError("Prediction times must be finite and row-aligned with history")
    age_days = np.expm1(history[..., FEEDBACK_AGE_CHANNEL].astype(np.float64))
    return time[:, None, None, None] - age_days * SECONDS_PER_DAY, time


def assert_strictly_earlier(available, prediction_time, valid):
    """Raise on the first observed step whose feedback is not strictly earlier.

    Masked positions are ignored: their channel values are padding, not
    evidence. Returns the number of observed steps that passed the check.
    """
    available = np.asarray(available, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    time = np.asarray(prediction_time, dtype=np.float64)
    if available.shape != valid.shape or valid.ndim != 4 or time.shape[0] != valid.shape[0]:
        raise ValueError("Availability, mask and prediction times are not aligned")
    prediction = np.broadcast_to(time[:, None, None, None], valid.shape)
    breached = valid & ~(available < prediction)
    if breached.any():
        row, side, player, step = (int(v) for v in np.argwhere(breached)[0])
        raise ValueError(
            f"History feedback at {available[row, side, player, step]:.3f} is not strictly "
            f"earlier than prediction_time {time[row]:.3f} "
            f"(row={row} side={side} player={player} step={step})"
        )
    return int(valid.sum())
