# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Validation-only positive-slope binary logit calibration.

Temperature mode preserves the zero-logit decision boundary. Affine mode also
fits an intercept. Both preserve ranking except at floating-point saturation.
The caller must provide validation data, never the final evaluation labels.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _logits(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Expected a nonempty finite logit vector")
    return result


def apply_binary_calibration(logits, parameters):
    """Apply saved slope/intercept to logits without labels or fitting."""
    z = _logits(logits)
    slope, intercept = float(parameters["slope"]), float(parameters["intercept"])
    if not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise ValueError("Calibration requires a finite positive slope and finite intercept")
    return expit(slope * z + intercept)


def fit_binary_calibration(logits, validation_labels, mode="temperature"):
    """Fit convex held-out NLL with bounded positive slope and optional intercept."""
    z = _logits(logits)
    y = np.asarray(validation_labels)
    if y.shape != z.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Validation labels must align and contain both binary classes")
    if mode not in {"temperature", "affine"}:
        raise ValueError("Choose temperature or affine calibration")
    design = z[:, None] if mode == "temperature" else np.column_stack([z, np.ones(len(z))])
    initial = np.array([1.0] if mode == "temperature" else [1.0, 0.0])
    bounds = [(0.001, 1000.0)] + ([(-20.0, 20.0)] if mode == "affine" else [])

    def objective(parameters):
        value = design @ parameters
        loss = np.mean(np.logaddexp(0, value) - y * value)
        gradient = design.T @ (expit(value) - y) / len(y)
        return loss, gradient

    result = minimize(
        objective,
        initial,
        jac=True,
        bounds=bounds,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Calibration did not converge: {result.message}")
    before = float(objective(initial)[0])
    if result.fun > before + 1e-9:
        raise RuntimeError("Calibration increased its validation objective")
    return {
        "mode": mode,
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]) if mode == "affine" else 0.0,
        "validation_rows": len(y),
        "validation_nll_before": before,
        "validation_nll_after": float(result.fun),
        "bounds": bounds,
        "at_bound": any(
            np.isclose(value, lower) or np.isclose(value, upper)
            for value, (lower, upper) in zip(result.x, bounds)
        ),
    }
