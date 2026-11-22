# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Validate experiment counts before creating outputs or loading large data."""

import importlib.metadata
import math
import platform
from numbers import Real


def validate_training_settings(settings):
    """Reject malformed counts and nonfinite optimizer settings before side effects."""
    for name in (
        "epochs",
        "batch_size",
        "patience",
        "embedding_dim",
        "hidden_dim",
        "width",
        "heads",
        "layers",
        "threads",
        "max_trees",
        "tree_patience",
        "history_length",
    ):
        if name in settings and (type(settings[name]) is not int or settings[name] < 1):
            raise ValueError(f"{name} must be a positive integer")
    for name in (
        "learning_rate",
        "weight_decay",
        "tree_learning_rate",
        "logistic_c",
        "dropout",
        "deepsets_dropout",
    ):
        if name not in settings:
            continue
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
        if name in {"dropout", "deepsets_dropout"}:
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        elif value < 0 or (name != "weight_decay" and value == 0):
            raise ValueError(f"Invalid {name}")
    if "device" in settings and settings["device"] not in ("cpu", "cuda", "mps"):
        raise ValueError("Explicit device must be cpu, cuda or mps")


def runtime_identity():
    """Record versions that must not be silently mixed in a resumed run."""
    versions = {}
    for package in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "joblib",
        "threadpoolctl",
        "lightgbm",
        "trueskill",
        "openskill",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "dependencies": versions,
    }


def validate_matrix(config, axes):
    for name in axes:
        values = config.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a nonempty list")
        if name == "seeds":
            if any(type(value) is not int or not 0 <= value < 2**32 for value in values):
                raise ValueError("Seeds must be integers in [0, 2**32)")
        elif any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{name} entries must be nonempty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {name} entries")
    for name in ["max_train_rows", "max_eval_rows", "lobbies", "history_length"]:
        value = config.get(name)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer or null")
