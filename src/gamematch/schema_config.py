# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Training configuration for the declarative SchemaModel path.

The boundary is deliberate:

- The **feature schema** declares what the data is and how sides/timing work
  (`structure`, `orientation`, shapes, vocab sizes, output head).
- This config declares only the **optimization process** (learning rate,
  epochs, batch size, seed, device, token width). It cannot override schema
  semantics, so a config can never accidentally change what a run measures.

The same JSON convention, validation helpers and ``asdict`` artifact pattern
as :mod:`gamematch.runner` are reused, so schema-model runs stay auditable in
the same way as native runner experiments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .configuration import validate_training_settings

__all__ = ["SchemaTrainConfig", "load_schema_train_config"]


@dataclass(frozen=True)
class SchemaTrainConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    patience: int = 5
    dim: int = 16
    seed: int = 0
    device: str = "cpu"
    threads: int = 2

    def validate(self):
        """Reject malformed values before creating outputs or touching data."""
        # The shared validator covers epochs/batch_size/patience/threads/dim is
        # checked here because it is this path's embedding width, not a runner
        # hidden width. weight_decay=0 is legal (shared validator allows it).
        settings = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "threads": self.threads,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "device": self.device,
        }
        validate_training_settings(settings)
        if type(self.dim) is not int or self.dim < 2:
            raise ValueError("dim must be an integer >= 2")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        return self

    def to_dict(self):
        return asdict(self)


def load_schema_train_config(source, *, overrides=None):
    """Load a SchemaTrainConfig from a JSON path or decoded object.

    Args:
        source: Path to a JSON file, JSON string, or decoded dict. The config
            may be the whole object or a ``schema_training`` block inside a
            larger benchmark config.
        overrides: Optional dict applied after loading (CLI flags).
    """
    if isinstance(source, (str, Path)) and str(source).strip().startswith("{"):
        data = json.loads(source)
    elif isinstance(source, (str, Path)):
        data = json.loads(Path(source).read_text())
    else:
        data = dict(source)
    if not isinstance(data, dict):
        raise ValueError("schema training config must be a JSON object")
    values = data.get("schema_training", data)
    if not isinstance(values, dict):
        raise ValueError("schema_training must be a JSON object")
    allowed = set(SchemaTrainConfig.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown schema training keys: {sorted(unknown)}")
    merged = dict(values)
    if overrides:
        unexpected = set(overrides) - allowed
        if unexpected:
            raise ValueError(f"unknown schema training overrides: {sorted(unexpected)}")
        merged.update(overrides)
    return SchemaTrainConfig(**merged).validate()
