# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Declarative, executable input/output contracts for Shuach benchmark tracks.

Modality (coefficient / dense / sequence), Scope (match / team / player), Side
(1 or 2) and Structure (set / ordered / single) are lifted into one canonical
``[B, S, N, T, C]`` layout. Availability has exactly one predicate:
``available_at < prediction_time``, evaluated by ``FeatureSchema.at(t)``.
Every model consumes the same ``ModelInputs`` interface; the head, loss and
side-swap semantics are all derived from ``OutputSpec``.

This package has no PyTorch dependency, so catalog and validation code can be
used from a lightweight base installation; trainable exports are lazy.
"""

from .schema import (
    SCHEMAS,
    Availability,
    FeatureBatch,
    FeatureEncoder,
    FeatureSchema,
    FeatureSpec,
    Modality,
    ModelInputs,
    Orientation,
    OutputSpec,
    SchemaView,
    Scope,
    Structure,
    predictor_schema,
    sample_inputs,
)

__all__ = [
    "SCHEMAS",
    "Availability",
    "FeatureBatch",
    "FeatureEncoder",
    "FeatureSchema",
    "FeatureSpec",
    "ModelInputs",
    "Modality",
    "Orientation",
    "OutputSpec",
    "SchemaEncoder",
    "SchemaModel",
    "SchemaView",
    "Scope",
    "Structure",
    "make_criterion",
    "make_head",
    "make_model_from_schema",
    "predictor_schema",
    "sample_inputs",
    "swap_sides",
]

# Trainable bridge needs the optional `neural` extra. Keep it lazy so a base
# installation can still import this package and use the contract layer.
_TRAINABLE = {
    "SchemaEncoder",
    "SchemaModel",
    "make_criterion",
    "make_head",
    "make_model_from_schema",
    "swap_sides",
}


def __getattr__(name):
    if name in _TRAINABLE:
        from . import torch_bridge

        return getattr(torch_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
