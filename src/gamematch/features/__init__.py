# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Declarative, executable input/output contracts for Shuach benchmark tracks.

Four orthogonal axes -- Modality (coefficient / dense / sequence), Scope
(match / team / player), Side (1 or 2) and Availability (pre/post match) --
are lifted into one canonical ``[B, S, N, T, C]`` layout. Every model consumes
the same ``ModelInputs`` interface; every head is declared by ``OutputSpec``.

This package has no PyTorch dependency, so catalog and validation code can be
used from a lightweight base installation.
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
    OutputSpec,
    Scope,
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
    "OutputSpec",
    "SchemaEncoder",
    "SchemaModel",
    "Scope",
    "make_head",
    "make_model_from_schema",
    "predictor_schema",
    "sample_inputs",
]

# Trainable bridge needs the optional `neural` extra. Keep it lazy so a base
# installation can still import this package and use the contract layer.
_TRAINABLE = {
    "SchemaEncoder",
    "SchemaModel",
    "make_head",
    "make_model_from_schema",
}


def __getattr__(name):
    if name in _TRAINABLE:
        from . import torch_bridge

        return getattr(torch_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
