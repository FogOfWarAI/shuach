# SPDX-License-Identifier: Apache-2.0
"""Prediction targets, losses and side-swap semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Orientation(str, Enum):
    """How a two-sided target behaves when the sides are swapped."""

    ANTISYMMETRIC = "antisymmetric"   # left_win: swapping sides flips the label
    SYMMETRIC = "symmetric"           # e.g. balance / fairness: label unchanged
    NONE = "none"                     # single-sided or non-relational target


# Deprecated: kept only so old JSON configs keep loading. The PRE/POST binary
# is no longer a predicate anywhere in validation, encoding or model building.
class Availability(str, Enum):
    PRE_MATCH = "pre_match"
    POST_MATCH = "post_match"


@dataclass(frozen=True)
class OutputSpec:
    """Declare what the model predicts, including ordered label semantics."""

    name: str
    kind: str                                    # binary|multiclass|ordinal|regression|assignment
    labels: tuple[str, ...] = ()
    dim: int = 1
    link: str = "sigmoid"
    orientation: Orientation | None = None       # None -> inferred from kind
    available: Availability | None = None        # DEPRECATED, ignored everywhere

    def __post_init__(self):
        kinds = {"binary", "multiclass", "ordinal", "regression", "assignment"}
        if self.kind not in kinds:
            raise ValueError(f"kind must be one of {sorted(kinds)}")
        if self.link not in {"sigmoid", "softmax", "identity", "none"}:
            raise ValueError(f"Unknown output link: {self.link}")
        if type(self.dim) is not int or self.dim < 1:
            raise ValueError("dim must be a positive integer")
        if self.kind in {"multiclass", "ordinal"}:
            if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
                raise ValueError("Declare >= 2 distinct ordered labels")
            if self.dim != len(self.labels):
                raise ValueError("dim must equal the number of labels")
        if self.kind == "binary" and self.labels and len(self.labels) != 2:
            raise ValueError("A binary head needs either no labels or exactly two")
        if self.kind == "binary" and self.dim != 1:
            raise ValueError("The binary head requires dim=1")
        if self.orientation is not None and not isinstance(self.orientation, Orientation):
            raise ValueError("orientation must be an Orientation member or None")

    @property
    def resolved_orientation(self) -> Orientation:
        """Win/loss-style targets flip when the two sides are swapped."""
        if self.orientation is not None:
            return self.orientation
        if self.kind == "binary":
            return Orientation.ANTISYMMETRIC
        if self.kind in {"multiclass", "ordinal"}:
            return Orientation.ANTISYMMETRIC
        return Orientation.NONE

    def criterion(self) -> str:
        """The canonical loss name for this head; consumed by the torch bridge."""
        if self.kind == "binary":
            return "bce_with_logits"
        if self.kind in {"multiclass", "ordinal"}:
            return "cross_entropy"
        if self.kind == "regression":
            return "mse"
        return "none"
