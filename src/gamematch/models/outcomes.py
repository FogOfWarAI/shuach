# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Explicit history-to-outcome contract, independent of a game's class count."""

from dataclasses import dataclass

import torch

from ..catalog import HISTORY_MODELS
from .history import HistoryDeepSets
from .omninet import OwO


@dataclass(frozen=True)
class HistoryOutcomeSpec:
    """Declare the shape and label vocabulary shared by a batch and its model.

    Attributes:
        history_features: Feature width at each historical time step.
        static_features: Number of pre-match summary features per player.
        team_sizes: Two equal, positive roster sizes in side order.
        history_length: Number of time steps per player, including padding.
        outcome_labels: Distinct labels in output-column order.
        domains: Number of domain IDs, or zero when domain input is absent.
    """
    history_features: int
    static_features: int
    team_sizes: tuple[int, int]
    history_length: int
    outcome_labels: tuple[str, ...]
    domains: int = 0

    def __post_init__(self):
        for name in ("history_features", "static_features", "history_length"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.team_sizes, tuple)
            or len(self.team_sizes) != 2
            or any(type(n) is not int or n < 1 for n in self.team_sizes)
        ):
            raise ValueError("Declare two positive team sizes")
        if self.team_sizes[0] != self.team_sizes[1]:
            raise ValueError("Current history encoders require equal teams; no silent 1v4 padding")
        if (
            not isinstance(self.outcome_labels, tuple)
            or len(self.outcome_labels) < 2
            or any(not isinstance(s, str) or not s.strip() for s in self.outcome_labels)
            or len(set(self.outcome_labels)) != len(self.outcome_labels)
        ):
            raise ValueError("Declare distinct ordered class labels")
        if type(self.domains) is not int or self.domains < 0:
            raise ValueError("domains must be a nonnegative integer")


def build_history_outcome_model(
    name,
    spec,
    *,
    width=32,
    heads=4,
    layers=2,
    dropout=0.1,
    pooling="team",
    norm_first=False,
    ordered_slots=False,
):
    """Build an outcome network; defaults are a declared adaptation, not an HPO result.

    Class index i always refers to spec.outcome_labels[i]. Call
    validate_history_batch at the data boundary, not on every GPU forward.
    """
    if name not in HISTORY_MODELS:
        raise ValueError(f"Unknown history outcome model: {name}")
    if not isinstance(spec, HistoryOutcomeSpec):
        raise ValueError("An explicit HistoryOutcomeSpec is required")
    for key, value in {"width": width, "heads": heads, "layers": layers}.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if isinstance(dropout, bool) or not isinstance(dropout, (float, int)) or not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if type(norm_first) is not bool or type(ordered_slots) is not bool:
        raise ValueError("norm_first and ordered_slots must be booleans")
    if pooling not in {"team", "global"}:
        raise ValueError("pooling must be team or global")
    if name in {"static_deepsets", "history_deepsets"}:
        if ordered_slots:
            raise ValueError("Deep Sets does not implement ordered role slots")
        return HistoryDeepSets(
            spec.history_features,
            spec.static_features,
            width=width,
            classes=len(spec.outcome_labels),
            use_history=name == "history_deepsets",
            dropout=dropout,
        )
    if width % heads:
        raise ValueError("width must be divisible by heads")
    if name == "dawn" and spec.domains == 0:
        raise ValueError(
            "DAWN requires a declared domain vocabulary, including a single-domain case"
        )
    return OwO(
        spec.history_features,
        spec.static_features,
        team_size=spec.team_sizes[0],
        history_length=spec.history_length,
        width=width,
        heads=heads,
        layers=layers,
        dropout=dropout,
        classes=len(spec.outcome_labels),
        domains=spec.domains if name == "dawn" else 0,
        ordered_slots=ordered_slots,
        omni=name != "owo_transformer",
        pooling=pooling,
        norm_first=norm_first,
    )


def validate_history_batch(
    spec,
    history,
    static,
    valid,
    *,
    domain=None,
    target=None,
    prediction_time=None,
    history_available_at=None,
):
    """Validate a history batch at the input boundary.

    Args:
        spec: Feature widths, team sizes, class order and domain vocabulary.
        history: Finite floating tensor [batch, 2, players, steps, history_features].
        static: Finite tensor [batch, 2, players, static_features] of the same dtype.
        valid: Boolean tensor [batch, 2, players, steps]. True marks observed steps.
        domain: int64 vector [batch], required when the spec declares domains.
        target: Optional int64 vector [batch] of indices into spec.outcome_labels.
        prediction_time: Optional finite numeric vector [batch].
        history_available_at: Availability times shaped like valid, in the same
            unit as prediction_time. Supply both time tensors together.

    Returns:
        Batch size, class labels, a temporal_evidence_checked flag and the
        number of players with an empty history across the batch.

    Raises:
        ValueError: Tensors disagree with the contract, or an observed step
            became available at or after its prediction time.
    """
    if not isinstance(spec, HistoryOutcomeSpec):
        raise ValueError("An explicit HistoryOutcomeSpec is required")
    if not all(isinstance(x, torch.Tensor) for x in (history, static, valid)):
        raise ValueError("History inputs must be tensors")
    if history.ndim != 5 or history.shape[0] < 1:
        raise ValueError("Expected nonempty history [batch, side, player, step, feature]")
    batch = history.shape[0]
    expected = (batch, 2, spec.team_sizes[0], spec.history_length, spec.history_features)
    if tuple(history.shape) != expected:
        raise ValueError("History tensor does not match the declared contract")
    if tuple(static.shape) != (*expected[:3], spec.static_features):
        raise ValueError("Static features do not match the declared contract")
    if valid.shape != history.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("History mask must be aligned and boolean")
    if (
        not history.is_floating_point()
        or not static.is_floating_point()
        or history.dtype != static.dtype
    ):
        raise ValueError("History and static features need the same floating dtype")
    if len({x.device for x in (history, static, valid)}) != 1:
        raise ValueError("All inputs must use the same device")
    if not torch.isfinite(history).all() or not torch.isfinite(static).all():
        raise ValueError("Nonfinite input features, including masked positions, are unsupported")

    def index_vector(value, name, count):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (batch,)
            or value.dtype != torch.long
            or value.device != history.device
            or ((value < 0) | (value >= count)).any()
        ):
            raise ValueError(f"{name} must be aligned int64 indices in [0, {count})")

    if spec.domains:
        index_vector(domain, "domain", spec.domains)
    elif domain is not None:
        raise ValueError("Domain values require a declared vocabulary")
    if target is not None:
        index_vector(target, "target", len(spec.outcome_labels))
    if (prediction_time is None) != (history_available_at is None):
        raise ValueError("Both prediction and history availability times are required")
    checked_time = prediction_time is not None
    if checked_time:
        for name, value, shape in [
            ("prediction_time", prediction_time, (batch,)),
            ("history_available_at", history_available_at, tuple(valid.shape)),
        ]:
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape
                or value.device != history.device
                or value.dtype == torch.bool
                or value.is_complex()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid {name}; use aligned finite numeric times in one unit")
        if (
            history_available_at[valid]
            >= prediction_time[:, None, None, None].expand_as(valid)[valid]
        ).any():
            raise ValueError("History feedback must be strictly earlier than prediction time")
    return {
        "batch_size": batch,
        "classes": list(spec.outcome_labels),
        "temporal_evidence_checked": checked_time,
        "empty_player_histories": int((~valid.any(-1)).sum()),
    }
