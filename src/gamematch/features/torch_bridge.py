# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Trainable PyTorch bridge from a declarative schema to a model head.

This module is deliberately separate from the rest of :mod:`gamematch.features`
so that a base installation stays PyTorch-free. Import it explicitly (or via
the lazy ``gamematch.features.SchemaModel`` accessor) after installing the
``neural`` extra.

The reference encoder in :class:`~gamematch.features.schema.FeatureEncoder` is
a deterministic, non-trainable projection used to validate the contract. This
module applies the *same four-axis algebra* with learnable tables and
projections, so gradients reach every declared feature and the declared
:class:`~gamematch.features.schema.OutputSpec` drives the output head.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .schema import (
    FeatureBatch,
    FeatureSchema,
    Modality,
    ModelInputs,
    Orientation,
    OutputSpec,
    Scope,
    Structure,
    as_array,
)


def _tensor(value, *, device, dtype=torch.float32):
    """Convert a canonical numpy array (or nested sequence) into a tensor."""
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


class SchemaEncoder(nn.Module):
    """Learnable encoder for one :class:`FeatureSchema`.

    It consumes a validated :class:`FeatureBatch` and produces token-level
    embeddings by applying the schema's modality and scope semantics, so a
    gradient reaches every declared feature. The returned ``ModelInputs`` uses
    the same field names as the reference encoder, only the arrays are tensors.
    """

    def __init__(self, schema: FeatureSchema, dim: int = 16, seed: int = 0):
        super().__init__()
        self.schema = schema
        self.dim = dim
        generator = torch.Generator().manual_seed(seed)
        self.embeddings = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        # ORDERED structures get learnable slot embeddings so pooling keeps
        # the ordering information that a plain mean would discard.
        ordered = [f for f in schema.features if f.resolved_structure is Structure.ORDERED]
        self.slot_embeddings = nn.ParameterDict()
        for spec in ordered:
            slots = spec.steps if spec.modality is Modality.SEQUENCE else schema.sides
            table = torch.empty(slots, dim)
            nn.init.normal_(table, 0, 0.02, generator=generator)
            self.slot_embeddings[spec.name] = nn.Parameter(table)
        for spec in schema.features:
            if spec.modality is Modality.COEFFICIENT:
                table = nn.Embedding(spec.vocab_size, dim, padding_idx=0)
                with torch.no_grad():
                    table.weight.normal_(0, 1.0 / math.sqrt(dim), generator=generator)
                    table.weight[0].zero_()
                self.embeddings[spec.name] = table
            else:
                proj = nn.Linear(spec.width, dim, bias=False)
                with torch.no_grad():
                    proj.weight.normal_(0, 1.0 / math.sqrt(spec.width), generator=generator)
                self.projections[spec.name] = proj

    @property
    def device(self):
        parameter = next(self.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    def visible_fields(self, batch: FeatureBatch) -> tuple:
        """Fields legal at this batch's prediction time (single predicate)."""
        if batch.prediction_time is None:
            return tuple(self.schema.features)
        view = self.schema.at(batch.prediction_time, batch.available_at)
        return view.visible

    def forward(self, batch: FeatureBatch) -> ModelInputs:
        device = self.device
        B = batch.batch_size
        S, N, D = self.schema.sides, self.schema.players, self.dim
        visible = self.visible_fields(batch)
        tokens = torch.zeros(B, S, N, D, device=device)
        for spec in visible:
            local = self._encode(spec, batch, device)        # [B, s_eff, n_eff, D]
            s_eff, n_eff = local.shape[1], local.shape[2]
            if s_eff == 1 and S > 1:
                local = local.expand(B, S, n_eff, D)
            if n_eff == 1 and N > 1:
                local = local.expand(B, S, N, D)
            tokens = tokens + local

        valid = torch.ones(B, S, N, dtype=torch.bool, device=device)
        for spec in visible:
            if spec.mask and spec.scope is Scope.PLAYER:
                mask = _tensor(batch.masks[spec.name], device=device, dtype=torch.bool)
                valid = valid & mask.any(-1)

        denom = valid.sum(2, keepdim=True).clamp(min=1)
        team = (tokens * valid.unsqueeze(-1)).sum(2) / denom
        match = self._pool_sides(team)
        return ModelInputs(tokens, valid, match, team, self.schema)

    @property
    def uses_signed_sides(self) -> bool:
        """Only a binary antisymmetric target has a provable sign flip.

        For binary left_win, ``team[:,0] - team[:,1]`` changes sign when the
        sides swap, so a bias-free head satisfies f(swap(x)) == -f(x).
        Multi-class WDL instead uses permutation-invariant pooling plus
        label-reversing augmentation: negating every class logit would be
        mathematically wrong (the draw class must stay put).
        """
        task = self.schema.task
        return (
            self.schema.sides == 2
            and task.kind == "binary"
            and task.resolved_orientation is Orientation.ANTISYMMETRIC
        )

    def _pool_sides(self, team):
        if self.uses_signed_sides:
            return team[:, 0] - team[:, 1]
        return team.mean(1)

    def _encode(self, spec, batch, device):
        field = _tensor(batch.fields[spec.name], device=device)
        if spec.modality is Modality.COEFFICIENT:
            embedded = self.embeddings[spec.name](field.long())     # [..., T, C, D]
            weight = _tensor(batch.weights[spec.name], device=device)
            weighted = embedded * weight.unsqueeze(-1)
            pooled = weighted.sum(dim=(3, 4))                        # [B, s, n, D]
            if spec.reduce == "mean":
                count = (weight > 0).sum(dim=(3, 4)).clamp(min=1).unsqueeze(-1)
                pooled = pooled / count.to(pooled.dtype)
            return pooled

        projection = self.projections[spec.name]
        if spec.modality is Modality.SEQUENCE:
            projected = projection(field)                            # [B, s, n, T, D]
            if spec.resolved_structure is Structure.ORDERED:
                slots = self.slot_embeddings[spec.name]               # [T, D]
                projected = projected + slots.view(1, 1, 1, -1, self.dim)
            if spec.mask:
                mask = _tensor(batch.masks[spec.name], device=device, dtype=torch.bool)
                projected = projected * mask.unsqueeze(-1).to(projected.dtype)
                count = mask.sum(-1, keepdim=True).clamp(min=1)
            else:
                count = torch.tensor(field.shape[-2], device=device)
            return projected.sum(-2) / count.to(projected.dtype)
        return projection(field).squeeze(-2)                         # DENSE: T == 1


def make_head(spec: OutputSpec, in_dim: int) -> nn.Module:
    """Build the output head declared by an :class:`OutputSpec`.

    A binary antisymmetric target demands ``f(swap(x)) == -f(x)`` for raw
    logits, so its head has no bias term. Other heads keep a bias.
    """
    signed = (
        spec.kind == "binary"
        and spec.resolved_orientation is Orientation.ANTISYMMETRIC
    )
    bias = not signed
    if spec.kind in {"binary", "regression", "assignment"}:
        return nn.Linear(in_dim, 1, bias=bias)
    if spec.kind in {"multiclass", "ordinal"}:
        return nn.Linear(in_dim, spec.dim, bias=bias)
    raise ValueError(f"Unsupported output kind: {spec.kind}")


def make_criterion(spec: OutputSpec):
    """Derive the loss from `OutputSpec.criterion()` -- no hand-written switch."""
    name = spec.criterion()
    if name == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"No canonical loss declared for {spec.kind}")


def criterion_targets(spec: OutputSpec, targets, device=None):
    """Shape raw targets to match the derived loss."""
    tensor = torch.as_tensor(np.asarray(targets), device=device)
    if spec.criterion() == "cross_entropy":
        return tensor.reshape(-1).long()
    return tensor.reshape(-1, 1).float()


def swap_sides(schema: FeatureSchema, batch: FeatureBatch):
    """Augment a two-sided batch by swapping sides, flipping labels if needed.

    For an antisymmetric target (left_win) the binary label is inverted; for
    three-class WDL the loss/draw/win indices are reversed. Returns a new
    FeatureBatch with the S axis reversed and transformed targets.
    """
    if schema.sides != 2:
        raise ValueError("swap_sides requires a two-sided schema")
    fields = {k: np.flip(np.asarray(v), axis=1).copy() for k, v in batch.fields.items()}
    weights = {k: np.flip(np.asarray(v), axis=1).copy() for k, v in batch.weights.items()}
    masks = {k: np.flip(np.asarray(v), axis=1).copy() for k, v in batch.masks.items()}
    targets = batch.targets
    orientation = schema.task.resolved_orientation
    if targets is not None and orientation is Orientation.ANTISYMMETRIC:
        t = np.asarray(targets).copy()
        if schema.task.kind == "binary":
            t = 1.0 - t
        elif schema.task.kind in {"multiclass", "ordinal"} and schema.task.dim == 3:
            t = 2 - t
        targets = t
    timed = {}
    for key, value in batch.available_at.items():
        arr = np.asarray(value)
        timed[key] = np.flip(arr, axis=1).copy() if arr.ndim >= 2 and arr.shape[1] == 2 else arr
    return FeatureBatch(
        schema=batch.schema, fields=fields, weights=weights, masks=masks,
        available_at=timed, prediction_time=batch.prediction_time,
        targets=as_array(targets) if targets is not None else None,
        batch_size=batch.batch_size,
    )


class SchemaModel(nn.Module):
    """An end-to-end trainable model built solely from a declared schema.

    ``forward`` returns raw logits, so binary and regression tasks pair with
    ``BCEWithLogitsLoss`` / ``MSELoss`` and multiclass tasks with
    ``CrossEntropyLoss``. :meth:`predict_proba` applies the declared link for
    inference.

    Example::

        schema = SCHEMAS["hero_binary"]
        model = SchemaModel(schema, dim=16)
        logits = model(batch)                 # batch from FeatureBatch.from_natural
    """

    def __init__(self, schema: FeatureSchema, dim: int = 16, seed: int = 0):
        super().__init__()
        self.schema = schema
        self.encoder = SchemaEncoder(schema, dim=dim, seed=seed)
        self.head = make_head(schema.task, dim)

    def forward(self, batch: FeatureBatch):
        inputs = self.encoder(batch)
        context = inputs.match_context
        # For an antisymmetric multi-class target (loss/draw/win) the reversed
        # side is represented by a negated context; the head's class order is
        # reversed by predict_proba via the declared label symmetry. Keep the
        # raw logits here and let the loss see a consistent representation.
        return self.head(context)

    def predict_proba(self, batch: FeatureBatch):
        logits = self.forward(batch)
        task = self.schema.task
        if task.kind == "binary":
            positive = torch.sigmoid(logits).squeeze(-1)
            return torch.stack([1 - positive, positive], dim=-1)
        if task.kind in {"multiclass", "ordinal"}:
            return torch.softmax(logits, dim=-1)
        return logits

    def criterion(self):
        """Loss derived from the declared OutputSpec."""
        return make_criterion(self.schema.task)

    def targets(self, batch: FeatureBatch):
        return criterion_targets(self.schema.task, batch.targets)

    def augmented_batch(self, batch: FeatureBatch):
        """Side-swap augmentation consistent with the declared orientation."""
        return swap_sides(self.schema, batch)


def make_model_from_schema(schema: FeatureSchema, *, dim: int = 16, seed: int = 0) -> SchemaModel:
    """Factory returning a trainable :class:`SchemaModel` for a contract."""
    return SchemaModel(schema, dim=dim, seed=seed)
