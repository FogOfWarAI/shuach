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
    Availability,
    FeatureBatch,
    FeatureSchema,
    Modality,
    ModelInputs,
    OutputSpec,
    Scope,
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

    def forward(self, batch: FeatureBatch) -> ModelInputs:
        device = self.device
        B = batch.batch_size
        S, N, D = self.schema.sides, self.schema.players, self.dim
        tokens = torch.zeros(B, S, N, D, device=device)
        for spec in self.schema.features:
            # PRE_MATCH features only: POST_MATCH fields are labels by contract.
            if spec.available is Availability.POST_MATCH:
                continue
            local = self._encode(spec, batch, device)        # [B, s_eff, n_eff, D]
            s_eff, n_eff = local.shape[1], local.shape[2]
            if s_eff == 1 and S > 1:
                local = local.expand(B, S, n_eff, D)
            if n_eff == 1 and N > 1:
                local = local.expand(B, S, N, D)
            tokens = tokens + local

        valid = torch.ones(B, S, N, dtype=torch.bool, device=device)
        for spec in self.schema.features:
            if spec.mask and spec.scope is Scope.PLAYER and spec.available is not Availability.POST_MATCH:
                mask = _tensor(batch.masks[spec.name], device=device, dtype=torch.bool)
                valid = valid & mask.any(-1)

        denom = valid.sum(2, keepdim=True).clamp(min=1)
        team = (tokens * valid.unsqueeze(-1)).sum(2) / denom
        match = team.mean(1)
        return ModelInputs(tokens, valid, match, team, self.schema)

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
            if spec.mask:
                mask = _tensor(batch.masks[spec.name], device=device, dtype=torch.bool)
                projected = projected * mask.unsqueeze(-1).to(projected.dtype)
                count = mask.sum(-1, keepdim=True).clamp(min=1)
            else:
                count = torch.tensor(field.shape[-2], device=device)
            return projected.sum(-2) / count.to(projected.dtype)
        return projection(field).squeeze(-2)                         # DENSE: T == 1


def make_head(spec: OutputSpec, in_dim: int) -> nn.Module:
    """Build the output head declared by an :class:`OutputSpec`."""
    if spec.kind in {"binary", "regression", "assignment"}:
        return nn.Linear(in_dim, 1)
    if spec.kind in {"multiclass", "ordinal"}:
        return nn.Linear(in_dim, spec.dim)
    raise ValueError(f"Unsupported output kind: {spec.kind}")


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
        return self.head(inputs.match_context)

    def predict_proba(self, batch: FeatureBatch):
        logits = self.forward(batch)
        task = self.schema.task
        if task.kind == "binary":
            positive = torch.sigmoid(logits).squeeze(-1)
            return torch.stack([1 - positive, positive], dim=-1)
        if task.kind in {"multiclass", "ordinal"}:
            return torch.softmax(logits, dim=-1)
        return logits


def make_model_from_schema(schema: FeatureSchema, *, dim: int = 16, seed: int = 0) -> SchemaModel:
    """Factory returning a trainable :class:`SchemaModel` for a contract."""
    return SchemaModel(schema, dim=dim, seed=seed)
