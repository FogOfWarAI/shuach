# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""
Unified feature-input contract for game-matchmaking benchmarks.

WHY THIS EXISTS
---------------
Shuach (gamematch) currently defines inputs as seven parallel, hard-coded
tensor contracts: HeroRecords [B,2,5], history [B,2,6,10,5] + static [B,2,6,13],
DBD one-hot [B,51/66], Dota snapshots [B,12,6], synthetic [B,2,5], etc.
`docs/data-contracts.md` describes them in a Markdown table and `catalog.py`
stores `input_schemas` as *strings*. Neither drives validation, encoding, or
model construction. Adding a feature means editing an adapter, a model
signature and a runner at once.

This module turns that table into an executable contract.

DESIGN: FOUR ORTHOGONAL AXES
----------------------------
  Modality   COEFFICIENT | DENSE | SEQUENCE
             - COEFFICIENT: sparse / ID-keyed coefficients over a vocabulary
               (hero IDs, signed -1/0/1 indicators, rating score terms s/abs(s)).
               Represented as (index, weight) pairs; index 0 is the unknown /
               padding token. `hero_counts` is just a sum-reduced special case.
             - DENSE: fixed-width floating vectors (13 static summaries,
               9 DBD numerics, standardized residual references).
             - SEQUENCE: ordered time steps plus a validity mask
               (10 history steps x 5 fields, 12 snapshots x 6 fields).
  Scope      MATCH | TEAM | PLAYER   (the "individual / team" axis, plus MATCH
             for map, mode, domain, prediction_time, lobby -- these have no
             home in a two-level hierarchy).
  Side       layout.sides: 2 for two-sided matchmaking, 1 for single-sided
             tasks (DBD predator, Dota focal player). Kept explicit so left /
             right information can never silently mix.
  Time       Availability PRE_MATCH / POST_MATCH plus an optional
             `available_at` timestamp field. Leakage check becomes a failed
             assertion instead of a paragraph in the docs.

CANONICAL LAYOUT
----------------
Every field, regardless of modality or scope, is lifted to one 5-D shape
[B, S, N, T, C]: batch, sides, players-per-side, steps, channels.
Scope decides which axes collapse to 1; SEQUENCE is the only modality with
T > 1. One validator, one encoder, one downstream model interface.

  Scope / Modality     Natural input        Canonical
  MATCH    dense       [B, C]               [B,1,1,1,C]
  TEAM     dense       [B,S,C]              [B,S,1,1,C]
  PLAYER   dense       [B,S,N,C]            [B,S,N,1,C]
  MATCH    sequence    [B,T,C]              [B,1,1,T,C]
  TEAM     sequence    [B,S,T,C]            [B,S,1,T,C]
  PLAYER   sequence    [B,S,N,T,C]          [B,S,N,T,C]
  any      coef        same as dense        same, int64 indices + float weights

OUTPUT is declared too: OutputSpec covers binary, multiclass, ordinal,
regression and assignment heads, with an ordered label vocabulary.

DEPENDENCIES
------------
The contract/validation layer is pure standard library. numpy is used
automatically when installed (recommended for real batches); otherwise a
small reference array is used so the contract can always be self-checked.
For integration with Shuach, replace FeatureEncoder with a torch.nn.Module
that consumes the same FeatureBatch -- the contract does not change.

Usage
-----
    schema = SCHEMAS["overwatch_history"]
    batch  = FeatureBatch.from_natural(schema, history=..., static=..., ...)
    report = schema.validate(batch)
    inputs = FeatureEncoder(schema, dim=32)(batch)   # -> ModelInputs
    print(schema.to_json())
"""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass, field, replace
from enum import Enum

try:
    import numpy as _np
except ImportError:  # contract layer must work without a numerical library
    _np = None

__all__ = [
    "Modality", "Scope", "Availability", "FeatureSpec", "OutputSpec",
    "FeatureSchema", "FeatureBatch", "ModelInputs", "FeatureEncoder",
    "SCHEMAS", "predictor_schema",
]


# --------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------

class Modality(str, Enum):
    COEFFICIENT = "coefficient"
    DENSE = "dense"
    SEQUENCE = "sequence"


class Scope(str, Enum):
    MATCH = "match"
    TEAM = "team"
    PLAYER = "player"


class Availability(str, Enum):
    PRE_MATCH = "pre_match"
    POST_MATCH = "post_match"


# --------------------------------------------------------------------------
# Minimal array shim (used only when numpy is absent)
# --------------------------------------------------------------------------

class _LiteArray:
    __slots__ = ("flat", "shape")

    def __init__(self, flat, shape):
        self.flat, self.shape = list(flat), tuple(shape)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return self.flat[key]
        index = 0
        for k, size in zip(key, self.shape):
            index = index * size + (k if size > 1 else 0)
        return self.flat[index]

    def __iter__(self):
        return iter(self.flat)

    def __len__(self):
        return len(self.flat)


def _flatten(value):
    """Return (flat_values, shape) for a nested sequence."""
    if not isinstance(value, (list, tuple)):
        return [value], ()
    if not value:
        return [], (0,)
    flat, shape = [], None
    for item in value:
        sub, sub_shape = _flatten(item)
        if shape is None:
            shape = sub_shape
        elif sub_shape != shape:
            raise ValueError("Ragged input is not a tensor")
        flat.extend(sub)
    return flat, (len(value), *shape)


def as_array(value):
    """Convert a nested sequence / numpy array into an indexable array."""
    if _np is not None:
        return _np.asarray(value)
    if isinstance(value, _LiteArray):
        return value
    flat, shape = _flatten(value)
    return _LiteArray(flat, shape)


def _array(shape, fill=0.0, dtype=None):
    size = math.prod(shape)
    if _np is not None:
        return _np.full(shape, fill, dtype=dtype or float)
    return _LiteArray([fill] * size, shape)


# --------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    """One named field. This is the only thing a user writes to add a feature."""

    name: str
    modality: Modality
    scope: Scope
    width: int                                   # C: dense channels / coef slots
    steps: int = 1                               # T: >1 only for SEQUENCE
    vocab_size: int = 0                          # required by COEFFICIENT
    mask: bool = False                           # explicit validity mask
    available: Availability = Availability.PRE_MATCH
    available_at: str | None = None              # timestamp field name in batch
    reduce: str = "sum"                          # coef: sum|mean|max; seq: mean|last
    transform: str = "identity"                  # identity|log1p|standardize|onehot
    group: str = ""                              # semantic group, for ablations
    description: str = ""

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Feature name must be a nonempty string")
        if not isinstance(self.modality, Modality) or not isinstance(self.scope, Scope):
            raise ValueError("modality and scope must be enum members")
        for key in ("width", "steps", "vocab_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.width < 1 or self.steps < 1:
            raise ValueError("width and steps must be positive")
        if self.modality is Modality.SEQUENCE:
            if self.steps < 2:
                raise ValueError("SEQUENCE needs steps >= 2; use DENSE for static features")
            if not self.mask:
                raise ValueError("SEQUENCE must declare an explicit validity mask")
        elif self.steps != 1:
            raise ValueError("steps > 1 is only valid for SEQUENCE")
        if self.modality is Modality.COEFFICIENT:
            if self.vocab_size < 2:
                raise ValueError("COEFFICIENT needs vocab_size >= 2 (0 is the unknown token)")
            if self.reduce not in {"sum", "mean", "max"}:
                raise ValueError("coefficient reduce must be sum, mean or max")
        elif self.vocab_size:
            raise ValueError("vocab_size is only valid for COEFFICIENT")
        if self.reduce not in {"sum", "mean", "max", "last"}:
            raise ValueError("Unknown reduce")
        if self.transform not in {"identity", "log1p", "standardize", "onehot"}:
            raise ValueError("Unknown transform")

    @property
    def key(self) -> tuple[str, str]:
        """The (modality, scope) cell of the design matrix."""
        return (self.modality.value, self.scope.value)


@dataclass(frozen=True)
class OutputSpec:
    """Declare what the model predicts, including ordered label semantics."""

    name: str
    kind: str                                    # binary|multiclass|ordinal|regression|assignment
    labels: tuple[str, ...] = ()
    dim: int = 1
    link: str = "sigmoid"
    available: Availability = Availability.POST_MATCH

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


@dataclass(frozen=True)
class FeatureSchema:
    """The complete input/output contract for one benchmark track."""

    name: str
    sides: int = 2
    players: int = 5
    features: tuple[FeatureSpec, ...] = ()
    task: OutputSpec = field(default_factory=lambda: OutputSpec("win", "binary"))

    def __post_init__(self):
        if type(self.sides) not in (int,) or self.sides < 1:
            raise ValueError("sides must be a positive integer")
        if type(self.players) not in (int,) or self.players < 1:
            raise ValueError("players must be a positive integer")
        if len({f.name for f in self.features}) != len(self.features):
            raise ValueError("Duplicate feature names")
        # sides == 1 is legitimate (DBD predator, Dota focal player): the S axis
        # simply collapses. Scope never changes the allowed side count.

    # -- shape algebra -----------------------------------------------------

    def canonical_shape(self, spec: FeatureSpec, batch_size: int = -1) -> tuple[int, ...]:
        s = self.sides if spec.scope in (Scope.TEAM, Scope.PLAYER) else 1
        n = self.players if spec.scope is Scope.PLAYER else 1
        t = spec.steps if spec.modality is Modality.SEQUENCE else 1
        return (batch_size, s, n, t, spec.width)

    def natural_shape(self, spec: FeatureSpec, batch_size: int) -> tuple[int, ...]:
        parts = [batch_size]
        if spec.scope in (Scope.TEAM, Scope.PLAYER):
            parts.append(self.sides)
        if spec.scope is Scope.PLAYER:
            parts.append(self.players)
        if spec.modality is Modality.SEQUENCE:
            parts.append(spec.steps)
        parts.append(spec.width)
        return tuple(parts)

    def mask_shape(self, batch_size: int) -> tuple[int, ...]:
        longest = max((f.steps for f in self.features), default=1)
        return (batch_size, self.sides, self.players, longest)

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name, "sides": self.sides, "players": self.players,
            "task": {
                "name": self.task.name, "kind": self.task.kind,
                "labels": list(self.task.labels), "dim": self.task.dim,
                "link": self.task.link, "available": self.task.available.value,
            },
            "features": [
                {
                    "name": f.name, "modality": f.modality.value, "scope": f.scope.value,
                    "width": f.width, "steps": f.steps, "vocab_size": f.vocab_size,
                    "mask": f.mask, "available": f.available.value,
                    "available_at": f.available_at, "reduce": f.reduce,
                    "transform": f.transform, "group": f.group,
                    "description": f.description,
                }
                for f in self.features
            ],
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=kwargs.pop("indent", 2), **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSchema":
        task = OutputSpec(
            name=data["task"]["name"], kind=data["task"]["kind"],
            labels=tuple(data["task"].get("labels", ())), dim=data["task"].get("dim", 1),
            link=data["task"].get("link", "sigmoid"),
            available=Availability(data["task"].get("available", "post_match")),
        )
        features = tuple(
            FeatureSpec(
                name=f["name"], modality=Modality(f["modality"]), scope=Scope(f["scope"]),
                width=f["width"], steps=f.get("steps", 1), vocab_size=f.get("vocab_size", 0),
                mask=f.get("mask", False),
                available=Availability(f.get("available", "pre_match")),
                available_at=f.get("available_at"), reduce=f.get("reduce", "sum"),
                transform=f.get("transform", "identity"), group=f.get("group", ""),
                description=f.get("description", ""),
            )
            for f in data["features"]
        )
        return cls(data["name"], data.get("sides", 2), data.get("players", 5), features, task)

    def cells(self) -> dict[tuple[str, str], list[str]]:
        """Design matrix: which features occupy each (modality, scope) cell."""
        result: dict[tuple[str, str], list[str]] = {}
        for f in self.features:
            result.setdefault(f.key, []).append(f.name)
        return result

    def validate(self, batch: "FeatureBatch", *, strict_time: bool = True) -> dict:
        return batch.validate(strict_time=strict_time)


# --------------------------------------------------------------------------
# Runtime batch
# --------------------------------------------------------------------------

@dataclass
class FeatureBatch:
    """Canonical tensors plus optional masks, timestamps and targets."""

    schema: FeatureSchema
    fields: dict[str, object]                       # canonical [B,S,N,T,C]
    weights: dict[str, object] = field(default_factory=dict)
    masks: dict[str, object] = field(default_factory=dict)
    available_at: dict[str, object] = field(default_factory=dict)
    prediction_time: object | None = None
    targets: object | None = None
    batch_size: int = 0

    @classmethod
    def from_natural(cls, schema: FeatureSchema, *, prediction_time=None,
                     targets=None, available_at=None, **raw) -> "FeatureBatch":
        """Lift natural-shaped inputs into the canonical 5-D layout.

        `raw` maps feature names to arrays, or to (indices, weights) for
        COEFFICIENT fields. A trailing channel axis of size 1 may be omitted.
        """
        if not raw:
            raise ValueError("No feature arrays supplied")
        sample = next(v for k, v in raw.items() if not k.endswith("__valid"))
        sample = sample[0] if isinstance(sample, tuple) else sample
        batch = int(as_array(sample).shape[0])
        fields, weights, masks = {}, {}, {}
        for spec in schema.features:
            value = raw.get(spec.name)
            if value is None:
                raise ValueError(f"Missing feature array: {spec.name}")
            if isinstance(value, tuple):
                index_raw, weight_raw = value
                weights[spec.name] = cls._lift(schema, spec, weight_raw, batch)
            else:
                index_raw = value
                if spec.modality is Modality.COEFFICIENT:
                    weights[spec.name] = _array(schema.canonical_shape(spec, batch), 1.0)
            fields[spec.name] = cls._lift(schema, spec, index_raw, batch)
            if spec.mask:
                mask_key = f"{spec.name}__valid"
                if mask_key not in raw:
                    raise ValueError(f"{spec.name} declares a mask; supply {mask_key}")
                masks[spec.name] = cls._lift_mask(
                    raw[mask_key], (batch, *schema.canonical_shape(spec, batch)[1:4])
                )
        return cls(
            schema=schema, fields=fields, weights=weights, masks=masks,
            available_at=dict(available_at or {}),
            prediction_time=as_array(prediction_time) if prediction_time is not None else None,
            targets=as_array(targets) if targets is not None else None,
            batch_size=batch,
        )

    @staticmethod
    def _lift(schema, spec, raw, batch):
        arr = as_array(raw)
        natural = schema.natural_shape(spec, batch)
        if tuple(arr.shape) == natural[:-1] and natural[-1] == 1:
            arr = _append_axis(arr)
        if tuple(arr.shape) != natural:
            raise ValueError(
                f"{spec.name}: expected natural shape {natural}, got {tuple(arr.shape)}. "
                f"scope={spec.scope.value}, modality={spec.modality.value}"
            )
        axes = []
        if spec.scope in (Scope.TEAM, Scope.PLAYER):
            axes.append(1)                                   # S
        if spec.scope is Scope.PLAYER:
            axes.append(2)                                   # N
        if spec.modality is Modality.SEQUENCE:
            axes.append(3)                                   # T
        return _to_canonical(arr, natural, schema.canonical_shape(spec, batch), axes)

    @staticmethod
    def _lift_mask(raw, shape):
        arr = as_array(raw)
        if tuple(arr.shape) != shape:
            raise ValueError(f"mask expects {shape}, got {tuple(arr.shape)}")
        return arr

    # -- validation --------------------------------------------------------

    def validate(self, *, strict_time: bool = True) -> dict:
        schema, batch = self.schema, self.batch_size
        problems = []
        for spec in schema.features:
            arr = self.fields[spec.name]
            expected = schema.canonical_shape(spec, batch)
            if tuple(arr.shape) != expected:
                problems.append(f"{spec.name}: shape {tuple(arr.shape)} != {expected}")
                continue
            values = _values(arr)
            if spec.modality is Modality.COEFFICIENT:
                vocab = spec.vocab_size
                if any(not float(v).is_integer() or not (0 <= v < vocab) for v in values):
                    problems.append(f"{spec.name}: coefficient indices must be ints in [0,{vocab})")
                w = self.weights.get(spec.name)
                if w is None or tuple(w.shape) != expected or not _all_finite(_values(w)):
                    problems.append(f"{spec.name}: coefficient weights missing or nonfinite")
            elif not _all_finite(values):
                problems.append(f"{spec.name}: nonfinite values")
            if spec.mask:
                m = self.masks.get(spec.name)
                if m is None or tuple(m.shape) != expected[:4]:
                    problems.append(f"{spec.name}: mask not aligned with [B,S,N,T]")
                elif not any(bool(v) for v in _values(m)):
                    problems.append(f"{spec.name}: mask has no observed steps")
            if strict_time and spec.available is Availability.POST_MATCH:
                problems.append(
                    f"{spec.name}: POST_MATCH feature entered a predictor batch "
                    f"(use an audit batch or re-declare availability)"
                )
        # temporal leakage: strict '<' mirrors the framework's replay contract
        if self.prediction_time is not None:
            pt = _values(self.prediction_time)
            for name, times in self.available_at.items():
                spec = next((f for f in schema.features if f.name == name), None)
                if spec is None:
                    problems.append(f"available_at refers to unknown feature {name}")
                    continue
                mask = self.masks.get(name)
                ts = _values(as_array(times))
                observed = [bool(v) for v in _values(mask)] if mask is not None else [True] * len(ts)
                per_row = math.prod(schema.canonical_shape(spec, batch)[1:4])
                for i, (t, ok) in enumerate(zip(ts, observed)):
                    row = i // per_row
                    if row >= len(pt):
                        break
                    if ok and t >= pt[row]:
                        problems.append(
                            f"{name}: feedback at {t} is not strictly earlier than "
                            f"prediction_time {pt[row]} (row {row})"
                        )
                        break
        if self.targets is not None and self.schema.task.kind not in {"regression", "assignment"}:
            dim = self.schema.task.dim
            t = as_array(self.targets)
            if not (len(t.shape) == 1 or (len(t.shape) == 2 and t.shape[1] == dim)):
                problems.append(f"targets do not match OutputSpec(dim={dim})")
        if problems:
            raise ValueError("Invalid FeatureBatch:\n  - " + "\n  - ".join(problems))
        return {
            "track": schema.name,
            "batch": batch,
            "features": len(schema.features),
            "cells": {f"{k[0]}/{k[1]}": v for k, v in schema.cells().items()},
            "canonical_shapes": {
                f.name: tuple(self.fields[f.name].shape) for f in schema.features
            },
            "temporal_evidence_checked": self.prediction_time is not None,
            "post_match_features": [
                f.name for f in schema.features
                if f.available is Availability.POST_MATCH
            ],
        }


def _values(arr):
    return list(arr.flat) if hasattr(arr, "flat") else list(arr)


def _all_finite(values) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def _append_axis(arr):
    shape = tuple(arr.shape)
    if _np is not None and isinstance(arr, _np.ndarray):
        return arr.reshape(*shape, 1)
    return _LiteArray(list(arr.flat), (*shape, 1))


def _to_canonical(arr, natural, target, axes):
    """Insert singleton axes so [B,(S),(N),(T),C] becomes [B,S,N,T,C].

    `axes` says which canonical axes the natural mid-axes actually occupy;
    e.g. a TEAM sequence has mids (S,T) at axes (1,3), leaving N = 1.
    """
    padded = [natural[0], 1, 1, 1, natural[-1]]
    for axis, size in zip(axes, natural[1:-1]):
        padded[axis] = size
    if tuple(arr.shape) == tuple(target):
        return arr
    flat = list(arr.flat) if hasattr(arr, "flat") else list(arr.reshape(-1))
    out = []
    for coords in itertools.product(*(range(d) for d in target)):
        src = 0
        for c, size in zip(coords, padded):
            src = src * size + (c if size > 1 else 0)
        out.append(flat[src])
    if _np is not None and isinstance(arr, _np.ndarray):
        return _np.asarray(out, dtype=arr.dtype).reshape(target)
    return _LiteArray(out, target)


# --------------------------------------------------------------------------
# Encoding: everything becomes one token tensor
# --------------------------------------------------------------------------

@dataclass
class ModelInputs:
    """The single interface every downstream model consumes."""

    tokens: object          # [B,S,N,D] per-player tokens
    valid: object           # [B,S,N] bool
    match_context: object   # [B,D]
    team_context: object    # [B,S,D]
    schema: FeatureSchema


class FeatureEncoder:
    """Deterministic reference encoder; swap for torch.nn.Module in Shuach.

    COEFFICIENT -> embedding-bag over C with sum/mean/max reduce
    DENSE       -> linear projection of C
    SEQUENCE    -> masked mean over T
    MATCH/TEAM contributions broadcast to player tokens. Modality+scope type
    bias keeps the fusion informed about which cell each feature came from.
    """

    def __init__(self, schema: FeatureSchema, dim: int = 16, seed: int = 0):
        rng = random.Random(seed)
        self.schema, self.dim = schema, dim
        self.tables, self.projections, self.biases = {}, {}, {}
        for spec in schema.features:
            scale = 1.0 / math.sqrt(dim)
            if spec.modality is Modality.COEFFICIENT:
                self.tables[spec.name] = [
                    [rng.gauss(0, scale) for _ in range(dim)]
                    for _ in range(spec.vocab_size)
                ]
                self.tables[spec.name][0] = [0.0] * dim      # unknown / padding
            else:
                self.projections[spec.name] = [
                    [rng.gauss(0, 1 / math.sqrt(spec.width)) for _ in range(dim)]
                    for _ in range(spec.width)
                ]
            cell = (spec.modality.value, spec.scope.value)
            self.biases[spec.name] = [
                rng.gauss(0, 0.02) + 0.01 * (hash(cell) % 7) for _ in range(dim)
            ]

    def __call__(self, batch: FeatureBatch) -> ModelInputs:
        B, S, N, D = batch.batch_size, self.schema.sides, self.schema.players, self.dim
        tokens = _array((B, S, N, D), 0.0)
        context = _array((B, D), 0.0)
        for spec in self.schema.features:
            local = self._encode_spec(spec, batch)          # [B,S_eff,N_eff,D]
            for b, s, n in itertools.product(range(B), range(S), range(N)):
                ls = s if local.shape[1] > 1 else 0
                ln = n if local.shape[2] > 1 else 0
                for d in range(D):
                    _set(tokens, (b, s, n, d),
                         _get(tokens, (b, s, n, d)) + _get(local, (b, ls, ln, d)))
            if spec.scope is Scope.MATCH:
                for b in range(B):
                    for d in range(D):
                        _set(context, (b, d), _get(context, (b, d)) + _get(local, (b, 0, 0, d)))
        valid = _array((B, S, N), 1, dtype=bool if _np is not None else None)
        for spec in self.schema.features:
            if not spec.mask or spec.scope is not Scope.PLAYER:
                continue
            m = batch.masks[spec.name]
            for b, s, n in itertools.product(range(B), range(S), range(N)):
                if not any(bool(_get(m, (b, s, n, t))) for t in range(m.shape[3])):
                    _set(valid, (b, s, n), False)
        team = _array((B, S, D), 0.0)
        for b, s in itertools.product(range(B), range(S)):
            for d in range(D):
                total = sum(_get(tokens, (b, s, n, d)) for n in range(N))
                _set(team, (b, s, d), total / N)
        return ModelInputs(tokens, valid, context, team, self.schema)

    def _encode_spec(self, spec, batch):
        B, D = batch.batch_size, self.dim
        shape = batch.fields[spec.name].shape
        _, S, N, T, C = shape
        out = _array((B, S, N, D), 0.0)
        arr = batch.fields[spec.name]
        wgt = batch.weights.get(spec.name)
        mask = batch.masks.get(spec.name)
        table = self.tables.get(spec.name)
        proj = self.projections.get(spec.name)
        for b, s, n in itertools.product(range(B), range(S), range(N)):
            acc = [0.0] * D
            count = 0
            for t in range(T):
                if mask is not None and not bool(_get(mask, (b, s, n, t))):
                    continue
                count += 1
                for c in range(C):
                    value = float(_get(arr, (b, s, n, t, c)))
                    if spec.modality is Modality.COEFFICIENT:
                        row = table[int(value)]
                        w = float(_get(wgt, (b, s, n, t, c))) if wgt is not None else 1.0
                        for d in range(D):
                            acc[d] += row[d] * w
                    else:
                        for d in range(D):
                            acc[d] += value * proj[c][d]
            if count and (spec.reduce == "mean" or spec.modality is Modality.SEQUENCE):
                acc = [v / count for v in acc]
            bias = self.biases[spec.name]
            for d in range(D):
                _set(out, (b, s, n, d), acc[d] + bias[d])
        return out


def _get(arr, key):
    return float(arr[key])


def _set(arr, key, value):
    if _np is not None and isinstance(arr, _np.ndarray):
        arr[key] = value
    else:
        idx = 0
        for k, size in zip(key, arr.shape):
            idx = idx * size + k
        arr.flat[idx] = value


# --------------------------------------------------------------------------
# Coverage proof: the seven existing Shuach tracks expressed as schemas
# --------------------------------------------------------------------------

def _schema(name, sides, players, task, features):
    return FeatureSchema(name, sides, players, tuple(features), task)


SCHEMAS: dict[str, FeatureSchema] = {
    # docs/data-contracts.md row 1: [B,2,5] hero IDs -> count vector / embeddings
    "hero_binary": _schema(
        "hero_binary", 2, 5, OutputSpec("left_win", "binary", labels=("loss", "win")),
        [FeatureSpec("hero_id", Modality.COEFFICIENT, Scope.PLAYER, 1,
                     vocab_size=114, reduce="sum", group="roster",
                     description="signed -1/0/1 indicators and hero_counts are reductions of this")],
    ),
    # row 2: Overwatch ratings, 6v6 WDL
    "overwatch_rating": _schema(
        "overwatch_rating", 2, 6,
        OutputSpec("wdl", "multiclass", labels=("loss", "draw", "win"), dim=3, link="softmax"),
        [
            FeatureSpec("rating_state", Modality.DENSE, Scope.PLAYER, 2, group="rating",
                        description="Elo/TrueSkill/OpenSkill mu,sigma indexed by roster"),
            FeatureSpec("calibration_terms", Modality.DENSE, Scope.MATCH, 4, group="rating",
                        description="s, abs(s), target mode d, s*d"),
        ],
    ),
    # row 3: the existing HistoryOutcomeSpec(5, 13, (6,6), 10, ...)
    "overwatch_history": _schema(
        "overwatch_history", 2, 6,
        OutputSpec("wdl", "multiclass", labels=("loss", "draw", "win"), dim=3, link="softmax"),
        [
            FeatureSpec("history", Modality.SEQUENCE, Scope.PLAYER, 5, steps=10, mask=True,
                        available_at="history_available_at", group="history",
                        description="10 right-aligned steps x 5 fields, masked"),
            FeatureSpec("static", Modality.DENSE, Scope.PLAYER, 13, group="summary",
                        description="3 scopes x 4 summaries + target mode"),
            FeatureSpec("target_mode", Modality.COEFFICIENT, Scope.MATCH, 1, vocab_size=2,
                        group="context"),
        ],
    ),
    # row 4: DBD predator, single side, ordinal 0..4 kills
    "dbd_predator": _schema(
        "dbd_predator", 1, 1,
        OutputSpec("kills", "ordinal", labels=("0", "1", "2", "3", "4"), dim=5, link="softmax"),
        [
            FeatureSpec("environment_id", Modality.COEFFICIENT, Scope.MATCH, 1, vocab_size=16,
                        group="context"),
            FeatureSpec("avatar_id", Modality.COEFFICIENT, Scope.MATCH, 1, vocab_size=32,
                        group="context"),
            FeatureSpec("history_numeric", Modality.DENSE, Scope.MATCH, 9, group="history",
                        description="9 smoothed history numerics"),
        ],
    ),
    # row 5: Dota snapshots are POST-GAME -- Availability makes that explicit
    "dota_future_rating": _schema(
        "dota_future_rating", 1, 1,
        OutputSpec("rating_residual", "regression", dim=1, link="identity"),
        [
            FeatureSpec("snapshots", Modality.SEQUENCE, Scope.PLAYER, 6, steps=12, mask=True,
                        available=Availability.POST_MATCH, group="snapshot",
                        description="12 three-minute snapshots; not valid pre-match"),
            FeatureSpec("lineups", Modality.COEFFICIENT, Scope.TEAM, 5, vocab_size=114,
                        reduce="sum", group="roster"),
            FeatureSpec("current_mean", Modality.DENSE, Scope.MATCH, 1, group="rating"),
        ],
    ),
    # row 6: synthetic worlds
    "synthetic_world": _schema(
        "synthetic_world", 2, 5, OutputSpec("left_win", "binary"),
        [FeatureSpec("participant_id", Modality.COEFFICIENT, Scope.PLAYER, 1,
                     vocab_size=101, reduce="sum", group="roster")],
    ),
    # row 7: assignment decision (output is an assignment, not a probability)
    "decision_assignment": _schema(
        "decision_assignment", 2, 5,
        OutputSpec("assignment", "assignment", dim=1, link="none"),
        [FeatureSpec("lobby", Modality.COEFFICIENT, Scope.MATCH, 10, vocab_size=101,
                     reduce="sum", group="roster",
                     description="10 participant IDs before teams are chosen")],
    ),
}


# league-history is a separate 5v5, single-mode driver with the same 5/13
# feature widths but a binary head and no target-domain identifier.
SCHEMAS["league_history"] = _schema(
    "league_history", 2, 5, OutputSpec("left_win", "binary"),
    [
        FeatureSpec("history", Modality.SEQUENCE, Scope.PLAYER, 5, steps=10, mask=True,
                    available_at="history_available_at", group="history"),
        FeatureSpec("static", Modality.DENSE, Scope.PLAYER, 13, group="summary"),
    ],
)


def predictor_schema(schema: FeatureSchema) -> FeatureSchema:
    """Drop POST_MATCH fields -- the only schema a predictor is allowed to see."""
    kept = tuple(f for f in schema.features if f.available is Availability.PRE_MATCH)
    return replace(schema, name=schema.name + "@predictor", features=kept)


def sample_inputs(schema: FeatureSchema, batch: int = 4, *, seed: int = 0,
                  prediction_time: float = 1_700_000_000.0, with_targets: bool = True):
    """Build a complete ``FeatureBatch`` of random data for any schema.

    This is the fastest way to go from a declared contract to a real tensor:
    every array is shaped from the schema itself, masks and coefficient slots
    are filled consistently, and POST_MATCH fields become the targets while
    PRE_MATCH fields keep a strictly earlier ``available_at``. Useful for smoke
    tests, tutorials and checking a new adapter before wiring real data.

    Args:
        schema: The contract to sample from.
        batch: Number of rows.
        seed: Seed for reproducible samples.
        prediction_time: Timestamp used for the availability check.
        with_targets: Whether to expose the POST_MATCH fields as targets.

    Returns:
        A validated-ready ``FeatureBatch``.
    """
    if _np is None:
        raise RuntimeError("sample_inputs needs numpy (included in the base install)")
    rng = _np.random.default_rng(seed)
    raw, available_at = {}, {}
    for spec in schema.features:
        shape = schema.natural_shape(spec, batch)
        if spec.modality is Modality.COEFFICIENT:
            values = rng.integers(1, max(2, spec.vocab_size), size=shape)
            raw[spec.name] = values
        else:
            raw[spec.name] = rng.normal(size=shape)
        if spec.mask:
            raw[f"{spec.name}__valid"] = rng.random(shape[:-1]) < 0.85
        if spec.available_at and spec.available is Availability.PRE_MATCH:
            available_at[spec.name] = _np.full(shape[:-1], prediction_time - 86_400.0)

    targets = None
    post = [f for f in schema.features if f.available is Availability.POST_MATCH]
    if with_targets and post:
        spec = post[0]
        shape = schema.natural_shape(spec, batch)
        classes = max(2, schema.task.dim)
        values = rng.integers(0, classes, size=shape)
        targets = values[..., 0] if shape[-1] == 1 else values

    return FeatureBatch.from_natural(
        schema,
        prediction_time=prediction_time,
        available_at=available_at or None,
        targets=targets,
        **raw,
    )
