# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Compact history references and batch-only expansion for large team data."""

import json
from pathlib import Path

import numpy as np

from .heroes import file_hash


def prior_observation_indices(players, starts, available_at, length=10):
    """Index strictly delivered prior appearances; zero is a missing sentinel.

    Returns one-based row references, full observed counts and last availability.
    The same participant can have overlapping appearances; completion order,
    rather than input order, determines which observations can be retrieved.
    """
    players = np.asarray(players)
    starts = np.asarray(starts, dtype=np.float64)
    available_at = np.asarray(available_at, dtype=np.float64)
    if (
        players.ndim != 1
        or starts.shape != players.shape
        or available_at.shape != players.shape
        or not len(players)
        or type(length) is not int
        or length < 1
    ):
        raise ValueError("Aligned nonempty vectors and a positive integer history length required")
    if not np.issubdtype(players.dtype, np.integer) or (players < 0).any():
        raise ValueError("Nonnegative integer player lookup codes required")
    if (
        not np.isfinite(starts).all()
        or not np.isfinite(available_at).all()
        or (available_at < starts).any()
    ):
        raise ValueError("Finite timestamps and nonnegative feedback delay required")
    if len(players) >= np.iinfo(np.int32).max:
        raise ValueError("History exceeds the int32 reference capacity")
    order = np.lexsort((np.arange(len(players)), available_at, players))
    ordered_players = players[order]
    boundaries = np.r_[
        0, np.flatnonzero(ordered_players[1:] != ordered_players[:-1]) + 1, len(players)
    ]
    indices = np.zeros((len(players), length), dtype=np.int32)
    counts = np.zeros(len(players), dtype=np.int32)
    latest = np.full(len(players), np.nan, dtype=np.float64)
    offsets = np.arange(-length, 0)
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        if end - begin == 1:
            continue
        rows = order[begin:end]
        delivered = available_at[rows]
        positions = np.searchsorted(delivered, starts[rows], side="left")
        counts[rows] = positions
        active = positions > 0
        latest[rows[active]] = delivered[positions[active] - 1]
        gather = positions[:, None] + offsets
        valid = gather >= 0
        selected = rows[np.maximum(gather, 0)] + 1
        indices[rows] = np.where(valid, selected, 0)
    return indices, counts, latest


class IndexedHistoryStore:
    """Memory-map persisted arrays, expanding only requested matches.

    The builder owns validation, normalization and file hashes in manifest.json.
    Player identities are used solely for historical lookup, not model inputs.
    """

    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        names = [
            "performance.npy",
            "history-index.npy",
            "appearance-available.npy",
            "match-start.npy",
            "static.npy",
            "target.npy",
            "row-id.npy",
            "train-rows.npy",
            "valid-rows.npy",
            "test-rows.npy",
        ]
        if set(self.manifest["files_sha256"]) != set(names):
            raise ValueError("Unexpected indexed-history artifact inventory")
        for name in names:
            if file_hash(directory / name) != self.manifest["files_sha256"][name]:
                raise ValueError("Indexed-history file checksum mismatch: " + name)
        self.performance = np.load(directory / "performance.npy", mmap_mode="r", allow_pickle=False)
        self.history = np.load(directory / "history-index.npy", mmap_mode="r", allow_pickle=False)
        self.available = np.load(
            directory / "appearance-available.npy", mmap_mode="r", allow_pickle=False
        )
        self.time = np.load(directory / "match-start.npy", mmap_mode="r", allow_pickle=False)
        self.static = np.load(directory / "static.npy", mmap_mode="r", allow_pickle=False)
        self.target = np.load(directory / "target.npy", mmap_mode="r", allow_pickle=False)
        self.row_id = np.load(directory / "row-id.npy", mmap_mode="r", allow_pickle=False)
        self.mean = np.asarray(self.manifest["normalizer"]["history_mean"], dtype=np.float32)
        self.scale = np.asarray(self.manifest["normalizer"]["history_scale"], dtype=np.float32)
        self.static_mean = np.asarray(self.manifest["normalizer"]["static_mean"], dtype=np.float32)
        self.static_scale = np.asarray(
            self.manifest["normalizer"]["static_scale"], dtype=np.float32
        )
        n = len(self.target)
        expected = (n, 2, self.manifest["team_size"], self.manifest["history_length"])
        if (
            self.history.shape != expected
            or self.static.shape[:3] != expected[:3]
            or self.time.shape != (n,)
            or self.row_id.shape != (n,)
            or self.available.shape != (len(self.performance),)
            or self.mean.shape != (self.performance.shape[-1] + 1,)
            or self.scale.shape != self.mean.shape
            or self.static_mean.shape != (self.static.shape[-1],)
            or self.static_scale.shape != self.static_mean.shape
        ):
            raise ValueError("Inconsistent indexed-history tensor shapes")
        if (
            not np.isfinite(np.r_[self.mean, self.scale, self.static_mean, self.static_scale]).all()
            or (self.scale <= 0).any()
            or (self.static_scale <= 0).any()
        ):
            raise ValueError("Invalid train-fitted normalizer")
        if (
            not np.issubdtype(self.history.dtype, np.integer)
            or (self.history < 0).any()
            or (self.history >= len(self.performance)).any()
        ):
            raise ValueError("History reference outside performance storage")

    def batch(self, rows, performance_columns=None):
        rows = np.asarray(rows)
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("Batch rows must be a vector of integer indices")
        if ((rows < 0) | (rows >= len(self.target))).any():
            raise ValueError("Batch index outside the stored match range")
        references = self.history[rows]
        valid = references != 0
        times = self.time[rows, None, None, None]
        if np.any(valid & (self.available[references] >= times)):
            raise ValueError("History contains feedback unavailable at prediction time")
        age = np.maximum(times - self.available[references], 0)
        age_feature = np.log1p(age / 86400)
        if performance_columns is None:
            raw = self.performance[references]
            history = np.concatenate([raw, age_feature[..., None]], axis=-1)
            history = ((history - self.mean) / self.scale).astype(np.float32)
        else:
            # Select before gathering so small information sets do not expand
            # and normalize every performance field in the feature store.
            columns = np.arange(self.performance.shape[-1] + 1)[..., performance_columns]
            flat_columns = columns.reshape(-1)
            age_columns = flat_columns == self.performance.shape[-1]
            history = np.empty(
                (*references.shape, len(flat_columns)),
                dtype=np.result_type(self.performance.dtype, age_feature.dtype),
            )
            history[..., ~age_columns] = self.performance[
                references[..., None], flat_columns[~age_columns]
            ]
            history[..., age_columns] = age_feature[..., None]
            history = (
                (history - self.mean[flat_columns]) / self.scale[flat_columns]
            ).astype(np.float32)
            history = history.reshape((*references.shape, *columns.shape))
        history[~valid] = 0
        static = ((self.static[rows] - self.static_mean) / self.static_scale).astype(np.float32)
        return (history, static, valid, np.zeros(len(rows), dtype=np.int64)), np.asarray(
            self.target[rows]
        )
