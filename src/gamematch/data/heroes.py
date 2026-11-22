# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Static hero-roster adapters. These records contain no player identity.

Pilot experiments use only an inner split of the reserved training partition.
The official UCI test file is never opened in pilot mode. Identical lineups,
including within-team permutations and reversed sides, share a split group.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {
    "neuralac-dota2018": "extracted/NAC-main/data/dota2018.csv",
    "neuralac-lol": "extracted/neuralac-lol/lol_75W.csv",
    "uci-dota2": "extracted/uci-dota2/dota2Train.csv",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class HeroRecords:
    """Keep roster rows, labels and split identities aligned.

    Attributes:
        rosters: Raw hero IDs [matches, 2, 5], with the positive side first.
        labels: One binary outcome per match.
        row_ids: Source-row identifiers used to align saved predictions.
        groups: Split-group IDs shared by permutations and side mirrors.
    """
    rosters: np.ndarray
    labels: np.ndarray
    row_ids: np.ndarray
    groups: np.ndarray

    def take(self, indices: np.ndarray) -> "HeroRecords":
        """Select the same rows from all four fields using NumPy indexing."""
        return HeroRecords(
            *(x[indices] for x in (self.rosters, self.labels, self.row_ids, self.groups))
        )

    def __len__(self) -> int:
        return len(self.labels)


def group_ids(rosters: np.ndarray) -> np.ndarray:
    """Hash each pair of unordered lineups so side mirrors share a split group."""
    result = []
    for roster in rosters:
        a, b = sorted((tuple(sorted(roster[0].tolist())), tuple(sorted(roster[1].tolist()))))
        key = repr((a, b)).encode()
        result.append(hashlib.sha256(key).hexdigest()[:24])
    return np.asarray(result)


def group_uniform(groups: np.ndarray, salt: str) -> np.ndarray:
    return np.fromiter(
        (
            int.from_bytes(hashlib.sha256((salt + ":" + group).encode()).digest()[:8], "big")
            / 2**64
            for group in groups
        ),
        dtype=np.float64,
        count=len(groups),
    )


def read_records(path: Path, uci: bool = False) -> HeroRecords:
    if uci:
        values = pd.read_csv(path, header=None).to_numpy()
        if values.shape[1] != 117:
            raise ValueError("UCI expects outcome, cluster, mode, type and 113 heroes")
        if not np.isfinite(values).all() or not (values == np.floor(values)).all():
            raise ValueError("UCI fields must be finite integers; fractional values cannot be cast")
        values = values.astype(np.int64)
        signed = values[:, 4:]
        if not np.isin(signed, [-1, 0, 1]).all():
            raise ValueError("Invalid signed hero indicator")
        if not ((signed == 1).sum(1) == 5).all() or not ((signed == -1).sum(1) == 5).all():
            raise ValueError("Hero track requires two complete five-member teams")
        if not np.isin(values[:, 0], [-1, 1]).all():
            raise ValueError("Unexpected UCI outcome")
        left = np.where(signed == 1)[1].reshape(-1, 5) + 1
        right = np.where(signed == -1)[1].reshape(-1, 5) + 1
        rosters = np.stack([left, right], axis=1)
        labels = (values[:, 0] == 1).astype(np.float32)
    else:
        frame = pd.read_csv(path)
        columns = [f"hero{i}" for i in range(1, 11)]
        raw = frame[columns].to_numpy()
        if not np.isfinite(raw).all() or not (raw == np.floor(raw)).all():
            raise ValueError("Hero identifiers must be non-null integers")
        rosters = raw.astype(np.int64).reshape(-1, 2, 5)
        labels = frame["target"].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("The hero track accepts binary outcomes only")
    labels = labels.astype(np.float32)
    flat = np.sort(rosters.reshape(-1, 10), axis=1)
    if np.any(np.diff(flat, axis=1) == 0):
        raise ValueError("Repeated hero within a match")
    row_ids = np.asarray([f"{path.name}:{i + 1}" for i in range(len(labels))])
    return HeroRecords(rosters, labels, row_ids, group_ids(rosters))


class HeroVocabulary:
    """Encode raw hero IDs using a vocabulary fitted on the training partition."""

    def fit(self, rosters: np.ndarray) -> "HeroVocabulary":
        """Store the sorted unique identifiers and return this encoder."""
        self.values = np.unique(rosters)
        return self

    @property
    def size(self) -> int:
        """Return the number of embedding rows, including the unknown token."""
        return len(self.values) + 1

    def transform(self, rosters: np.ndarray) -> np.ndarray:
        """Return int64 indices with the input shape. Unseen heroes map to zero."""
        if len(self.values) == 0:
            return np.zeros(np.shape(rosters), dtype=np.int64)
        indices = np.searchsorted(self.values, rosters)
        clipped = np.minimum(indices, len(self.values) - 1)
        known = (indices < len(self.values)) & (self.values[clipped] == rosters)
        return np.where(known, indices + 1, 0).astype(np.int64)


def _cap(records: HeroRecords, maximum: int | None, seed: int) -> HeroRecords:
    if maximum is None or len(records) <= maximum:
        return records
    indices = np.sort(np.random.default_rng(seed).choice(len(records), maximum, replace=False))
    return records.take(indices)


def prepare(
    dataset: str,
    data_root: Path,
    protocol: str = "pilot",
    split_seed: int = 0,
    max_train_rows: int | None = None,
    max_eval_rows: int | None = None,
):
    """Build disjoint hero-roster splits and fit their training vocabulary.

    Args:
        dataset: One of the keys in SOURCES.
        data_root: Directory containing the extracted adapter inputs.
        protocol: "pilot" for an inner development split, or "final" for the
            reference evaluation. Only final UCI loads the official test file.
        split_seed: Salt for deterministic split-group allocation and row caps.
        max_train_rows: Optional cap applied after forming the training split.
        max_eval_rows: Optional cap applied separately to validation and test.

    Returns:
        A mapping of train, valid and test HeroRecords, the fitted HeroVocabulary,
        and a manifest recording sources, split counts, hashes and unknown tokens.
    """
    if dataset not in SOURCES:
        raise ValueError(f"Unsupported hero dataset: {dataset}")
    if protocol not in {"pilot", "final"}:
        raise ValueError("protocol must be pilot or final")
    path = data_root / SOURCES[dataset]
    records = read_records(path, uci=dataset == "uci-dota2")
    paths = [path]
    source_rows = len(records)
    u = group_uniform(records.groups, f"outer:{split_seed}")
    training_cut = 0.9 if dataset == "uci-dota2" else 0.8
    excluded_overlap = 0
    if protocol == "pilot":
        reserved = records.take(np.flatnonzero(u < training_cut))
        inner = group_uniform(reserved.groups, f"pilot:{split_seed}")
        splits = {
            "train": reserved.take(np.flatnonzero(inner < 0.8)),
            "valid": reserved.take(np.flatnonzero((inner >= 0.8) & (inner < 0.9))),
            "test": reserved.take(np.flatnonzero(inner >= 0.9)),
        }
        split_name = "pilot_inner_group_holdout; publication holdouts untouched"
    else:
        train = records.take(np.flatnonzero(u < training_cut))
        valid = records.take(
            np.flatnonzero(
                (u >= training_cut)
                & (np.ones(len(u), dtype=bool) if dataset == "uci-dota2" else u < 0.9)
            )
        )
        if dataset == "uci-dota2":
            test_path = path.with_name("dota2Test.csv")
            paths.append(test_path)
            test = read_records(test_path, uci=True)
            test_groups = set(test.groups.tolist())
            old_count = len(train) + len(valid)
            train = train.take(np.flatnonzero([g not in test_groups for g in train.groups]))
            valid = valid.take(np.flatnonzero([g not in test_groups for g in valid.groups]))
            excluded_overlap = old_count - len(train) - len(valid)
        else:
            test = records.take(np.flatnonzero(u >= 0.9))
        splits = {"train": train, "valid": valid, "test": test}
        split_name = (
            "official_test_group_purged" if dataset == "uci-dota2" else "static_group_80_10_10"
        )
    before_caps = {name: len(rows) for name, rows in splits.items()}
    splits = {
        name: _cap(rows, max_train_rows if name == "train" else max_eval_rows, split_seed + i + 100)
        for i, (name, rows) in enumerate(splits.items())
    }
    if min(map(len, splits.values())) == 0:
        raise ValueError("An empty split cannot be trained or evaluated")
    groups = {name: set(rows.groups.tolist()) for name, rows in splits.items()}
    if any(
        groups[a] & groups[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]
    ):
        raise ValueError("Roster group overlap across splits")
    vocab = HeroVocabulary().fit(splits["train"].rosters)
    manifest = {
        "dataset": dataset,
        "entity_kind": "hero",
        "task": "static_roster_binary_prediction",
        "information_set": "two_hero_rosters_only",
        "protocol": protocol,
        "split_name": split_name,
        "split_seed": split_seed,
        "source_rows": source_rows,
        "grouping": "sorted heroes in each team; canonicalized team order for splitting only",
        "sources": [{"path": str(p), "sha256": file_hash(p)} for p in paths],
        "rows_before_caps": before_caps,
        "rows": {n: len(r) for n, r in splits.items()},
        "excluded_training_validation_rows_overlapping_official_test": excluded_overlap,
        "unknown_token": 0,
        "vocabulary": vocab.values.tolist(),
        "unknown_tokens": {
            n: int((vocab.transform(r.rosters) == 0).sum()) for n, r in splits.items()
        },
        "label_semantics": "target==1 in NAC exports; outcome==+1 for UCI positive-indicator team",
        "limitations": [
            "No player identity or event times",
            "No counterfactual matchmaking evaluation",
            "Group split is a declared adaptation of authors' row-random protocol",
        ],
    }
    return splits, vocab, manifest
