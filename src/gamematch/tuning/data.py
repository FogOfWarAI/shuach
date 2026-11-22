# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Reuse the primary partitions and train-fitted transforms for tuning."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data.asymmetric import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_asymmetric
from ..data.heroes import HeroRecords, HeroVocabulary, file_hash, prepare
from ..data.history import history_split_masks, standardize_histories
from ..data.skill import load_skill, skill_splits
from ..history_runner import cached_history
from ..models.tabular import hero_counts


@dataclass
class Partition:
    inputs: tuple
    target: np.ndarray
    row_id: np.ndarray
    metadata: dict
    tabular: object = None


@dataclass
class Bundle:
    name: str
    task: str
    parts: dict
    specification: dict
    provenance: dict


def partition(arrays, target, ids, metadata, tabular=None):
    return Partition(
        tuple(torch.from_numpy(np.asarray(x)) for x in arrays),
        np.asarray(target),
        np.asarray(ids),
        metadata,
        tabular,
    )


def load_bundle(root: Path, name: str, world=0):
    if name in {"uci-dota2", "neuralac-dota2018", "neuralac-lol"} or name.startswith("synthetic-"):
        if name.startswith("synthetic-"):
            base = root / "runs/decisions-v1" / f"{name[10:]}-seed-{world}"
            with np.load(base / "data.npz", allow_pickle=False) as h:
                splits = {
                    p: HeroRecords(
                        *(h[f"{p}_{f}"] for f in ["rosters", "labels", "row_ids", "groups"])
                    )
                    for p in ["train", "valid", "test"]
                }
            vocab = HeroVocabulary().fit(splits["train"].rosters)
            provenance = {
                "cached_data": str(base / "data.npz"),
                "sha256": file_hash(base / "data.npz"),
                "world": world,
                "decision_environment": str(base / "decision-environment.npz"),
                "decision_environment_sha256": file_hash(base / "decision-environment.npz"),
            }
        else:
            splits, vocab, provenance = prepare(name, root / "datasets", "final", 0)
        parts = {}
        for p, d in splits.items():
            encoded = vocab.transform(d.rosters)
            parts[p] = partition(
                [encoded],
                d.labels,
                d.row_ids,
                {"group_id": d.groups},
                hero_counts(encoded, vocab.size),
            )
        return Bundle(
            name,
            "binary",
            parts,
            {"vocab_size": vocab.size, "vocabulary": vocab.values.tolist()},
            provenance,
        )
    if name.startswith("dbd-"):
        d = load_asymmetric(name, root / "datasets", 0)
        masks, bounds = history_split_masks(d, 0.7, 0.15)
        transform = ColumnTransformer(
            [
                ("numeric", StandardScaler(), NUMERIC_FEATURES),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        transform.fit(d.features[masks["train"]])
        parts = {}
        for p, m in masks.items():
            x = np.asarray(transform.transform(d.features[m]), dtype=np.float32)
            parts[p] = partition(
                [x],
                d.outcome[m],
                d.row_ids[m],
                {"player_id": d.player_ids[m], "time": d.time[m]},
                x,
            )
        return Bundle(
            name,
            "multiclass",
            parts,
            {"features": parts["train"].inputs[0].shape[-1], "classes": 5},
            {"data": d.manifest, "time_boundaries": bounds, "preprocessor": transform},
        )
    if name == "overwatch":
        cfg = json.loads((root / "benchmarks/local-overwatch-history.json").read_text())
        d, provenance = cached_history(root, cfg)
        masks, bounds = history_split_masks(d, 0.7, 0.15)
        normalizer = standardize_histories(d, masks["train"])
        parts = {
            p: partition(
                [d.history[m], d.static[m], d.valid[m], d.domain[m]],
                d.outcome[m],
                d.row_ids[m],
                {"time": d.time[m], "domain": d.domain[m]},
            )
            for p, m in masks.items()
        }
        spec = {
            "history_features": d.history.shape[-1],
            "static_features": d.static.shape[-1],
            "team_size": d.history.shape[2],
            "history_length": d.history.shape[3],
            "classes": 3,
        }
        return Bundle(
            name,
            "history",
            parts,
            spec,
            {"data": provenance, "time_boundaries": bounds, "normalizer": normalizer},
        )
    if name == "future-rating":
        d, matches = load_skill(root)
        masks, bounds = skill_splits(d, matches)
        observed = d.snapshots[masks["train"]][d.valid[masks["train"]]]
        mean, scale = observed.mean(0), observed.std(0).clip(1e-6)
        x = ((d.snapshots - mean) / scale).astype(np.float32)
        x[~d.valid] = 0
        delta = d.target_mu - d.current_mu
        mu, sigma = (
            float(delta[masks["train"]].mean()),
            float(delta[masks["train"]].std().clip(1e-6)),
        )
        normalized = ((delta - mu) / sigma).astype(np.float32)
        vocab = HeroVocabulary().fit(d.lineups[masks["train"]])
        rosters = vocab.transform(d.lineups)
        matrix = np.concatenate(
            [
                x.reshape(len(x), -1),
                d.valid.astype(np.float32),
                hero_counts(rosters, vocab.size).toarray(),
            ],
            1,
        )
        parts = {
            p: partition(
                [x[m], rosters[m], d.valid[m]],
                normalized[m],
                d.row_id[m],
                {
                    "current_mu": d.current_mu[m],
                    "y_true": d.target_mu[m],
                    "player_id": d.player_id[m],
                    "match_id": d.match_id[m],
                    "time": d.time[m],
                },
                matrix[m],
            )
            for p, m in masks.items()
        }
        return Bundle(
            name,
            "regression",
            parts,
            {
                "features": x.shape[-1],
                "hero_vocab": vocab.size,
                "delta_mean": mu,
                "delta_scale": sigma,
                "vocabulary": vocab.values.tolist(),
            },
            {
                "data": d.manifest,
                "time_boundaries": bounds,
                "normalizer": {"mean": mean.tolist(), "scale": scale.tolist()},
            },
        )
    raise ValueError(name)
