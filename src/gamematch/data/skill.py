# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Post-game snapshots for a future observed-rating proxy, not true novice skill."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .heroes import file_hash


@dataclass
class SkillData:
    snapshots: np.ndarray
    valid: np.ndarray
    lineups: np.ndarray
    current_mu: np.ndarray
    target_mu: np.ndarray
    time: np.ndarray
    target_available_at: np.ndarray
    player_id: np.ndarray
    match_id: np.ndarray
    row_id: np.ndarray
    low_history: np.ndarray
    manifest: dict


def _indices(values, count, name):
    values = np.asarray(values)
    if (
        values.ndim != 1
        or not np.issubdtype(values.dtype, np.number)
        or not np.isfinite(values).all()
        or np.any(values != np.floor(values))
        or np.any((values < 0) | (values >= count))
    ):
        raise ValueError(f"{name} must contain finite integer indices in [0, {count})")
    return values.astype(np.int64)


def load_skill(workspace: Path):
    reference = workspace / "datasets/derived/dota2-reference-v2"
    metadata = json.loads((reference / "manifest.json").read_text())
    if metadata["status"] != "completed":
        raise ValueError("Complete the temporally audited reference ratings first")
    for name in ["skill-examples.npz", "main-rosters.npz"]:
        if file_hash(reference / name) != metadata[name + "_sha256"]:
            raise ValueError("Reference rating artifact checksum changed")
    with np.load(reference / "skill-examples.npz", allow_pickle=False) as handle:
        examples = dict(handle)
    with np.load(reference / "main-rosters.npz", allow_pickle=False) as handle:
        matches = dict(handle)
    count = len(matches["match_id"])
    if not np.array_equal(matches["match_id"], np.arange(count)):
        raise ValueError("Unexpected main match ID layout")
    example_matches = _indices(examples["match_id"], count, "Reference match_id")
    slots = [0, 1, 2, 3, 4, 128, 129, 130, 131, 132]
    example_slots = _indices(examples["player_slot"], 133, "Reference player_slot")
    if not np.isin(example_slots, slots).all():
        raise ValueError("Reference player_slot must identify one of the ten players")
    raw = workspace / "datasets/extracted/dota2-matches-v2/player_time.csv"
    identity = {
        "source_sha256": file_hash(raw),
        "reference_examples_sha256": metadata["skill-examples.npz_sha256"],
        "reference_rosters_sha256": metadata["main-rosters.npz_sha256"],
        "builder_sha256": file_hash(Path(__file__)),
        "snapshot_seconds": list(range(180, 2161, 180)),
    }
    # Keep each identity separate, including the legacy snapshots.npz used by
    # frozen experiments. Updating a builder/input must not overwrite evidence.
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = reference / "snapshot-cache" / key
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / "snapshots.npz"
    marker = directory / "manifest.json"
    if (
        cached.exists()
        and marker.exists()
        and json.loads(marker.read_text())["identity"] == identity
    ):
        if file_hash(cached) != json.loads(marker.read_text())["cached_sha256"]:
            raise ValueError("Snapshot cache checksum changed; remove the cache to rebuild it")
        with np.load(cached, allow_pickle=False) as handle:
            arrays = dict(handle)
    else:
        snapshots = np.zeros((count, 10, 12, 3), dtype=np.float32)
        valid = np.zeros((count, 12), dtype=bool)
        columns = [f"{feature}_{slot}" for slot in slots for feature in ["gold_t", "lh_t", "xp_t"]]
        with pd.read_csv(raw, chunksize=100000, usecols=["match_id", "times", *columns]) as reader:
            for chunk in reader:
                selected = chunk[
                    (chunk.times >= 180) & (chunk.times <= 2160) & (chunk.times % 180 == 0)
                ]
                mid = _indices(selected.match_id.to_numpy(), count, "Snapshot match_id")
                selected = selected[selected.times.to_numpy() <= matches["duration"][mid]]
                mid = selected.match_id.to_numpy(np.int64)
                step = selected.times.to_numpy(np.int64) // 180 - 1
                if len(set(zip(mid.tolist(), step.tolist()))) != len(mid) or valid[mid, step].any():
                    raise ValueError("Duplicate match snapshot time")
                values = selected[columns].to_numpy(np.float32).reshape(-1, 10, 3)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("Invalid snapshot cumulative performance features")
                snapshots[mid, :, step, :] = np.log1p(values)
                valid[mid, step] = True
        mid, slot = example_matches, example_slots
        player_index = np.where(slot < 128, slot, slot - 128 + 5)
        observed = valid[mid].any(1)
        mid, player_index = mid[observed], player_index[observed]
        examples = {k: v[observed] for k, v in examples.items()}
        x = snapshots[mid, player_index]
        extra = np.column_stack(
            [
                examples["current_mu"],
                examples["current_sigma"],
                np.log1p(examples["historical_count"] + examples["main_observed_count"]),
            ]
        ).astype(np.float32)
        x = np.concatenate([x, np.broadcast_to(extra[:, None], (len(x), 12, 3))], -1)
        mask = valid[mid]
        x[~mask] = 0
        lineups = matches["heroes"][mid].copy()
        opponent_first = examples["player_slot"] >= 128
        lineups[opponent_first] = lineups[opponent_first, ::-1, :]
        arrays = {
            "snapshots": x,
            "valid": mask,
            "lineups": lineups,
            "current_mu": examples["current_mu"],
            "target_mu": examples["future_mu"],
            "time": examples["completed_at"],
            "target_available_at": examples["target_available_at"],
            "player_id": examples["account_id"].astype(np.int64),
            "match_id": mid,
            "row_id": np.asarray([f"{m}:{s}" for m, s in zip(mid, examples["player_slot"])]),
            "low_history": (examples["historical_count"] <= 20)
            & (examples["main_observed_count"] <= 3),
        }
        np.savez_compressed(cached, **arrays)
        marker.write_text(
            json.dumps(
                {
                    "identity": identity,
                    "cached_sha256": file_hash(cached),
                    "cache_file": str(cached.relative_to(reference)),
                    "examples": len(x),
                    "zero_snapshot_examples_excluded": int((~observed).sum()),
                },
                indent=2,
            )
            + "\n"
        )
    manifest = {
        "reference": metadata,
        "snapshots": json.loads(marker.read_text()),
        "feature_names": [
            "log1p_gold",
            "log1p_last_hits",
            "log1p_experience",
            "current_reference_mu",
            "current_reference_sigma",
            "log1p_observed_matches",
        ],
        "scope": "QuickSkill-inspired future skill-state regression after a completed game; six public features, not original 115-feature novice task",
        "label": "OpenSkill mean after five additional observed games, including uncertainty and missing-observation effects",
        "selection_limit": "Only returning observed accounts with five future records have labels; right-censored and anonymous accounts excluded",
        "novice_limit": "low_history is a limited-observed-history subgroup, not verified account novelty",
    }
    return SkillData(**arrays, manifest=manifest), matches


def skill_splits(data, match_metadata):
    start = np.sort(match_metadata["start_time"][match_metadata["valid"]])
    bounds = [float(start[int(len(start) * 0.7)]), float(start[int(len(start) * 0.85)])]
    train = (data.time < bounds[0]) & (data.target_available_at < bounds[0])
    valid = (
        (data.time >= bounds[0]) & (data.time < bounds[1]) & (data.target_available_at < bounds[1])
    )
    test = data.time >= bounds[1]
    if not np.all(data.target_available_at > data.time):
        raise ValueError("Future-rating target must be strictly later than the input game")
    if min(train.sum(), valid.sum(), test.sum()) < 20:
        raise ValueError("Insufficient uncensored data after purging target-time leakage")
    return {"train": train, "valid": valid, "test": test}, bounds
