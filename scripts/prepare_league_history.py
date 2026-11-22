# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Prepare compact, strictly prior performance histories from verified LoL shards."""

import argparse
import gc
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from audit_league_data import IDENTITY, KEY, META, ROOT, validate_manifest

from gamematch.data.heroes import file_hash
from gamematch.data.indexed_history import IndexedHistoryStore, prior_observation_indices

OUT = ROOT / "datasets/derived/leagueData-v1"
PERFORMANCE = [
    "kills",
    "deaths",
    "assists",
    "goldEarned",
    "totalMinionsKilled",
    "neutralMinionsKilled",
    "totalDamageDealtToChampions",
    "totalDamageTaken",
    "visionScore",
    "wardsPlaced",
    "wardsKilled",
    "totalHeal",
    "totalDamageShieldedOnTeammates",
    "timeCCingOthers",
    "champExperience",
    "champLevel",
]
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY", "UNKNOWN"]
HISTORY_FIELDS = (
    ["previous_win", "previous_log_duration_minutes"]
    + ["previous_log1p_per_minute_" + f for f in PERFORMANCE]
    + ["previous_role_" + role for role in ROLES]
    + ["log1p_days_since_feedback"]
)
STATIC_FIELDS = [
    "log1p_completed",
    "log1p_days_since_feedback",
    "last10_mean_result",
    "last10_result_sd",
    "history_coverage",
    "no_completed_history",
]


def read(path):
    return json.loads(path.read_text())


def verify_existing(directory):
    """Verify immutable data against its producing sources, not today's source."""
    store = IndexedHistoryStore(directory)
    record = store.manifest
    identity = record["identity"]
    if file_hash(META / "manifest.json") != identity["download_manifest_sha256"]:
        raise ValueError("The original download manifest changed")
    producing = {
        "prepare_league_history.py": "driver_sha256",
        "audit_league_data.py": "manifest_validator_sha256",
        "indexed_history.py": "history_index_builder_sha256",
    }
    for name, key in producing.items():
        if file_hash(directory / "source-snapshot" / name) != identity[key]:
            raise ValueError("Producing source archive changed: " + name)
    if record["storage_bytes"] != sum(
        (directory / name).stat().st_size for name in record["files_sha256"]
    ):
        raise ValueError("Storage byte count changed")
    for part, count in record["split_counts"].items():
        rows = np.load(directory / f"{part}-rows.npy", allow_pickle=False)
        if len(rows) != count:
            raise ValueError("Split row count changed: " + part)
    print(
        json.dumps(
            {
                "status": "verified",
                "matches": len(store.target),
                "storage_bytes": record["storage_bytes"],
                "source_basis": "archived producing bytes; current source may differ",
            }
        ),
        flush=True,
    )


def main():
    import pyarrow.parquet as pq

    manifest = read(META / "manifest.json")
    paths = validate_manifest(manifest, read(META / "parquet-files.json"), ROOT)
    identity = {
        "download_manifest_sha256": file_hash(META / "manifest.json"),
        "driver_sha256": file_hash(Path(__file__)),
        "manifest_validator_sha256": file_hash(ROOT / "scripts/audit_league_data.py"),
        "history_index_builder_sha256": file_hash(ROOT / "src/gamematch/data/indexed_history.py"),
    }
    marker = OUT / "manifest.json"
    if marker.exists():
        saved = read(marker)
        if saved["identity"] != identity:
            raise ValueError("Prepared-data identity changed; use a new output version")
        for name, digest in saved["files_sha256"].items():
            if file_hash(OUT / name) != digest:
                raise ValueError("Prepared data changed: " + name)
        print("Already prepared and verified", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fields = IDENTITY + ["teamId", "win", "teamPosition"] + PERFORMANCE
    people, parts, raw_count = {}, [], 0
    for path in paths:
        part = pq.read_table(path, columns=fields).to_pandas()
        raw_count += len(part)
        if part[IDENTITY + ["teamId", "win"]].isna().any().any():
            raise ValueError("Required identity/target field is missing")
        # Stable file-order integer codes are lookup keys only, never model inputs.
        codes = []
        for value in part.puuid:
            if value not in people:
                people[value] = len(people)
            codes.append(people[value])
        part["puuid"] = np.asarray(codes, dtype=np.int32)
        part["teamPosition"] = part.teamPosition.where(
            part.teamPosition.isin(ROLES[:-1]), "UNKNOWN"
        )
        parts.append(part)
        print("READ", path.name, len(part), flush=True)
    data = pd.concat(parts, ignore_index=True)
    del parts, people, codes, part
    gc.collect()
    duplicate = data.duplicated(IDENTITY, keep=False)
    conflict = (
        data.loc[duplicate]
        .groupby(IDENTITY, sort=False)[["teamId", "win", "teamPosition"] + PERFORMANCE]
        .nunique(dropna=False)
    )
    bad_context = pd.MultiIndex.from_frame(
        conflict[(conflict > 1).any(axis=1)].reset_index()[KEY]
    ).unique()
    original = len(data)
    data.drop_duplicates(IDENTITY, inplace=True)
    duplicates_removed = original - len(data)
    invalid = (
        ~data.teamId.isin([100, 200])
        | ~data.participantId.isin(range(1, 11))
        | (data.gameDuration < 900)
        | ~data.gameStartTimestamp.between(1577836800000, 1735689600000)
    )
    numbers = data[PERFORMANCE].to_numpy(dtype=np.float64)
    invalid |= ~np.isfinite(numbers).all(1) | (numbers < 0).any(1)
    del numbers
    bad_context = bad_context.union(pd.MultiIndex.from_frame(data.loc[invalid, KEY]).unique())
    data = data.loc[~pd.MultiIndex.from_frame(data[KEY]).isin(bad_context)].copy()
    groups = data.groupby(KEY, sort=True)
    counts = groups.agg(
        rows=("puuid", "size"),
        people=("puuid", "nunique"),
        slots=("participantId", "nunique"),
        wins=("win", "sum"),
    )
    sides = data.groupby(KEY + ["teamId"]).agg(rows=("puuid", "size"), results=("win", "nunique"))
    complete_sides = ((sides.rows == 5) & (sides.results == 1)).groupby(level=KEY).sum()
    good = (
        (counts.rows == 10)
        & (counts.people == 10)
        & (counts.slots == 10)
        & (counts.wins == 5)
        & (complete_sides.reindex(counts.index, fill_value=0) == 2)
    )
    data = data.loc[pd.MultiIndex.from_frame(data[KEY]).isin(counts.index[good])].copy()
    data.sort_values(KEY + ["teamId", "participantId"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    del groups, conflict, counts, sides, complete_sides
    gc.collect()
    n = len(data) // 10
    if n == 0 or n * 10 != len(data):
        raise ValueError("No complete ten-person records")
    print("ADMITTED", n, "matches", flush=True)
    starts = data.gameStartTimestamp.to_numpy(dtype=np.float64) / 1000
    duration = data.gameDuration.to_numpy(dtype=np.float64)
    available = starts + duration
    refs, observed, last = prior_observation_indices(data.puuid.to_numpy(), starts, available, 10)
    np.save(OUT / "history-index.npy", refs.reshape(n, 2, 5, 10), allow_pickle=False)
    np.save(OUT / "appearance-available.npy", np.r_[0.0, available], allow_pickle=False)
    np.save(OUT / "match-start.npy", starts.reshape(n, 10)[:, 0], allow_pickle=False)
    target = data.win.to_numpy(dtype=np.int64).reshape(n, 10)[:, 0]
    np.save(OUT / "target.npy", target, allow_pickle=False)
    contexts = data.iloc[::10]
    row_ids = np.asarray(
        [
            f"leagueData:{int(t)}:{int(d)}:{v}"
            for t, d, v in contexts[KEY].itertuples(index=False, name=None)
        ]
    )
    if len(np.unique(row_ids)) != n:
        raise ValueError("Reconstructed context IDs are not unique")
    np.save(OUT / "row-id.npy", row_ids, allow_pickle=False)
    del contexts
    performance = np.lib.format.open_memmap(
        OUT / "performance.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(data) + 1, len(HISTORY_FIELDS) - 1),
    )
    performance[0] = 0
    performance[1:, 0] = data.win.to_numpy(dtype=np.float32)
    performance[1:, 1] = np.log1p(duration / 60)
    for j, field in enumerate(PERFORMANCE, 2):
        performance[1:, j] = np.log1p(data[field].to_numpy(dtype=float) / (duration / 60))
    for j, role in enumerate(ROLES, 2 + len(PERFORMANCE)):
        performance[1:, j] = (data.teamPosition == role).to_numpy(dtype=np.float32)
    performance.flush()
    static = np.lib.format.open_memmap(
        OUT / "static.npy", mode="w+", dtype=np.float32, shape=(n * 10, 6)
    )
    static[:, 0] = np.log1p(observed)
    static[:, 1] = np.log1p(np.where(observed > 0, starts - last, 0) / 86400)
    static[:, 4] = np.minimum(observed, 10) / 10
    static[:, 5] = observed == 0
    for begin in range(0, n * 10, 40960):
        end = min(begin + 40960, n * 10)
        valid = refs[begin:end] > 0
        win = performance[refs[begin:end], 0]
        denominator = np.maximum(valid.sum(1), 1)
        mean = (win * valid).sum(1) / denominator
        static[begin:end, 2] = np.where(valid.any(1), mean, 0.5)
        static[begin:end, 3] = np.sqrt(
            np.maximum(0, (win * win * valid).sum(1) / denominator - mean * mean)
        )
    static.flush()
    del data
    gc.collect()
    match_starts = starts.reshape(n, 10)[:, 0]
    match_ends = available.reshape(n, 10)[:, 0]
    boundaries = [float(match_starts[int(n * 0.7)]), float(match_starts[int(n * 0.85)])]
    masks = {
        "train": (match_starts < boundaries[0]) & (match_ends < boundaries[0]),
        "valid": (match_starts >= boundaries[0])
        & (match_starts < boundaries[1])
        & (match_ends < boundaries[1]),
        "test": match_starts >= boundaries[1],
    }
    total = np.zeros(len(HISTORY_FIELDS), dtype=np.float64)
    squares = total.copy()
    denominator = 0
    available_padded = np.r_[0.0, available]
    for split, mask in masks.items():
        rows = np.flatnonzero(mask)
        if not len(rows):
            raise ValueError("Empty chronological partition")
        np.save(OUT / f"{split}-rows.npy", rows, allow_pickle=False)
        if split != "train":
            continue
        for begin in range(0, len(rows), 2048):
            selected = rows[begin : begin + 2048]
            r = refs.reshape(n, 10, 10)[selected]
            valid = r > 0
            elapsed = np.maximum(match_starts[selected, None, None] - available_padded[r], 0)
            values = np.concatenate(
                [performance[r], np.log1p(elapsed / 86400)[..., None]], axis=-1
            )[valid]
            if np.any(
                available_padded[r[valid]]
                >= np.broadcast_to(match_starts[selected, None, None], r.shape)[valid]
            ):
                raise ValueError("Future observation entered training history")
            total += values.sum(0)
            squares += np.square(values).sum(0)
            denominator += len(values)
    mean = total / denominator
    scale = np.sqrt(np.maximum(0, squares / denominator - mean * mean)).clip(1e-6)
    training_static = static.reshape(n, 10, 6)[masks["train"]].reshape(-1, 6)
    normalizer = {
        "history_mean": mean.tolist(),
        "history_scale": scale.tolist(),
        "static_mean": training_static.mean(0, dtype=np.float64).tolist(),
        "static_scale": training_static.std(0, dtype=np.float64).clip(1e-6).tolist(),
        "fitted_partition": "train",
        "observed_history_entries": denominator,
    }
    # Persist the match-shaped view, not a second full copy in memory.
    static_view = np.array(static.reshape(n, 2, 5, 6), copy=True)
    del static
    np.save(OUT / "static.npy", static_view, allow_pickle=False)
    files = {p.name: file_hash(p) for p in sorted(OUT.glob("*.npy"))}
    result = {
        "identity": identity,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_rows": raw_count,
        "duplicate_identity_rows_removed": duplicates_removed,
        "invalid_or_conflicting_contexts": len(bad_context),
        "matches": n,
        "history_fields": HISTORY_FIELDS,
        "static_fields": STATIC_FIELDS,
        "normalizer": normalizer,
        "boundaries": boundaries,
        "split_counts": {s: int(m.sum()) for s, m in masks.items()},
        "files_sha256": files,
        "storage_bytes": sum((OUT / p).stat().st_size for p in files),
        "classes": 2,
        "team_size": 5,
        "history_length": 10,
        "information_set": "Player identities only for prior lookup; past performance and historical role, no current draft/performance/rank snapshot.",
        "limitation": "Reconstructed match contexts; same-game role/performance only used after completion. Single ranked mode, selected active-player sample.",
        "status": "prepared_not_yet_trained",
    }
    archive = OUT / "source-snapshot"
    archive.mkdir(exist_ok=True)
    for source in [
        Path(__file__),
        ROOT / "scripts/audit_league_data.py",
        ROOT / "src/gamematch/data/indexed_history.py",
    ]:
        shutil.copyfile(source, archive / source.name)
    marker.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["normalizer", "files_sha256", "history_fields"]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Read-only verification against archived producing source",
    )
    arguments = parser.parse_args()
    OUT = arguments.output.resolve()
    if arguments.verify:
        verify_existing(OUT)
    else:
        main()
