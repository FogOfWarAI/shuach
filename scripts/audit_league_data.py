# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Inspect a public LoL candidate without treating snapshots as pre-match inputs.

Optional research dependency: pyarrow. No model is trained by this script.
"""

import heapq
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gamematch.data.heroes import file_hash

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "datasets/metadata/leagueData"
KEY = ["gameStartTimestamp", "gameDuration", "gameVersion"]
IDENTITY = KEY + ["participantId", "puuid"]
FIELDS = IDENTITY + ["teamId", "win"]
SOURCE = "https://huggingface.co/datasets/renecotyfanboy/leagueData"
SOURCE_REVISION = "d91ee149784e9867d03eabbc8fd0cc1f75877157"
PARQUET_REVISION = "787e032920b7abfb126fd1e5be898a3f0b546f46"
EXPECTED_SHARDS = {f"{index:04d}.parquet" for index in range(12)}


def validate_manifest(manifest, official_files, root):
    """Require the fixed twelve unique shards before reading any Parquet data."""
    if (
        manifest.get("status") != "downloaded"
        or manifest.get("source") != SOURCE
        or manifest.get("source_revision") != SOURCE_REVISION
        or manifest.get("parquet_revision") != PARQUET_REVISION
    ):
        raise ValueError("Expected the downloaded, fixed-revision leagueData source")
    official = {}
    for record in official_files:
        name = Path(record["path"]).name
        if (
            name not in EXPECTED_SHARDS
            or record["path"] != f"default/train/{name}"
            or name in official
            or record.get("type") != "file"
            or type(record.get("size")) is not int
            or record["size"] <= 0
            or record["lfs"]["size"] != record["size"]
        ):
            raise ValueError("Invalid or duplicate shard in the saved official file list")
        official[name] = (record["size"], record["lfs"]["oid"])
    if set(official) != EXPECTED_SHARDS:
        raise ValueError("Saved official file list must contain all twelve expected shards")
    records = manifest.get("files", [])
    if len(records) != len(EXPECTED_SHARDS):
        raise ValueError("All twelve verified files are required")
    root = Path(root).resolve()
    raw_root = root / "datasets/raw/leagueData"
    seen, resolved_seen, verified = set(), set(), []
    for record in records:
        relative = record["file"]
        name = Path(relative).name
        if (
            name not in EXPECTED_SHARDS
            or relative != f"datasets/raw/leagueData/{name}"
            or name in seen
        ):
            raise ValueError("Unexpected, escaping or duplicate local shard path")
        path = (root / relative).resolve()
        if not path.is_relative_to(raw_root) or not path.is_file():
            raise ValueError("Shard must be a regular file inside the expected data directory")
        if path in resolved_seen:
            raise ValueError("Duplicate resolved local shard path")
        size, digest = official[name]
        if record.get("bytes") != size or record.get("sha256") != digest:
            raise ValueError("Manifest size/SHA256 differs from the saved official file list")
        if path.stat().st_size != size or file_hash(path) != digest:
            raise ValueError("Downloaded file size/SHA256 changed")
        seen.add(name)
        resolved_seen.add(path)
        verified.append(path)
    return verified


def main():
    manifest = json.loads((META / "manifest.json").read_text())
    official_files = json.loads((META / "parquet-files.json").read_text())
    verified_paths = validate_manifest(manifest, official_files, ROOT)
    import pyarrow.parquet as pq

    parts, schemas = [], []
    for path in verified_paths:
        file = pq.ParquetFile(path)
        schemas.append(file.schema.names)
        parts.append(file.read(columns=FIELDS).to_pandas())
    if not all(names == schemas[0] for names in schemas):
        raise ValueError("Parquet schemas differ")
    raw = pd.concat(parts, ignore_index=True)
    del parts
    print("RAW", len(raw), flush=True)
    missing = raw[FIELDS].isna().any(axis=1)
    clean = raw.loc[~missing].copy()
    invalid = (
        ~clean.teamId.isin([100, 200])
        | ~clean.participantId.isin(range(1, 11))
        | (clean.gameDuration < 900)
        | ~clean.gameStartTimestamp.between(1577836800000, 1735689600000)
    )
    invalid_keys = pd.MultiIndex.from_frame(clean.loc[invalid, KEY]).unique()
    # Conflicting duplicate identities must not be silently resolved by row order.
    duplicate_mask = clean.duplicated(IDENTITY, keep=False)
    dup = clean.loc[duplicate_mask]
    variation = dup.groupby(IDENTITY, sort=False)[["teamId", "win"]].nunique()
    conflict_index = variation[(variation > 1).any(axis=1)].reset_index()[KEY]
    conflict_keys = pd.MultiIndex.from_frame(conflict_index).unique()
    deduplicated = clean.drop_duplicates(IDENTITY)
    bad_keys = invalid_keys.union(conflict_keys)
    deduplicated = deduplicated.loc[
        ~pd.MultiIndex.from_frame(deduplicated[KEY]).isin(bad_keys)
    ].copy()
    groups = deduplicated.groupby(KEY, sort=True)
    counts = groups.agg(
        rows=("puuid", "size"),
        people=("puuid", "nunique"),
        slots=("participantId", "nunique"),
        wins=("win", "sum"),
    )
    team = deduplicated.groupby(KEY + ["teamId"], sort=True).agg(
        rows=("puuid", "size"), unique_results=("win", "nunique")
    )
    good_team = ((team.rows == 5) & (team.unique_results == 1)).groupby(level=KEY).sum()
    keep = (counts.rows == 10) & (counts.people == 10) & (counts.slots == 10) & (counts.wins == 5)
    keep &= good_team.reindex(counts.index, fill_value=0) == 2
    keys = counts.index[keep]
    accepted = deduplicated.loc[pd.MultiIndex.from_frame(deduplicated[KEY]).isin(keys)].copy()
    accepted = accepted.sort_values(KEY + ["teamId", "participantId"])
    count = len(accepted) // 10
    if len(accepted) != count * 10:
        raise ValueError("Non-ten-player accepted cohort")
    people = accepted.puuid.to_numpy().reshape(count, 10)
    starts = accepted.gameStartTimestamp.to_numpy().reshape(count, 10)[:, 0] / 1000
    durations = accepted.gameDuration.to_numpy().reshape(count, 10)[:, 0]
    available = starts + durations
    # Prospective coverage includes only previously completed retained games.
    pending, observed = [], defaultdict(int)
    history = np.zeros((count, 10), dtype=np.int32)
    for i, start in enumerate(starts):
        while pending and pending[0][0] < start:
            _, index = heapq.heappop(pending)
            for player in people[index]:
                observed[player] += 1
        history[i] = [observed[p] for p in people[i]]
        heapq.heappush(pending, (available[i], i))
    bounds = [starts[int(count * 0.7)], starts[int(count * 0.85)]]
    masks = {
        "train": (starts < bounds[0]) & (available < bounds[0]),
        "valid": (starts >= bounds[0]) & (starts < bounds[1]) & (available < bounds[1]),
        "test": starts >= bounds[1],
    }
    coverage = {}
    for split, mask in masks.items():
        h = history[mask]
        coverage[split] = {
            "matches": int(mask.sum()),
            "zero_history_player_fraction": float((h == 0).mean()),
            "median_history": float(np.median(h)),
            "all_players_observed_matches": int((h >= 1).all(1).sum()),
            "all_players_at_least_5_matches": int((h >= 5).all(1).sum()),
            "player_at_least_10_fraction": float((h >= 10).mean()),
        }
    report = {
        "status": "candidate_identity_and_roster_audit_complete_not_formal_benchmark",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": file_hash(Path(__file__)),
        "manifest_sha256": file_hash(META / "manifest.json"),
        "raw_rows": len(raw),
        "columns": len(schemas[0]),
        "raw_unique_puuids": int(raw.puuid.nunique()),
        "missing_required_rows": int(missing.sum()),
        "invalid_context_rows": int(invalid.sum()),
        "duplicate_identity_rows_removed": len(clean) - len(clean.drop_duplicates(IDENTITY)),
        "conflicting_duplicate_contexts": len(conflict_keys),
        "candidate_contexts": len(counts),
        "accepted_matches": count,
        "rejected_contexts": int((~keep).sum()),
        "accepted_players": int(accepted.puuid.nunique()),
        "time_range_utc": [
            datetime.fromtimestamp(float(t), timezone.utc).isoformat()
            for t in (starts.min(), starts.max())
        ],
        "provisional_coverage": coverage,
        "fields": schemas[0],
        "rich_past_performance_fields": [
            f
            for f in [
                "kills",
                "deaths",
                "assists",
                "goldEarned",
                "goldPerMinute",
                "champExperience",
                "totalMinionsKilled",
                "totalDamageDealtToChampions",
                "damagePerMinute",
                "visionScore",
                "wardsPlaced",
                "wardsKilled",
                "killParticipation",
                "teamDamagePercentage",
                "championId",
                "teamPosition",
                "role",
                "gameVersion",
            ]
            if f in schemas[0]
        ],
        "forbidden_pre_match_fields": [
            "current-match performance",
            "elo",
            "is_in_reference_sample",
        ],
        "limitations": [
            "Native match ID is absent; context grouping uses millisecond start + duration + version and rejects ambiguous/incomplete ten-player rosters.",
            "Duration is used only for record grouping and feedback timing, never as a current-match predictor.",
            "The uploader's elo is a later sampling snapshot, not a historical pre-match rank.",
            "Historical performance duplicate consistency and a full adapter still require verification.",
            "The provisional 70/15/15 coverage split is a descriptive audit, not a locked training protocol.",
            "Sampling players with >200 games and removing short games induces selection bias.",
        ],
    }
    (META / "admission.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["fields", "limitations"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
