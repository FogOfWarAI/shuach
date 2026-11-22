# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Rebuild an outcome-only skill proxy without the supplied rating table.

The supplied historical data overlap the first main matches. We use only a
strict prefix ending at least 24 hours before the earliest main start. Historical
durations are absent, so that embargo is a declared availability assumption.
Anonymous account 0 receives a fresh rating on every occurrence and is never
persisted or used as a training target.
"""

import argparse
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openskill.models import PlackettLuce

from gamematch.data.heroes import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve() / "extracted/dota2-matches-v2"
    out = (args.output or args.data_root / "derived/dota2-reference-v2").resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        name: file_hash(root / name) for name in ["match.csv", "players.csv", "match_outcomes.csv"]
    }
    identity = {
        "sources": sources,
        "builder_sha256": file_hash(Path(__file__)),
        "historical_embargo_seconds": 86400,
        "reference_model": "openskill.PlackettLuce default parameters, anonymous occurrences independent",
        "future_horizon_observed_matches": 5,
    }
    marker = out / "manifest.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get("identity") != identity:
            raise ValueError("Reference output belongs to different source/protocol")
        if old.get("status") == "completed":
            print("Reference already completed")
            return
    manifest = {
        "identity": identity,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "target": "future observed OpenSkill posterior mean after five additional observed matches",
        "scope": "Skill-state proxy, not true skill, actual MMR, or verified novice identity",
        "excluded_source": "player_ratings.csv not used because it pools a temporally overlapping archive",
        "historical_timestamp_limit": "No duration; fixed 24h embargo assumption, chronology by reported start",
        "main_timestamp_policy": "Update at reported start+duration; current-game snapshots used only after completion",
    }
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(__file__, out / "reference-builder.py")
    matches = pd.read_csv(root / "match.csv").set_index("match_id").sort_index()
    players = pd.read_csv(
        root / "players.csv", usecols=["match_id", "account_id", "hero_id", "player_slot"]
    )
    historical = pd.read_csv(root / "match_outcomes.csv")
    positive = historical[historical.rad == 1].sort_values("match_id").reset_index(drop=True)
    negative = historical[historical.rad == 0].sort_values("match_id").reset_index(drop=True)
    if not np.array_equal(positive.match_id, negative.match_id):
        raise ValueError("Historical team pairing is incomplete")
    columns = [f"account_id_{i}" for i in range(5)]
    rosters = np.stack(
        [positive[columns].to_numpy(np.int64), negative[columns].to_numpy(np.int64)], 1
    )
    flat = np.sort(rosters.reshape(-1, 10), axis=1)
    duplicate = ((np.diff(flat, axis=1) == 0) & (flat[:, :-1] != 0)).any(1)
    valid = (
        (positive.start_time.to_numpy() == negative.start_time.to_numpy())
        & ((positive.win.to_numpy() + negative.win.to_numpy()) == 1)
        & ~duplicate
    )
    historical_cutoff = float(matches.start_time.min()) - 86400
    prefix = positive.start_time.to_numpy() < historical_cutoff
    order = np.flatnonzero(valid & prefix)
    order = order[np.argsort(positive.start_time.to_numpy()[order], kind="stable")]
    manifest["historical_audit"] = {
        "total_matches": len(positive),
        "invalid_pair_or_repeated_known_identity": int((~valid).sum()),
        "retained_prefix_matches": len(order),
        "cutoff_unix": historical_cutoff,
    }
    model = PlackettLuce()
    state = {}
    counts = defaultdict(int)

    def update(teams, win):
        groups = []
        for team in teams:
            group = []
            for player in team:
                player = int(player)
                if player and player not in state:
                    state[player] = model.rating()
                group.append(state[player] if player else model.rating())
            groups.append(group)
        updated = model.rate(groups, ranks=[0, 1] if win else [1, 0])
        for ids, ratings in zip(teams, updated):
            for player, rating in zip(ids, ratings):
                if player:
                    state[int(player)] = rating
                    counts[int(player)] += 1

    warm = out / "historical-state.npz"
    started = time.perf_counter()
    if warm.exists():
        with np.load(warm, allow_pickle=False) as f:
            for player, mu, sigma, n in zip(f["account_id"], f["mu"], f["sigma"], f["count"]):
                state[int(player)] = model.rating(mu=float(mu), sigma=float(sigma))
                counts[int(player)] = int(n)
        print("Loaded historical checkpoint", len(state), flush=True)
    else:
        for j, index in enumerate(order):
            update(rosters[index], bool(positive.win.iloc[index]))
            if (j + 1) % 50000 == 0:
                print("Historical prefix", j + 1, "/", len(order), flush=True)
        keys = np.array(sorted(state), dtype=np.int64)
        np.savez_compressed(
            warm,
            account_id=keys,
            mu=np.array([state[p].mu for p in keys]),
            sigma=np.array([state[p].sigma for p in keys]),
            count=np.array([counts[p] for p in keys]),
        )
        print("Saved historical checkpoint", len(state), flush=True)
    manifest["historical_seconds"] = time.perf_counter() - started
    prior_counts = dict(counts)
    del historical, positive, negative, rosters, flat
    slots = [0, 1, 2, 3, 4, 128, 129, 130, 131, 132]
    accounts = players.pivot(index="match_id", columns="player_slot", values="account_id").reindex(
        index=matches.index, columns=slots
    )
    heroes = players.pivot(index="match_id", columns="player_slot", values="hero_id").reindex(
        index=matches.index, columns=slots
    )
    if accounts.isna().any().any() or heroes.isna().any().any():
        raise ValueError("Missing main roster slots")
    rosters = accounts.to_numpy(np.int64).reshape(-1, 2, 5)
    hero_rosters = heroes.to_numpy(np.int64).reshape(-1, 2, 5)
    flat = np.sort(rosters.reshape(-1, 10), axis=1)
    duplicate = ((np.diff(flat, axis=1) == 0) & (flat[:, :-1] != 0)).any(1)
    known_heroes = (hero_rosters > 0).all((1, 2)) & (
        np.diff(np.sort(hero_rosters.reshape(-1, 10), axis=1), axis=1) != 0
    ).all(1)
    end = (matches.start_time + matches.duration).to_numpy(float)
    valid = ~duplicate & known_heroes & (matches.duration.to_numpy() > 0)
    order = np.flatnonzero(valid)
    order = order[np.argsort(end[order], kind="stable")]
    seen = defaultdict(int)
    records = []
    for j, index in enumerate(order):
        update(rosters[index], bool(matches.radiant_win.iloc[index]))
        for side in range(2):
            for position in range(5):
                player = int(rosters[index, side, position])
                if not player:
                    continue
                seen[player] += 1
                rating = state[player]
                records.append(
                    (
                        int(matches.index[index]),
                        player,
                        slots[side * 5 + position],
                        end[index],
                        rating.mu,
                        rating.sigma,
                        prior_counts.get(player, 0),
                        seen[player],
                    )
                )
        if (j + 1) % 10000 == 0:
            print("Main completed matches", j + 1, "/", len(order), flush=True)
    frame = pd.DataFrame(
        records,
        columns=[
            "match_id",
            "account_id",
            "player_slot",
            "completed_at",
            "current_mu",
            "current_sigma",
            "historical_count",
            "main_observed_count",
        ],
    )
    # The table is already ordered by completion, including deterministic ties.
    groups = frame.groupby("account_id", sort=False)
    frame["future_mu"] = groups.current_mu.shift(-5)
    frame["target_available_at"] = groups.completed_at.shift(-5)
    eligible = frame.dropna(subset=["future_mu", "target_available_at"])
    arrays = {column: eligible[column].to_numpy() for column in eligible.columns}
    np.savez_compressed(out / "skill-examples.npz", **arrays)
    np.savez_compressed(
        out / "main-rosters.npz",
        match_id=matches.index.to_numpy(np.int64),
        heroes=hero_rosters,
        accounts=rosters,
        start_time=matches.start_time.to_numpy(),
        duration=matches.duration.to_numpy(),
        valid=valid,
        outcome=matches.radiant_win.to_numpy(np.int64),
    )
    manifest.update(
        status="completed",
        main_valid_matches=int(valid.sum()),
        main_excluded_matches=int((~valid).sum()),
        known_player_records=len(frame),
        eligible_future_proxy_examples=len(eligible),
        eligible_accounts=int(eligible.account_id.nunique()),
        eligible_low_observed_history=int(
            ((eligible.historical_count <= 20) & (eligible.main_observed_count <= 3)).sum()
        ),
        updated_utc=datetime.now(timezone.utc).isoformat(),
    )
    for q in ["historical-state.npz", "skill-examples.npz", "main-rosters.npz"]:
        manifest[q + "_sha256"] = file_hash(out / q)
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ["identity"]}, indent=2))


if __name__ == "__main__":
    main()
