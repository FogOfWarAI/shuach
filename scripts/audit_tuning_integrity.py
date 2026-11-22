# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Replay selection using archived validation predictions, never test scores."""

import hashlib
import json
from pathlib import Path

import numpy as np

from gamematch.tuning.engine import primary_loss
from gamematch.tuning.search import select_trial


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(root, specialist=False):
    protocol = read(root / "protocol.json")
    digest = hashlib.sha256()
    source = root / "source-snapshot/gamematch"
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    key = "source_sha256" if specialist else "code_sha256"
    if digest.hexdigest() != protocol[key]:
        raise ValueError(f"Source identity changed: {root}")
    if specialist and sha(root / "source-driver.py") != protocol["driver_sha256"]:
        raise ValueError("Specialist driver identity changed")
    return protocol


def validation_record(path, configuration, manifest):
    result = read(path)
    if result["test_evaluated"] is not False or (path.parent / "predictions.npz").exists():
        raise ValueError(f"Search artifact contains test evaluation: {path}")
    if result["configuration"] != configuration:
        raise ValueError(f"Search configuration mismatch: {path}")
    if result["seed"] != int(path.parent.name.removeprefix("seed-")):
        raise ValueError(f"Seed identity mismatch: {path}")
    with np.load(path.parent / "validation.npz", allow_pickle=False) as saved:
        score = primary_loss(
            manifest["task"], saved["y_true"], saved["raw"], manifest["specification"]
        )
    if not np.isclose(score, result["validation_primary"], atol=1e-10, rtol=1e-10):
        raise ValueError(f"Validation prediction/score mismatch: {path}")
    if not configuration["tabular"]:
        history = read(path.parent / "history.json")
        if len(history) != result["epochs_completed"] or len(history) > configuration["epochs"]:
            raise ValueError(f"Epoch count mismatch: {path}")
        if not np.isclose(min(h["validation_primary"] for h in history), score, atol=1e-6):
            raise ValueError(f"Checkpoint does not minimize recorded validation loss: {path}")
    return result


def locked_choice(folder, protocol_path, records, seeds, config):
    selection = read(folder / "selection.json")
    winner = select_trial(records, seeds)
    if (
        selection["test_used"] is not False
        or selection["protocol_sha256"] != sha(protocol_path)
        or selection["trial"] != winner
        or selection["selection_seeds"] != seeds
        or selection["configuration"] != config(winner)
        or not np.isclose(
            selection["validation_mean"],
            np.mean([r["validation_primary"] for r in records[winner]]),
            atol=1e-12,
        )
    ):
        raise ValueError(f"Locked selection cannot be replayed: {folder}")
    for result_path in (folder / "final").glob("seed-*/result.json"):
        result = read(result_path)
        if (
            result["configuration"] != selection["configuration"]
            or result["selection_sha256"] != sha(folder / "selection.json")
            or result_path.stat().st_mtime_ns < (folder / "selection.json").stat().st_mtime_ns
        ):
            raise ValueError(f"Final fit violates the locked selection: {result_path}")
    return selection


def broad_selection(folder, protocol):
    dataset, name = folder.parent.name, folder.name
    configs = protocol["spaces"][dataset][name]
    manifest = read(folder.parent / "data-world-0/data-manifest.json")
    records = {}
    for i, cfg in enumerate(configs):
        path = folder / "search" / f"trial-{i:02d}" / "seed-100/metrics.json"
        records[i] = [validation_record(path, cfg, manifest)]
    shortlist = sorted(records, key=lambda i: (records[i][0]["validation_primary"], i))[:3]
    two_seeds = not configs[0]["tabular"] or dataset.startswith("synthetic-")
    if two_seeds:
        for i in shortlist:
            path = folder / "search" / f"trial-{i:02d}" / "seed-101/metrics.json"
            records[i].append(validation_record(path, configs[i], manifest))
    paths = list((folder / "search").glob("trial-*/seed-*/metrics.json"))
    if len(paths) != len(configs) + (3 if two_seeds else 0):
        raise ValueError(f"Undeclared search fits: {folder}")
    selection = locked_choice(
        folder,
        folder.parents[1] / "protocol.json",
        records,
        [100, 101] if two_seeds else [100],
        lambda i: configs[i],
    )
    if selection["shortlist"] != shortlist:
        raise ValueError(f"Shortlist mismatch: {folder}")
    return records


def specialist_selection(folder, protocol):
    configs = protocol["spaces"][folder.name]
    manifest = read(folder.parent / "data/data-manifest.json")
    active, stages, fit_count = list(range(16)), [], 0
    for rung, setup in enumerate(protocol["rungs"]):
        records = {}
        for i in active:
            records[i] = [
                validation_record(
                    folder / f"rung-{rung}" / f"trial-{i:02d}" / f"seed-{seed}/metrics.json",
                    {**configs[i], "epochs": setup["epochs"]},
                    manifest,
                )
                for seed in setup["seeds"]
            ]
            fit_count += len(records[i])
        ranking = sorted(
            records, key=lambda i: (np.mean([r["validation_primary"] for r in records[i]]), i)
        )
        stages.append({"rung": rung, "active": active, "ranking": ranking})
        if rung < len(protocol["rungs"]) - 1:
            count = protocol["rungs"][rung + 1]["trials"]
            active = [0] + [i for i in ranking if i != 0][: count - 1]
    if stages != read(folder / "promotions.json"):
        raise ValueError(f"Promotions cannot be replayed: {folder}")
    if len(list(folder.glob("rung-*/trial-*/seed-*/metrics.json"))) != fit_count:
        raise ValueError(f"Undeclared specialist fits: {folder}")
    locked_choice(
        folder,
        folder.parent / "protocol.json",
        records,
        [200, 201],
        lambda i: {**configs[i], "epochs": protocol["final_epoch_cap"]},
    )
    return fit_count
