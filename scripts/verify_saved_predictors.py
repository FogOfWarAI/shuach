# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Reload completed real-data models on CPU and compare sampled saved predictions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gamematch.data.asymmetric import load_asymmetric
from gamematch.data.heroes import file_hash, prepare
from gamematch.data.history import HistoryData
from gamematch.inference import AsymmetricPredictor, HeroPredictor, HistoryPredictor, SkillPredictor


def read(path):
    return json.loads(path.read_text())


def main():
    root = Path.cwd()
    torch.set_num_threads(2)
    checks = []

    def compare(path, actual, expected):
        error = float(np.max(np.abs(actual - expected)))
        if error > 5e-5 or not np.isfinite(actual).all():
            raise AssertionError(f"Reload mismatch {path}: {error}")
        checks.append(
            {
                "artifact": str(path.relative_to(root)),
                "rows": len(actual),
                "max_absolute_error": error,
            }
        )

    def sample(data):
        return np.sort(
            np.random.default_rng(20261114).choice(len(data), min(128, len(data)), replace=False)
        )

    for run_name in ["local-full-heroes-v1", "neuralac-ablations-v1", "neuralac-convergence-v1"]:
        run = root / "runs" / run_name
        for folder in sorted(run.glob("*/data-manifest.json")):
            dataset = folder.parent.name
            splits, _, manifest = prepare(dataset, root / "datasets", "final")
            if manifest["vocabulary"] != read(folder)["vocabulary"]:
                raise ValueError("Training vocabulary changed")
            ids = sample(splits["test"])
            for path in sorted(folder.parent.glob("seed-*/metrics.json")):
                with np.load(path.with_name("predictions.npz"), allow_pickle=False) as saved:
                    np.testing.assert_array_equal(splits["test"].row_ids, saved["row_id"])
                    actual = HeroPredictor(path.parent, batch_size=37).predict(
                        splits["test"].rosters[ids]
                    )
                    compare(path, actual, saved["p_positive"][ids])
    for run_name in [
        "overwatch-history-v1",
        "overwatch-target-only-v1",
        "overwatch-history-earlier-origin-v1",
    ]:
        run = root / "runs" / run_name
        if not (run / "manifest.json").exists():
            continue
        metadata = read(run / "manifest.json")["data"]
        identity = hashlib.sha256(
            json.dumps(metadata["identity"], sort_keys=True).encode()
        ).hexdigest()[:20]
        cached = root / f".cache/history/{identity}.npz"
        if file_hash(cached) != metadata["cache_sha256"]:
            raise ValueError("History cache changed")
        data = HistoryData.load(cached)
        lookup = {row: i for i, row in enumerate(data.row_ids)}
        for path in sorted(run.glob("seed-*/metrics.json")):
            with np.load(path.with_name("predictions.npz"), allow_pickle=False) as saved:
                ids = sample(saved["row_id"])
                selected = np.asarray([lookup[row] for row in saved["row_id"][ids]])
                actual = HistoryPredictor(path.parent, batch_size=37).predict(
                    data.history[selected],
                    data.static[selected],
                    data.valid[selected],
                    data.domain[selected],
                )
                compare(path, actual, saved["probabilities"][ids])
    run = root / "runs/asymmetric-v1"
    for dataset in ["dbd-2022", "dbd-2023"]:
        data = load_asymmetric(dataset, root / "datasets")
        lookup = {row: i for i, row in enumerate(data.row_ids)}
        for path in sorted((run / dataset).glob("seed-*/metrics.json")):
            with np.load(path.with_name("predictions.npz"), allow_pickle=False) as saved:
                ids = sample(saved["row_id"])
                selected = [lookup[row] for row in saved["row_id"][ids]]
                actual = AsymmetricPredictor(path.parent, batch_size=37).predict(
                    data.features.iloc[selected]
                )
                compare(path, actual, saved["probabilities"][ids])
    run = root / "runs/skill-v1"
    metadata = read(root / "datasets/derived/dota2-reference-v2/snapshots-manifest.json")
    cached = root / "datasets/derived/dota2-reference-v2/snapshots.npz"
    if file_hash(cached) != metadata["cached_sha256"]:
        raise ValueError("Snapshot cache changed")
    with np.load(cached, allow_pickle=False) as arrays:
        lookup = {row: i for i, row in enumerate(arrays["row_id"])}
        for path in sorted(run.glob("seed-*/metrics.json")):
            with np.load(path.with_name("predictions.npz"), allow_pickle=False) as saved:
                ids = sample(saved["row_id"])
                selected = [lookup[row] for row in saved["row_id"][ids]]
                actual = SkillPredictor(path.parent, batch_size=37).predict(
                    arrays["snapshots"][selected],
                    arrays["valid"][selected],
                    arrays["lineups"][selected],
                    arrays["current_mu"][selected],
                )
                compare(path, actual, saved["prediction"][ids])
    payload = {
        "status": "passed",
        "scope": "Completed artifacts present at invocation; 128 deterministically sampled test rows each, CPU reload with batch size 37, absolute tolerance 5e-5 for original CPU/MPS differences; no refitting",
        "checks": checks,
        "jobs": len(checks),
        "max_absolute_error": max(c["max_absolute_error"] for c in checks),
    }
    (root / "reports/predictor-reload-audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
