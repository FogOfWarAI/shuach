# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Exploratory cross-model comparisons under a common calibration policy."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from gamematch.evaluation.comparison import paired_cluster_bootstrap, paired_losses


def main():
    run = Path("runs/hero-calibration-v1")
    marker = run / "manifest.json"
    manifest = json.loads(marker.read_text())
    if manifest["status"] != "completed":
        raise ValueError("All declared calibration fits must finish first")
    config = manifest["config"]
    comparisons, artifacts = [], {}
    for dataset in config["datasets"]:
        for mode in config["variants"]:
            for model in config["models"]:
                if model == "logistic":
                    continue
                paths = [
                    [
                        run / dataset / f"seed-{seed}-{name}-{mode}" / "predictions.npz"
                        for seed in config["seeds"]
                    ]
                    for name in [model, "logistic"]
                ]
                delta, reference = paired_losses(*paths)
                result = paired_cluster_bootstrap(
                    delta, reference["group_id"], replicates=2500, seed=20261114
                )
                comparisons.append(
                    {
                        "dataset": dataset,
                        "first": model,
                        "second": "logistic",
                        "variant": mode,
                        **result,
                    }
                )
                for group in paths:
                    for path in group:
                        artifacts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_manifest_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Exploratory analysis specified after observing completed calibration summaries; all six non-logistic models compared against logistic regression within each of two calibration policies and three datasets (36 comparisons). No new fitting or test-driven change to the primary benchmark. Conditional paired seed/roster-group intervals; no multiplicity correction or claim of a new confirmation test.",
        "comparisons": comparisons,
        "prediction_sha256": artifacts,
    }
    Path("reports/calibration-cross-model.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Computed", len(comparisons), "exploratory comparisons under matched calibration policies"
    )


if __name__ == "__main__":
    main()
