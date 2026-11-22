# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Declared exploratory calibration of completed fixed hero checkpoints."""

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.special import logit

import gamematch
from gamematch.configuration import runtime_identity
from gamematch.data.heroes import file_hash, prepare
from gamematch.evaluation.binary import binary_metrics
from gamematch.evaluation.calibration import apply_binary_calibration, fit_binary_calibration
from gamematch.evaluation.comparison import paired_cluster_bootstrap, paired_losses
from gamematch.inference import HeroPredictor
from gamematch.runner import code_hash, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("benchmarks/hero-calibration-sensitivity.json")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/hero-calibration-v1"))
    args = parser.parse_args()
    root, output = Path.cwd(), args.output.resolve()
    config = read(args.config)
    primary = root / config["primary_run"]
    if read(primary / "manifest.json")["status"] != "completed":
        raise ValueError("Primary reference run must finish before calibration follow-up")
    identity = {
        "config": config,
        "code": code_hash(),
        "script": file_hash(Path(__file__)),
        "primary_manifest": file_hash(primary / "manifest.json"),
        "runtime": runtime_identity(),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "manifest.json"
    if marker.exists():
        manifest = read(marker)
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Calibration output belongs to a different identity")
    else:
        manifest = {
            "status": "running",
            "identity": identity,
            "fingerprint": fingerprint,
            "code_sha256": code_hash(),
            "config": config,
            "unit": "validation postprocessing fit, not additional model training",
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        shutil.copytree(
            Path(gamematch.__file__).parent,
            output / "source-snapshot/gamematch",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(__file__, output / "calibration-script.py")
    write_json(marker, manifest)
    torch.set_num_threads(2)
    eps = config["probability_clip_for_logit"]
    records = defaultdict(list)
    for dataset in config["datasets"]:
        splits, _, data_manifest = prepare(dataset, root / "datasets", "final")
        if data_manifest != read(primary / dataset / "data-manifest.json"):
            raise ValueError("Primary data/split manifest differs from reloaded adapter")
        for model in config["models"]:
            for seed in config["seeds"]:
                original = primary / dataset / f"seed-{seed}-{model}"
                results = []
                for mode in config["variants"]:
                    path = output / dataset / f"seed-{seed}-{model}-{mode}"
                    results.append((mode, path))
                if any(not (path / "metrics.json").exists() for _, path in results):
                    predictor = HeroPredictor(original, batch_size=1024)
                    # Use the same float64 probability-to-logit conversion as
                    # the saved test predictions and convex calibration fit.
                    validation = np.asarray(
                        predictor.predict(splits["valid"].rosters), dtype=float
                    )
                    with np.load(original / "predictions.npz", allow_pickle=False) as saved:
                        raw = dict(saved)
                    np.testing.assert_array_equal(raw["row_id"], splits["test"].row_ids)
                    np.testing.assert_array_equal(raw["y_true"], splits["test"].labels)
                    validation_logits = logit(np.clip(validation, eps, 1 - eps))
                    test_logits = logit(np.clip(raw["p_positive"].astype(float), eps, 1 - eps))
                    for mode, path in results:
                        path.mkdir(parents=True, exist_ok=True)
                        parameters = fit_binary_calibration(
                            validation_logits, splits["valid"].labels, mode
                        )
                        probability = apply_binary_calibration(test_logits, parameters)
                        np.savez_compressed(
                            path / "predictions.npz", **(raw | {"p_positive": probability})
                        )
                        np.savez_compressed(
                            path / "validation.npz",
                            row_id=splits["valid"].row_ids,
                            y_true=splits["valid"].labels,
                            p_positive=validation,
                        )
                        result = {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "variant": mode,
                            "metrics": binary_metrics(raw["y_true"], probability),
                            "parameters": parameters,
                            "primary_predictions_sha256": file_hash(original / "predictions.npz"),
                            "primary_checkpoint_sha256": {
                                p.name: file_hash(p) for p in original.glob("checkpoint.*")
                            },
                            "validation_split_sha256": file_hash(
                                primary / dataset / "split-manifest.npz"
                            ),
                            "source_primary": str(original.relative_to(root)),
                        }
                        write_json(path / "metrics.json", result)
                for mode, path in results:
                    records[(dataset, model, mode)].append((path, read(path / "metrics.json")))
                print(dataset, model, seed, "calibrated", flush=True)
    summaries, comparisons = [], []
    for (dataset, model, mode), values in sorted(records.items()):
        metrics = {}
        for metric in ["log_loss", "brier", "auc", "accuracy", "ece_15_equal_width"]:
            scores = [r["metrics"][metric] for _, r in values]
            metrics[metric] = {
                "mean": float(np.mean(scores)),
                "seed_sample_sd": float(np.std(scores, ddof=1)),
                "values": scores,
            }
        summaries.append({"dataset": dataset, "model": model, "variant": mode, "metrics": metrics})
        calibrated = [p / "predictions.npz" for p, _ in values]
        original = [root / r["source_primary"] / "predictions.npz" for _, r in values]
        differences, reference = paired_losses(calibrated, original)
        paired = paired_cluster_bootstrap(
            differences, reference["group_id"], replicates=2500, seed=20261114
        )
        comparisons.append({"dataset": dataset, "model": model, "variant": mode, **paired})
    write_json(
        output / "summary.json",
        {"config": config, "summaries": summaries, "paired_comparisons": comparisons},
    )
    manifest.update(
        status="completed",
        completed_postprocessing_fits=sum(len(v) for v in records.values()),
        updated_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(marker, manifest)


if __name__ == "__main__":
    main()
