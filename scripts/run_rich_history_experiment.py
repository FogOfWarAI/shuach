# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Frozen input-by-readout experiment; a CPU worker independent of MPS searches."""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from gamematch.configuration import runtime_identity
from gamematch.data.heroes import file_hash
from gamematch.data.history import HistoryData, history_split_masks, standardize_histories
from gamematch.data.rich_history import RICH_HISTORY_FEATURES, RICH_STATIC_FEATURES
from gamematch.runner import code_hash
from gamematch.tuning.data import Bundle, partition
from gamematch.tuning.engine import evaluate, fit, predict_raw, reload_model

ROOT = Path(os.environ.get("GAMEMATCH_WORKSPACE", Path(__file__).resolve().parents[1])).resolve()
OUT = ROOT / "runs/rich-history-v1"
VARIANTS = [("original", 5, 13), ("rich", 8, 46)]
SEEDS = [11, 22, 33]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def cell_name(cell):
    return f"{cell['model']}--{cell['features']}--{cell['pooling']}"


def execution_schedule(cells, seeds):
    """Finish every declared cell for one seed before advancing to the next."""
    return [(cell, seed) for seed in seeds for cell in cells]


def verify_snapshot(protocol):
    source = OUT / "source-snapshot/gamematch"
    if not source.is_dir():
        raise ValueError("Frozen source snapshot is missing")
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != protocol["source_sha256"]:
        raise ValueError("Frozen source snapshot changed")
    driver = OUT / "source-driver.py"
    if not driver.is_file() or file_hash(driver) != protocol["driver_sha256"]:
        raise ValueError("Frozen driver changed")


def require_fit_files(directory, completed=False):
    names = ["checkpoint.pt", "metrics.json", "validation.npz", "history.json"]
    if completed:
        names.append("predictions.npz")
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete saved fit {directory}: {missing}")


def training_identity(training, cell, config, seed):
    if (
        training["configuration"] != config
        or training["test_evaluated"] is not False
        or training["model"] != cell["model"]
        or training["seed"] != seed
    ):
        raise ValueError("Invalid training resume")


def load(name, history_width, static_width):
    data = HistoryData.load(ROOT / ".cache/rich-history-v1.npz")
    data.history = data.history[..., :history_width].copy()
    data.static = data.static[..., :static_width].copy()
    masks, boundaries = history_split_masks(data)
    coverage = data.valid.sum(-1)
    normalizer = standardize_histories(data, masks["train"])
    parts = {
        split: partition(
            [data.history[mask], data.static[mask], data.valid[mask], data.domain[mask]],
            data.outcome[mask],
            data.row_ids[mask],
            {"time": data.time[mask], "domain": data.domain[mask], "history_count": coverage[mask]},
        )
        for split, mask in masks.items()
    }
    spec = {
        "history_features": history_width,
        "static_features": static_width,
        "team_size": 6,
        "history_length": 10,
        "classes": 3,
    }
    return Bundle(
        name, "history", parts, spec, {"boundaries": boundaries, "normalizer": normalizer}
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / "protocol.json"
    cache = ROOT / ".cache/rich-history-v1.npz"
    cache_record = read(cache.with_suffix(".json"))
    if file_hash(cache) != cache_record["sha256"]:
        raise ValueError("Rich history cache checksum mismatch")
    raw_name = "datasets/raw/overwatch.jsonl"
    raw_inputs = {raw_name: cache_record["identity"][raw_name]}
    # Freeze every builder/probe identity on first use. A snapshot resume instead
    # verifies the immutable protocol, cache manifest, archived source and raw
    # data, without requiring mutable working-copy Python files to stay unchanged.
    checked_inputs = raw_inputs if marker.exists() else cache_record["identity"]
    for name, checksum in checked_inputs.items():
        if file_hash(ROOT / name) != checksum:
            raise ValueError("Feature builder/input changed: " + name)
    cfg = {
        "device": "cpu",
        "threads": 2,
        "width": 16,
        "heads": 2,
        "layers": 1,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "scheduler": True,
        "batch_size": 256,
        "epochs": 12,
        "patience": 5,
        "norm_first": False,
        "tabular": False,
    }
    cells = [
        {
            "features": name,
            "history_width": hw,
            "static_width": sw,
            "model": model,
            "pooling": pooling,
        }
        for model in ["history_deepsets", "owo", "dawn"]
        for name, hw, sw in VARIANTS
        for pooling in (["global"] if model == "history_deepsets" else ["global", "team"])
    ]
    protocol = {
        "cells": cells,
        "seeds": SEEDS,
        "fits": len(cells) * len(SEEDS),
        "configuration": cfg,
        "source_sha256": code_hash(),
        "driver_sha256": file_hash(Path(__file__)),
        "runtime": runtime_identity(),
        "cache_sha256": file_hash(cache),
        "cache_manifest_sha256": file_hash(cache.with_suffix(".json")),
        "cache_generation_identity": cache_record["identity"],
        "raw_input_sha256": raw_inputs,
        "history_fields": RICH_HISTORY_FEATURES,
        "static_fields": RICH_STATIC_FEATURES,
        "execution_order": "Seed-major: all ten declared cells for seed 11, then 22, then 33; no outcome-based pruning.",
        "selection": "No configuration search. Fixed budget; validation NLL selects checkpoint. Validation-only temperature calibration.",
        "comparison": "Input original/rich crossed with global/team readout for OwO/DAWN; two input sets for History Deep Sets.",
        "fairness": "Same width, optimizer, epoch cap, seed and batches; enriched input and team readout change parameter counts.",
        "scope": "Exploratory after observed baseline tests and validation-only feature probes. Not original-paper reproduction or independent confirmation.",
    }
    if marker.exists():
        if read(marker) != protocol:
            raise ValueError("Frozen rich-history protocol changed; resume with its snapshot")
    else:
        write(marker, protocol)
        shutil.copytree(
            ROOT / "src/gamematch",
            OUT / "source-snapshot/gamematch",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copyfile(__file__, OUT / "source-driver.py")
    verify_snapshot(protocol)
    protocol_sha256 = file_hash(marker)
    # Both bundles occupy approximately 0.52 GiB in persistent input arrays.
    # Load/standardize each information set once; fit/evaluate treat them as
    # read-only, so ten cells do not need ten decompressions and normalizations.
    bundles = {name: load(name, hw, sw) for name, hw, sw in VARIANTS}
    complete = set()
    completed_seeds = {cell_name(cell): [] for cell in cells}
    for cell in cells:
        name = cell_name(cell)
        folder = OUT / name
        bundle = bundles[cell["features"]]
        identity = {
            "specification": bundle.specification,
            "provenance": bundle.provenance,
            "counts": {p: len(d.target) for p, d in bundle.parts.items()},
        }
        data_path = folder / "data.json"
        if data_path.exists() and read(data_path) != identity:
            raise ValueError("Data identity mismatch: " + name)
        write(data_path, identity)
        for seed in SEEDS:
            directory = folder / f"seed-{seed}"
            config = {**cfg, "pooling": cell["pooling"]}
            result_path = directory / "result.json"
            if result_path.exists():
                result = read(result_path)
                if (
                    result["protocol_sha256"] != protocol_sha256
                    or result["configuration"] != config
                    or result["seed"] != seed
                    or result["model"] != cell["model"]
                    or result["cell"] != cell
                    or result["test_evaluated"] is not True
                ):
                    raise ValueError("Invalid completed fit identity")
                require_fit_files(directory, completed=True)
                training_identity(read(directory / "metrics.json"), cell, config, seed)
                complete.add(str(directory.relative_to(OUT)))
                completed_seeds[name].append(seed)
        write(
            folder / "status.json",
            {
                "status": "completed" if completed_seeds[name] == SEEDS else "pending",
                "updated_utc": stamp(),
                "completed_seeds": completed_seeds[name],
                "expected_seeds": SEEDS,
            },
        )
    for cell, seed in execution_schedule(cells, SEEDS):
        name = cell_name(cell)
        folder = OUT / name
        directory = folder / f"seed-{seed}"
        if str(directory.relative_to(OUT)) in complete:
            continue
        bundle = bundles[cell["features"]]
        config = {**cfg, "pooling": cell["pooling"]}
        result_path = directory / "result.json"
        write(
            folder / "status.json",
            {
                "status": "running",
                "updated_utc": stamp(),
                "seed": seed,
                "completed_seeds": completed_seeds[name],
                "expected_seeds": SEEDS,
            },
        )
        try:
            write(
                OUT / "status.json",
                {
                    "status": "running",
                    "updated_utc": stamp(),
                    "cell": name,
                    "seed": seed,
                    "completed_fits": len(complete),
                },
            )
            print("FIT", name, seed, flush=True)
            if (directory / "metrics.json").exists():
                training = read(directory / "metrics.json")
                training_identity(training, cell, config, seed)
                require_fit_files(directory)
                model = reload_model(
                    directory, "history", cell["model"], bundle.specification, config
                )
            else:
                model, training = fit(
                    "history",
                    cell["model"],
                    bundle.specification,
                    bundle.parts["train"],
                    bundle.parts["valid"],
                    config,
                    seed,
                    directory,
                )
            reloaded = reload_model(
                directory, "history", cell["model"], bundle.specification, config
            )
            first = predict_raw(model, bundle.parts["valid"], "history", config)
            second = predict_raw(reloaded, bundle.parts["valid"], "history", config)
            error = float(np.max(np.abs(first - second)))
            if not np.isfinite(error) or error > 1e-5:
                raise ValueError("Checkpoint reload changed predictions")
            evaluated = evaluate(reloaded, bundle, config, directory)
            with np.load(directory / "predictions.npz", allow_pickle=False) as values:
                y, p = values["y_true"], values["probabilities"]
                pred = p.argmax(1)
            evaluated["classification_diagnostics"] = {
                "predicted_counts": np.bincount(pred, minlength=3).tolist(),
                "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            }
            result = {
                **training,
                **evaluated,
                "test_evaluated": True,
                "cell": cell,
                "protocol_sha256": protocol_sha256,
                "reload_max_abs_error": error,
                "completed_utc": stamp(),
            }
            write(result_path, result)
            complete.add(str(directory.relative_to(OUT)))
            completed_seeds[name].append(seed)
            completed_seeds[name].sort(key=SEEDS.index)
            print(
                "DONE",
                name,
                seed,
                result["metrics"]["log_loss"],
                result["metrics"]["accuracy"],
                flush=True,
            )
            del model, reloaded
        except Exception as error:
            write(
                folder / "status.json",
                {
                    "status": "failed",
                    "updated_utc": stamp(),
                    "seed": seed,
                    "completed_seeds": completed_seeds[name],
                    "error": type(error).__name__ + ": " + str(error),
                },
            )
            raise
        write(
            folder / "status.json",
            {
                "status": "completed" if completed_seeds[name] == SEEDS else "pending",
                "updated_utc": stamp(),
                "completed_seeds": completed_seeds[name],
                "expected_seeds": SEEDS,
            },
        )
    write(
        OUT / "status.json",
        {
            "status": "completed",
            "updated_utc": stamp(),
            "completed_fits": len(complete),
            "fits": [
                f"{cell_name(cell)}/seed-{seed}" for cell, seed in execution_schedule(cells, SEEDS)
            ],
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write(
            OUT / "status.json",
            {
                "status": "failed",
                "updated_utc": stamp(),
                "error": type(error).__name__ + ": " + str(error),
            },
        )
        raise
