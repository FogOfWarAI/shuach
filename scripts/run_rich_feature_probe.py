# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Freeze and independently audit a train/validation-only rich-feature probe.

Trusted pipeline artifacts are stored separately from the earlier exploratory
table. No test labels or test predictions enter candidate fitting or scoring.
"""

import hashlib
import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import gamematch
from gamematch.configuration import runtime_identity
from gamematch.data.heroes import file_hash
from gamematch.data.history import HistoryData, history_split_masks
from gamematch.evaluation.multiclass import multiclass_metrics
from gamematch.models.tabular import require_logistic_convergence
from gamematch.runner import code_hash

ROOT = Path(os.environ.get("GAMEMATCH_WORKSPACE", Path(__file__).resolve().parents[1])).resolve()
OUT = Path(
    os.environ.get("GAMEMATCH_PROBE_OUTPUT", ROOT / "reports/richer-history/probe-v2")
).resolve()
VARIANTS = [
    ("original", 5, 13),
    ("richer_sequence", 8, 13),
    ("richer_static", 5, 46),
    ("richer_both", 8, 46),
]
REGULARIZATION = [0.01, 0.1, 1.0]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def source_hash(source):
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_aggregation():
    archived = Path(__file__).parent / "source-support/audit_rich_history.py"
    helper = archived if archived.exists() else ROOT / "scripts/audit_rich_history.py"
    specification = importlib.util.spec_from_file_location("probe_aggregation", helper)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.team_features, helper


def classification_metrics(target, probability):
    metrics = multiclass_metrics(target, probability)
    predicted = probability.argmax(1)
    metrics.update(
        balanced_accuracy=float(balanced_accuracy_score(target, predicted)),
        macro_f1=float(f1_score(target, predicted, average="macro", zero_division=0)),
    )
    return metrics


def validation_prediction(model, features, batch_size=257):
    if not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError("The probe requires ordered loss/draw/win probability columns")
    return np.concatenate(
        [
            model.predict_proba(features[start : start + batch_size])
            for start in range(0, len(features), batch_size)
        ]
    )


def audit_candidate(directory, protocol_hash, variant, c, valid, features):
    record = read(directory / "result.json")
    if (
        record["protocol_sha256"] != protocol_hash
        or record["features"] != variant
        or record["C"] != c
        or record["test_evaluated"] is not False
    ):
        raise ValueError("Completed candidate identity changed: " + str(directory))
    for name, field in [
        ("checkpoint.joblib", "checkpoint_sha256"),
        ("validation.npz", "validation_sha256"),
    ]:
        if file_hash(directory / name) != record[field]:
            raise ValueError("Completed candidate artifact changed: " + str(directory / name))
    with np.load(directory / "validation.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["row_id"], valid.row_ids) or not np.array_equal(
            saved["y_true"], valid.outcome
        ):
            raise ValueError("Saved validation identities differ from the declared partition")
        probability = saved["probabilities"]
    metrics = classification_metrics(valid.outcome, probability)
    if metrics != record["metrics"]:
        raise ValueError("Saved metrics do not match validation predictions")
    counts = np.bincount(probability.argmax(1), minlength=3).tolist()
    if counts != record["predicted_class_counts"]:
        raise ValueError("Saved predicted class counts differ")
    # The artifact is locally produced, frozen and hash-checked before joblib loading.
    restored = joblib.load(directory / "checkpoint.joblib")
    iterations = require_logistic_convergence(restored[-1])
    reloaded = validation_prediction(restored, features)
    error = float(np.max(np.abs(probability - reloaded)))
    saved_error = record["reload_max_abs_error"]
    if (
        iterations != record["iterations"]
        or not np.isfinite(error)
        or error > 1e-10
        or not np.isfinite(saved_error)
        or saved_error > 1e-10
    ):
        raise ValueError("Reloaded pipeline differs from the saved candidate")
    return record, error


def initialize_protocol(protocol, helper):
    marker = OUT / "protocol.json"
    if marker.exists():
        if read(marker) != protocol:
            raise ValueError("Probe identity changed; use its frozen source or a new output")
        if (
            source_hash(OUT / "source-snapshot/gamematch") != protocol["source_sha256"]
            or file_hash(OUT / "source-driver.py") != protocol["driver_sha256"]
            or file_hash(OUT / "source-support/audit_rich_history.py")
            != protocol["aggregation_helper_sha256"]
        ):
            raise ValueError("Frozen probe source changed")
        return marker
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError("Nonempty unsealed probe output; select a new output directory")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        Path(gamematch.__file__).parent,
        OUT / "source-snapshot/gamematch",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (OUT / "source-support").mkdir()
    shutil.copyfile(helper, OUT / "source-support/audit_rich_history.py")
    shutil.copyfile(__file__, OUT / "source-driver.py")
    if source_hash(OUT / "source-snapshot/gamematch") != protocol["source_sha256"]:
        raise ValueError("Package source changed while its snapshot was being written")
    write(marker, protocol)
    return marker


def main():
    team_features, helper = load_aggregation()
    cache = ROOT / ".cache/rich-history-v1.npz"
    cache_marker = cache.with_suffix(".json")
    cache_record = read(cache_marker)
    if file_hash(cache) != cache_record["sha256"]:
        raise ValueError("Feature cache checksum changed")
    for name, expected in cache_record["identity"].items():
        if name.startswith("src/gamematch/"):
            current = Path(gamematch.__file__).parent / name.removeprefix("src/gamematch/")
        elif name == "scripts/audit_rich_history.py":
            current = helper
        else:
            current = ROOT / name
        if file_hash(current) != expected:
            raise ValueError("Feature cache builder/input changed: " + name)
    reference_path = ROOT / "reports/richer-history/validation-ablation.json"
    legacy = read(reference_path)
    if legacy["identity"] != cache_record["identity"] or legacy["test_predictions_evaluated"]:
        raise ValueError("Legacy probe does not identify this validation-only feature cache")
    protocol = {
        "schema_version": 1,
        "variants": [
            dict(name=name, history_width=hw, static_width=sw) for name, hw, sw in VARIANTS
        ],
        "C": REGULARIZATION,
        "pipeline": "StandardScaler fitted on train, then LogisticRegression(C=C, max_iter=2000, solver=lbfgs)",
        "threads": 2,
        "selection": "Minimum validation log_loss within each input variant; C breaks ties",
        "test_evaluated": False,
        "source_sha256": code_hash(),
        "driver_sha256": file_hash(Path(__file__)),
        "aggregation_helper_sha256": file_hash(helper),
        "runtime": runtime_identity(),
        "cache_sha256": file_hash(cache),
        "cache_manifest_sha256": file_hash(cache_marker),
        "legacy_reference_sha256": file_hash(reference_path),
        "reference_check": "Compare all twelve candidates after fitting and selection; numeric metrics atol=1e-8, rtol=0, predicted counts exact; reference scores never select a candidate",
        "scope": "Exploratory validation-only probe with reused validation labels for regularization selection; no neural-model or independent-confirmation performance claim",
    }
    marker = initialize_protocol(protocol, helper)
    protocol_hash = file_hash(marker)
    data = HistoryData.load(cache)
    masks, boundaries = history_split_masks(data)
    # Only these two partitions reach feature aggregation, fitting and scoring.
    subsets = {
        split: HistoryData(**{name: values[masks[split]] for name, values in data.__dict__.items()})
        for split in ["train", "valid"]
    }
    del data, masks
    train, valid = subsets["train"], subsets["valid"]
    if not len(train.outcome) or not len(valid.outcome):
        raise ValueError("Nonempty training and validation partitions are required")
    split_path = OUT / "split-manifest.npz"
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["train_row_id"], train.row_ids) or not np.array_equal(
                saved["valid_row_id"], valid.row_ids
            ):
                raise ValueError("Probe split identities changed")
    else:
        np.savez_compressed(split_path, train_row_id=train.row_ids, valid_row_id=valid.row_ids)
    write(
        OUT / "status.json",
        {"status": "running", "updated_utc": stamp(), "completed_fits": 0, "expected_fits": 12},
    )
    results, reload_errors = [], []
    try:
        with threadpool_limits(limits=2):
            for variant, hwidth, swidth in VARIANTS:
                x_train = team_features(train, hwidth, swidth)
                x_valid = team_features(valid, hwidth, swidth)
                if not np.isfinite(x_train).all() or not np.isfinite(x_valid).all():
                    raise ValueError("Nonfinite probe features")
                candidates = []
                for c in REGULARIZATION:
                    directory = OUT / variant / f"C-{c:g}"
                    path = directory / "result.json"
                    if not path.exists():
                        if directory.exists() and any(directory.iterdir()):
                            raise ValueError(
                                "Incomplete candidate artifacts; use a new output: "
                                + str(directory)
                            )
                        directory.mkdir(parents=True, exist_ok=True)
                        model = make_pipeline(
                            StandardScaler(), LogisticRegression(C=c, max_iter=2000)
                        )
                        model.fit(x_train, train.outcome)
                        iterations = require_logistic_convergence(model[-1])
                        probability = model.predict_proba(x_valid)
                        if not np.array_equal(model.classes_, [0, 1, 2]):
                            raise ValueError("Missing or reordered training outcome classes")
                        metrics = classification_metrics(valid.outcome, probability)
                        joblib.dump(model, directory / "checkpoint.joblib")
                        np.savez_compressed(
                            directory / "validation.npz",
                            row_id=valid.row_ids,
                            y_true=valid.outcome,
                            probabilities=probability,
                        )
                        restored = joblib.load(directory / "checkpoint.joblib")
                        error = float(
                            np.max(np.abs(validation_prediction(restored, x_valid) - probability))
                        )
                        if not np.isfinite(error) or error > 1e-10:
                            raise ValueError("Reloaded pipeline predictions changed")
                        write(
                            path,
                            {
                                "features": variant,
                                "C": c,
                                "iterations": iterations,
                                "width": int(x_train.shape[1]),
                                "metrics": metrics,
                                "predicted_class_counts": np.bincount(
                                    probability.argmax(1), minlength=3
                                ).tolist(),
                                "test_evaluated": False,
                                "protocol_sha256": protocol_hash,
                                "checkpoint_sha256": file_hash(directory / "checkpoint.joblib"),
                                "validation_sha256": file_hash(directory / "validation.npz"),
                                "reload_max_abs_error": error,
                                "completed_utc": stamp(),
                            },
                        )
                    record, error = audit_candidate(
                        directory, protocol_hash, variant, c, valid, x_valid
                    )
                    candidates.append(record)
                    reload_errors.append(error)
                    write(
                        OUT / "status.json",
                        {
                            "status": "running",
                            "updated_utc": stamp(),
                            "completed_fits": len(reload_errors),
                            "expected_fits": 12,
                        },
                    )
                    print("AUDITED", variant, c, record["metrics"]["log_loss"], flush=True)
                selected = min(candidates, key=lambda r: (r["metrics"]["log_loss"], r["C"]))
                selection = {
                    "features": variant,
                    "C": selected["C"],
                    "validation_log_loss": selected["metrics"]["log_loss"],
                    "protocol_sha256": protocol_hash,
                    "test_used": False,
                }
                selection_path = OUT / variant / "selection.json"
                if selection_path.exists() and read(selection_path) != selection:
                    raise ValueError("Locked probe selection changed")
                if not selection_path.exists():
                    write(selection_path, selection)
                results.append(
                    {"features": variant, "candidates": candidates, "selected": selected}
                )
        reference = {item["features"]: item for item in legacy["ablations"]}
        comparisons, reference_consistent = [], True
        selected_configs = []
        for group in results:
            same_selection = group["selected"]["C"] == reference[group["features"]]["selected"]["C"]
            selected_configs.append(
                {
                    "features": group["features"],
                    "C": group["selected"]["C"],
                    "legacy_C": reference[group["features"]]["selected"]["C"],
                    "same_selection": same_selection,
                }
            )
            reference_consistent &= same_selection
            earlier = {item["C"]: item for item in reference[group["features"]]["candidates"]}
            for candidate in group["candidates"]:
                old = earlier[candidate["C"]]
                differences = {
                    name: candidate["metrics"][name] - value
                    for name, value in old["metrics"].items()
                    if isinstance(value, (int, float))
                }
                counts_equal = candidate["predicted_class_counts"] == old["predicted_class_counts"]
                consistent = counts_equal and all(
                    abs(value) <= 1e-8 for value in differences.values()
                )
                reference_consistent &= consistent
                comparisons.append(
                    {
                        "features": group["features"],
                        "C": candidate["C"],
                        "new_minus_legacy": differences,
                        "counts_equal": counts_equal,
                        "within_declared_tolerance": consistent,
                    }
                )
        report = {
            "status": "completed",
            "generated_utc": stamp(),
            "test_evaluated": False,
            "protocol_sha256": protocol_hash,
            "split_manifest_sha256": file_hash(split_path),
            "partition_counts": {"train": len(train.outcome), "valid": len(valid.outcome)},
            "split_boundaries": boundaries,
            "audited_fits": len(reload_errors),
            "max_reload_absolute_error": max(reload_errors),
            "ablations": results,
            "legacy_comparison": comparisons,
            "legacy_selected_configurations": selected_configs,
            "legacy_metrics_consistent": reference_consistent,
            "scope": protocol["scope"],
        }
        write(OUT / "results.json", report)
        write(
            OUT / "status.json",
            {
                "status": "completed",
                "updated_utc": stamp(),
                "completed_fits": 12,
                "expected_fits": 12,
                "legacy_metrics_consistent": reference_consistent,
            },
        )
        print(
            "COMPLETED",
            json.dumps(
                {
                    "fits": 12,
                    "legacy_metrics_consistent": reference_consistent,
                    "max_reload_absolute_error": max(reload_errors),
                }
            ),
            flush=True,
        )
    except Exception as error:
        write(
            OUT / "status.json",
            {
                "status": "failed",
                "updated_utc": stamp(),
                "completed_fits": len(reload_errors),
                "expected_fits": 12,
                "error": type(error).__name__ + ": " + str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
