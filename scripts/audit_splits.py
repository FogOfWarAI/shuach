# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Audit saved split identities and prediction membership without using adapters."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_split(path, manifest):
    with np.load(path, allow_pickle=False) as handle:
        data = dict(handle)
    if "train_row_ids" in data:
        ids = {part: data[f"{part}_row_ids"] for part in ["train", "valid", "test"]}
        groups = {part: set(data[f"{part}_groups"].tolist()) for part in ids}
        for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
            if groups[a] & groups[b]:
                raise ValueError(f"Overlapping roster groups: {path}: {a}/{b}")
        reference = {"row_id": ids["test"], "group_id": data["test_groups"]}
        expected_counts = read(path.with_name("data-manifest.json"))["rows"]
        timing = None
        excluded = None
    else:
        rows = data["row_id"]
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"Duplicate source row IDs: {path}")
        masks = {part: data[part] for part in ["train", "valid", "test"]}
        for part, mask in masks.items():
            if mask.dtype != np.bool_ or mask.shape != rows.shape:
                raise ValueError(f"Malformed split mask: {path}: {part}")
        if (sum(mask.astype(int) for mask in masks.values()) > 1).any():
            raise ValueError(f"An observation occupies multiple partitions: {path}")
        ids = {part: rows[mask] for part, mask in masks.items()}
        bounds = manifest.get("time_boundaries")
        if bounds is None:
            bounds = read(path.with_name("features.json"))["time_boundaries"]
        time = data["time"]
        delivery_name = "target_available_at" if "target_available_at" in data else "available_at"
        delivery = data[delivery_name]
        if (
            not np.isfinite(time).all()
            or not np.isfinite(delivery).all()
            or (delivery < time).any()
        ):
            raise ValueError(f"Invalid prediction/target availability times: {path}")
        rules = {
            "train": (time < bounds[0]) & (delivery < bounds[0]),
            "valid": (time >= bounds[0]) & (time < bounds[1]) & (delivery < bounds[1]),
            "test": time >= bounds[1],
        }
        for part, mask in masks.items():
            if (mask & ~rules[part]).any():
                raise ValueError(f"Temporal or availability boundary violation: {path}: {part}")
            cap = manifest["config"].get("max_train_rows" if part == "train" else "max_eval_rows")
            if cap is None and not np.array_equal(mask, rules[part]):
                raise ValueError(f"Unexplained temporal row exclusion: {path}: {part}")
        reference = {
            key: data[key][masks["test"]]
            for key in [
                "row_id",
                "player_id",
                "match_id",
                "time",
                "domain",
                "y_true",
                "target_available_at",
            ]
            if key in data
        }
        expected_counts = manifest.get("split_counts")
        if expected_counts and "train_completed" in expected_counts:
            expected_counts = {
                "train": expected_counts["train_completed"],
                "valid": expected_counts["calibration"],
                "test": expected_counts["test"],
            }
        timing = {
            "boundaries": bounds,
            "delivery_field": delivery_name,
            "strict_boundary_checks": True,
        }
        excluded = int(len(rows) - sum(len(value) for value in ids.values()))
    for part, values in ids.items():
        if not len(values) or len(np.unique(values)) != len(values):
            raise ValueError(f"Empty or duplicate partition identities: {path}: {part}")
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        if set(ids[a].tolist()) & set(ids[b].tolist()):
            raise ValueError(f"Overlapping partition row IDs: {path}: {a}/{b}")
    counts = {part: len(values) for part, values in ids.items()}
    if expected_counts and any(counts[part] != expected_counts[part] for part in counts):
        raise ValueError(f"Split counts differ from declared manifest: {path}")
    order = np.argsort(reference["row_id"])
    reference = {key: value[order] for key, value in reference.items()}
    checks, labels = [], reference.get("y_true")
    for metrics in sorted(path.parent.rglob("metrics.json")):
        prediction = metrics.with_name("predictions.npz")
        with np.load(prediction, allow_pickle=False) as saved:
            values = dict(saved)
        order = np.argsort(values["row_id"])
        for key, expected in reference.items():
            if key not in values or not np.array_equal(values[key][order], expected):
                raise ValueError(
                    f"Prediction differs from saved test partition: {prediction}: {key}"
                )
        if labels is None:
            labels = values["y_true"][order]
        elif not np.array_equal(labels, values["y_true"][order]):
            raise ValueError(f"Test labels differ between model jobs: {prediction}")
        checks.append({"prediction": str(prediction), "sha256": digest(prediction)})
    return {
        "split": str(path),
        "sha256": digest(path),
        "counts": counts,
        "excluded_boundary_rows": excluded,
        "temporal_checks": timing,
        "roster_groups_disjoint": "train_groups" in data,
        "completed_prediction_files_checked": len(checks),
        "predictions": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/split-audit.json"))
    args = parser.parse_args()
    records = []
    for run in args.runs:
        manifest = read(run / "manifest.json")
        for path in sorted(run.rglob("split-manifest.npz")):
            records.append(audit_split(path, manifest))
    if not records:
        raise ValueError("No supported split artifacts were found")
    report = {
        "status": "passed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved-artifact audit independent of data adapters: unique and disjoint row/group membership, strict time/target-availability boundaries, complete prediction membership and consistent test labels. It does not independently authenticate source timestamps or reconstruct every input feature. Synthetic splits are checked by audit_decisions.py.",
        "split_artifacts_checked": len(records),
        "prediction_files_checked": sum(r["completed_prediction_files_checked"] for r in records),
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Passed",
        len(records),
        "split artifacts and",
        report["prediction_files_checked"],
        "prediction files",
    )


if __name__ == "__main__":
    main()
