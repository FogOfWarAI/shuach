# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Independently recompute stored proper scores; verify frozen source snapshots."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def source_hash(root):
    digest = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        digest.update(str(p.relative_to(root)).encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def audit(run):
    manifest = json.loads((run / "manifest.json").read_text())
    expected = manifest.get("code_sha256", manifest.get("code"))
    snapshot = run / "source-snapshot/gamematch"
    code_ok = expected == source_hash(snapshot) if expected else None
    if expected is None:
        identity = {"config": manifest["config"], "code": source_hash(snapshot)}
        if "data" in manifest:
            identity["data"] = manifest["data"]
        candidate = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        code_ok = candidate == manifest.get("fingerprint")
    rows = []
    for path in sorted(run.rglob("metrics.json")):
        pred = path.with_name("predictions.npz")
        if not pred.exists():
            raise ValueError(f"Missing probabilities: {path}")
        with np.load(pred, allow_pickle=False) as f:
            data = dict(f)
        y = data["y_true"]
        if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
            raise ValueError(f"Invalid evaluation labels: {pred}")
        if data["row_id"].shape != y.shape:
            raise ValueError(f"Row identities do not align with labels: {pred}")
        if len(np.unique(data["row_id"])) != len(y):
            raise ValueError(f"Duplicate row IDs: {pred}")
        if "prediction" in data:
            if data["prediction"].shape != y.shape:
                raise ValueError(f"Regression output does not align with labels: {pred}")
            error = data["prediction"].astype(float) - y.astype(float)
            if not np.isfinite(error).all():
                raise ValueError(f"Nonfinite regression predictions: {pred}")
            calculated = {
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "mean_error": float(error.mean()),
            }
        elif "p_positive" in data:
            p = data["p_positive"].astype(float)
            if (
                p.shape != y.shape
                or not np.isfinite(p).all()
                or ((p < 0) | (p > 1)).any()
                or not np.isin(y, [0, 1]).all()
            ):
                raise ValueError(f"Invalid binary probabilities or labels: {pred}")
            ep = np.finfo(float).eps
            p = np.clip(p, ep, 1 - ep)
            calculated = {
                "log_loss": float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
                "accuracy": float(((p >= 0.5) == y).mean()),
                "brier": float(np.square(p - y).mean()),
            }
        else:
            if not np.equal(y, np.floor(y)).all():
                raise ValueError(f"Noninteger class labels: {pred}")
            y = y.astype(int)
            p = data["probabilities"].astype(float)
            if (
                p.ndim != 2
                or p.shape[0] != len(y)
                or p.shape[1] < 2
                or not np.allclose(p.sum(1), 1, atol=1e-6)
                or not np.isfinite(p).all()
                or ((p < 0) | (p > 1)).any()
                or ((y < 0) | (y >= p.shape[1])).any()
            ):
                raise ValueError(f"Invalid simplex: {pred}")
            calculated = {
                "log_loss": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
                "accuracy": float((p.argmax(1) == y).mean()),
                "brier_multiclass_sum": float(np.square(p - np.eye(p.shape[1])[y]).sum(1).mean()),
            }
        result = json.loads(path.read_text())
        stored = result["metrics"].get("evaluation", result["metrics"])
        if not all(np.isfinite(stored[k]) for k in calculated):
            raise ValueError(f"Nonfinite stored metrics: {path}")
        errors = {k: abs(v - stored[k]) for k, v in calculated.items()}
        if max(errors.values()) > 1e-7:
            raise ValueError(f"Metric mismatch: {path}: {errors}")
        rows.append(
            {
                "artifact": str(path.relative_to(run)),
                "examples": len(y),
                "max_absolute_error": max(errors.values()),
            }
        )
    if code_ok is False:
        raise ValueError(f"Frozen source hash mismatch: {run}")
    return {
        "run": str(run),
        "status": manifest["status"],
        "audited_model_jobs": len(rows),
        "snapshot_code_sha256": source_hash(snapshot),
        "snapshot_matches_manifest": code_ok,
        "max_absolute_metric_error": max((r["max_absolute_error"] for r in rows), default=0),
        "checks": [
            "Unique row identifiers",
            "Stored probability simplex",
            "Independent NumPy log loss, accuracy and Brier, or regression MAE/RMSE/bias",
            "Source snapshot hash when recorded",
        ],
        "jobs": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    value = [audit(run) for run in args.runs]
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    for v in value:
        print(v["run"], v["audited_model_jobs"], v["max_absolute_metric_error"])


if __name__ == "__main__":
    main()
