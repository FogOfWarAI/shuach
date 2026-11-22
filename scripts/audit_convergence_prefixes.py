# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Check whether extended-budget runs reproduce their primary training prefixes."""

import json
from pathlib import Path

import numpy as np


def main():
    pairs = [
        (Path("runs/neuralac-convergence-v1"), Path("runs/local-full-heroes-v1")),
        (Path("runs/decisions-convergence-v1"), Path("runs/decisions-v1")),
    ]
    checks = []
    for extended, primary in pairs:
        for path in sorted(extended.rglob("metrics.json")):
            relative = path.relative_to(extended)
            baseline_path = primary / relative
            a = json.loads(path.with_name("history.json").read_text())
            b = json.loads(baseline_path.with_name("history.json").read_text())
            count = min(len(a), len(b))
            errors = {
                field: float(
                    np.max(
                        np.abs(
                            np.array([row[field] for row in a[:count]])
                            - np.array([row[field] for row in b[:count]])
                        )
                    )
                )
                for field in ["train_bce", "valid_log_loss"]
            }
            if count < len(b) or max(errors.values()) > 1e-7:
                raise ValueError(
                    f"Extended run does not reproduce primary prefix: {path}: {errors}"
                )
            with (
                np.load(path.with_name("predictions.npz"), allow_pickle=False) as current,
                np.load(baseline_path.with_name("predictions.npz"), allow_pickle=False) as before,
            ):
                for field in ["row_id", "group_id", "y_true"]:
                    np.testing.assert_array_equal(current[field], before[field])
            shared_artifacts = []
            for filename in ["split-manifest.npz", "data.npz", "decision-environment.npz"]:
                current_path = path.parent.parent / filename
                original_path = baseline_path.parent.parent / filename
                if not current_path.exists() and not original_path.exists():
                    continue
                with (
                    np.load(current_path, allow_pickle=False) as current,
                    np.load(original_path, allow_pickle=False) as before,
                ):
                    if set(current.files) != set(before.files):
                        raise ValueError(f"Shared experiment schema changed: {filename}")
                    for field in current.files:
                        np.testing.assert_array_equal(current[field], before[field])
                shared_artifacts.append(filename)
            hidden = path.parent.parent / "hidden-mechanism.json"
            if hidden.exists():
                original_hidden = baseline_path.parent.parent / hidden.name
                if json.loads(hidden.read_text()) != json.loads(original_hidden.read_text()):
                    raise ValueError("Hidden mechanism changed in the budget extension")
                shared_artifacts.append(hidden.name)
            checks.append(
                {
                    "extended_artifact": str(path),
                    "primary_artifact": str(baseline_path),
                    "prefix_epochs": count,
                    "max_absolute_loss_differences": errors,
                    "test_rows_groups_labels_identical": True,
                    "identical_shared_artifacts": shared_artifacts,
                }
            )
    report = {
        "status": "passed",
        "completed_jobs_checked": len(checks),
        "checks": checks,
        "scope": "Compares all original training/validation loss entries, saved split/data arrays, exact evaluation identities, hidden mechanism, lobby environment and sampled candidate budgets for each completed extension, where applicable. Wall-clock times excluded. This checks reproducibility of the shared experiment and training prefix, not global convergence.",
    }
    Path("reports/convergence-prefix-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Verified", len(checks), "completed convergence prefixes")


if __name__ == "__main__":
    main()
