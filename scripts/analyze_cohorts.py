# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Descriptive availability cohorts; no subgroup hypothesis selection or refitting."""

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gamematch.evaluation.comparison import load_predictions, per_example_loss


def read(path):
    return json.loads(path.read_text())


def summarize(name, reference_path, runs, metric):
    reference = load_predictions(reference_path)
    if metric == "log_loss":
        limited = reference["completed_history"][:, 0] == 0
        cohorts = {
            "at_least_one_first_observed_player": limited,
            "all_players_previously_observed": ~limited,
        }
    else:
        limited = reference["low_history"].astype(bool)
        cohorts = {"limited_observed_history": limited, "other_returning_players": ~limited}
    rows, artifacts = [], []
    for run_name in runs:
        run = Path("runs") / run_name
        manifest = read(run / "manifest.json")
        expected = sorted(manifest["config"].get("seeds", [0]))
        grouped = defaultdict(list)
        for path in sorted(run.rglob("metrics.json")):
            result = read(path)
            model = result["model"]
            if "separate_domains" in result:
                model += "-separate" if result["separate_domains"] else "-pooled"
            grouped[model].append((result.get("seed", 0), path))
        for model, pairs in sorted(grouped.items()):
            if sorted(seed for seed, _ in pairs) != expected:
                continue
            values = defaultdict(list)
            for _, path in sorted(pairs):
                prediction = path.with_name("predictions.npz")
                data = load_predictions(prediction)
                for field in ["row_id", "y_true", "time"]:
                    np.testing.assert_array_equal(data[field], reference[field])
                losses = per_example_loss(data, metric)
                for cohort, mask in cohorts.items():
                    if mask.any():
                        values[cohort].append(float(losses[mask].mean()))
                artifacts.append(
                    {
                        "path": str(prediction),
                        "sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
                    }
                )
            for cohort, losses in values.items():
                rows.append(
                    {
                        "run": run_name,
                        "model": model,
                        "cohort": cohort,
                        "n_examples": int(cohorts[cohort].sum()),
                        "seeds": expected,
                        "metric": metric,
                        "mean": float(np.mean(losses)),
                        "seed_sample_sd": float(np.std(losses, ddof=1))
                        if len(losses) > 1
                        else None,
                        "values": losses,
                    }
                )
    return {
        "name": name,
        "cohort_source": str(reference_path),
        "cohort_source_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "test_examples": len(reference["row_id"]),
        "cohort_counts": {key: int(mask.sum()) for key, mask in cohorts.items()},
        "cohort_descriptions": {
            key: {
                "calendar_days": int(len(np.unique(np.floor(reference["time"][mask] / 86400)))),
                **(
                    {
                        "outcome_counts_loss_draw_win": np.bincount(
                            reference["y_true"][mask].astype(int), minlength=3
                        ).tolist(),
                        "domain_counts": np.bincount(
                            reference["domain"][mask].astype(int), minlength=2
                        ).tolist(),
                    }
                    if metric == "log_loss"
                    else {
                        "unique_accounts": int(len(np.unique(reference["player_id"][mask]))),
                        "unique_matches": int(len(np.unique(reference["match_id"][mask]))),
                    }
                ),
            }
            for key, mask in cohorts.items()
        },
        "rows": rows,
        "artifacts": artifacts,
    }


def main():
    tasks = [
        (
            "overwatch-primary",
            Path("runs/overwatch-ratings-v1/elo-pooled/predictions.npz"),
            ["overwatch-ratings-v1", "overwatch-history-v1"],
            "log_loss",
        ),
        (
            "overwatch-earlier-origin",
            Path("runs/overwatch-ratings-earlier-origin-v1/elo-pooled/predictions.npz"),
            ["overwatch-ratings-earlier-origin-v1", "overwatch-history-earlier-origin-v1"],
            "log_loss",
        ),
        (
            "future-rating",
            Path("runs/skill-v1/seed-0-current/predictions.npz"),
            ["skill-v1"],
            "absolute_error",
        ),
    ]
    groups = [summarize(*task) for task in tasks if task[1].exists()]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-hoc descriptive stratification by already recorded history-availability flags. All complete model/seed sets retained. Not independent confirmation, subgroup significance testing, verified novice identity, or causal history effects. Limited-history future-rating test cohort contains only 21 observations.",
        "groups": groups,
    }
    Path("reports/cohort-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print({g["name"]: g["cohort_counts"] for g in groups})


if __name__ == "__main__":
    main()
