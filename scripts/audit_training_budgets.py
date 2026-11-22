# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Describe stopping and reference budgets without treating them as convergence proofs."""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RUNS = [
    "local-full-heroes-v1",
    "overwatch-history-v1",
    "asymmetric-v1",
    "skill-v1",
    "decisions-v1",
    "neuralac-ablations-v1",
    "overwatch-target-only-v1",
    "neuralac-convergence-v1",
    "decisions-convergence-v1",
    "overwatch-history-earlier-origin-v1",
]


def main():
    groups = defaultdict(list)
    for name in RUNS:
        run = Path("runs") / name
        if not (run / "manifest.json").exists():
            continue
        config = json.loads((run / "manifest.json").read_text())["config"]
        for metrics in sorted(run.rglob("metrics.json")):
            curve = metrics.with_name("history.json")
            if not curve.exists():
                continue
            history = json.loads(curve.read_text())
            if not history:
                continue
            result = json.loads(metrics.read_text())
            model = result["model"]
            cap = config["training"]["epochs"]
            cap = config.get("model_training", {}).get(model, {}).get("epochs", cap)
            dataset = result.get("dataset", "overwatch" if "overwatch" in name else "future-rating")
            if name.startswith("decisions"):
                dataset = metrics.parent.parent.name.split("-seed-")[0]
            records = {
                "artifact": str(metrics),
                "seed": result["seed"],
                "epoch_cap": cap,
                "epochs_completed": len(history),
                "selected_epoch": result.get("best_epoch"),
                "reached_epoch_cap": len(history) == cap,
                "selected_final_allowed_epoch": result.get("best_epoch") == cap,
            }
            groups[(name, dataset, model)].append(records)
    summaries = []
    for (run, dataset, model), records in sorted(groups.items()):
        summaries.append(
            {
                "run": run,
                "dataset": dataset,
                "model": model,
                "completed_jobs": len(records),
                "jobs_reaching_cap": sum(r["reached_epoch_cap"] for r in records),
                "jobs_selecting_cap": sum(r["selected_final_allowed_epoch"] for r in records),
                "mean_selected_epoch": float(np.mean([r["selected_epoch"] for r in records])),
                "jobs": records,
            }
        )
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Completed neural jobs only; hitting a cap motivates a budget sensitivity, while patience stopping does not prove global optimization or exhaustive hyperparameter search. Concurrent duration is not used as an isolated speed measure.",
        "summaries": summaries,
    }
    Path("reports/training-budget-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Audited", sum(s["completed_jobs"] for s in summaries), "neural training histories")


if __name__ == "__main__":
    main()
