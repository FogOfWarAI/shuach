# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Describe saved calibration fits and marginal probabilities, without refitting.

This post-hoc diagnostic is conditional on the completed primary/calibration
artifacts. Matching the overall positive rate is not sufficient for calibration.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/hero-calibration-v1"
    marker = run / "manifest.json"
    manifest = json.loads(marker.read_text())
    if manifest["status"] != "completed":
        raise ValueError("All declared validation fits must be complete")
    config = manifest["config"]
    artifacts = {}

    def register(path):
        artifacts[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def load(path):
        register(path)
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    register(marker)
    register(Path(__file__).resolve())
    records, summaries = [], []
    for dataset in config["datasets"]:
        for model in config["models"]:
            values = []
            for seed in config["seeds"]:
                folder = run / dataset / f"seed-{seed}-{model}-affine"
                path = folder / "metrics.json"
                register(path)
                result = json.loads(path.read_text())
                params = result["parameters"]
                valid = load(folder / "validation.npz")
                raw = load(root / result["source_primary"] / "predictions.npz")
                calibrated = load(folder / "predictions.npz")
                for key in ["row_id", "y_true", "group_id"]:
                    np.testing.assert_array_equal(raw[key], calibrated[key])
                eps = config["probability_clip_for_logit"]
                vp = valid["p_positive"].astype(float)
                tp = raw["p_positive"].astype(float)
                vc = expit(params["slope"] * logit(np.clip(vp, eps, 1 - eps)) + params["intercept"])
                tc = expit(params["slope"] * logit(np.clip(tp, eps, 1 - eps)) + params["intercept"])
                np.testing.assert_allclose(tc, calibrated["p_positive"], rtol=0, atol=1e-14)
                row = {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "slope": params["slope"],
                    "intercept": params["intercept"],
                    "validation_positive_rate": float(valid["y_true"].mean()),
                    "validation_raw_mean_probability": float(vp.mean()),
                    "validation_affine_mean_probability": float(vc.mean()),
                    "test_positive_rate": float(raw["y_true"].mean()),
                    "test_raw_mean_probability": float(tp.mean()),
                    "test_affine_mean_probability": float(tc.mean()),
                    "test_auc_change": float(
                        roc_auc_score(raw["y_true"], tc) - roc_auc_score(raw["y_true"], tp)
                    ),
                    "test_clipped_probabilities": int(((tp < eps) | (tp > 1 - eps)).sum()),
                    "validation_rows": len(vp),
                    "test_rows": len(tp),
                }
                values.append(row)
                records.append(row)
            summaries.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seeds": list(config["seeds"]),
                    "mean_over_seed_runs": {
                        key: float(np.mean([v[key] for v in values]))
                        for key in values[0]
                        if key not in {"dataset", "model", "seed"}
                    },
                    "range_over_seed_runs": {
                        key: [min(v[key] for v in values), max(v[key] for v in values)]
                        for key in ["slope", "intercept"]
                    },
                }
            )
    scope = (
        "Exploratory descriptive analysis after observing primary and calibration results. "
        "All 21 model/dataset combinations and 105 affine fits are retained. No new fitting, "
        "checkpoint selection or uncertainty interval is introduced. Mean predicted probability "
        "is a marginal diagnostic, not a sufficient calibration criterion. Antisymmetry of the "
        "raw NeuralAC score does not mathematically force a 0.5 mean on an arbitrary roster "
        "distribution. An affine intercept cannot identify a causal map-side advantage."
        " For a constant predictor, slope and intercept are not separately identifiable; "
        "only their resulting constant probability should be interpreted."
    )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "summaries": summaries,
        "fits": records,
        "artifact_sha256": artifacts,
    }
    target = root / "reports/calibration-diagnostics.json"
    target.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Calibration parameter and marginal-probability diagnostic",
        "",
        scope,
        "",
        "The table reports means across the five stored seed runs. Validation and test labels "
        "are unchanged across seeds. Repeated deterministic fits are not independent samples.",
        "",
        "Dataset | Model | Slope | Intercept | Validation prevalence | Raw validation mean | Test prevalence | Raw test mean | Affine test mean",
        "--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    keys = [
        "slope",
        "intercept",
        "validation_positive_rate",
        "validation_raw_mean_probability",
        "test_positive_rate",
        "test_raw_mean_probability",
        "test_affine_mean_probability",
    ]
    for row in summaries:
        means = row["mean_over_seed_runs"]
        lines.append(
            " | ".join([row["dataset"], row["model"], *[f"{means[key]:.6f}" for key in keys]])
        )
    lines += [
        "",
        "All fitted parameters, per-seed diagnostics and input hashes are in "
        "[the machine-readable report](calibration-diagnostics.json). These diagnostics "
        "supplement the paired NLL comparisons; they do not replace proper-score or "
        "conditional-calibration evaluation.",
    ]
    target.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"Described {len(records)} existing affine fits across {len(summaries)} combinations")


if __name__ == "__main__":
    main()
