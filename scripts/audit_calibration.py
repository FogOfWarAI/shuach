# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Independently replay saved binary calibration and verify its validation scope."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as handle:
        return dict(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/hero-calibration-v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/calibration-audit.json"))
    args = parser.parse_args()
    manifest = read(args.run / "manifest.json")
    if manifest["status"] != "completed":
        raise ValueError("Calibration run must be complete")
    config = manifest["config"]
    primary = Path(config["primary_run"])
    if digest(primary / "manifest.json") != manifest["identity"]["primary_manifest"]:
        raise ValueError("Primary run identity changed")
    records = []
    clip = config["probability_clip_for_logit"]
    for dataset in config["datasets"]:
        split_path = primary / dataset / "split-manifest.npz"
        split = arrays(split_path)
        for model in config["models"]:
            for seed in config["seeds"]:
                source = primary / dataset / f"seed-{seed}-{model}"
                raw_path = source / "predictions.npz"
                raw = arrays(raw_path)
                reference_validation = None
                for mode in config["variants"]:
                    folder = args.run / dataset / f"seed-{seed}-{model}-{mode}"
                    result = read(folder / "metrics.json")
                    if (result["dataset"], result["model"], result["seed"], result["variant"]) != (
                        dataset,
                        model,
                        seed,
                        mode,
                    ):
                        raise ValueError("Misidentified calibration artifact")
                    if digest(raw_path) != result["primary_predictions_sha256"]:
                        raise ValueError("Raw primary predictions changed")
                    if digest(split_path) != result["validation_split_sha256"]:
                        raise ValueError("Original split artifact changed")
                    actual_checkpoints = {p.name: digest(p) for p in source.glob("checkpoint.*")}
                    if actual_checkpoints != result["primary_checkpoint_sha256"]:
                        raise ValueError("Primary checkpoint changed")
                    calibrated = arrays(folder / "predictions.npz")
                    if calibrated.keys() != raw.keys():
                        raise ValueError("Calibration changed the prediction schema")
                    for key in raw:
                        if key != "p_positive":
                            np.testing.assert_array_equal(calibrated[key], raw[key])
                    valid = arrays(folder / "validation.npz")
                    np.testing.assert_array_equal(valid["row_id"], split["valid_row_ids"])
                    if reference_validation is not None:
                        for key in valid:
                            np.testing.assert_array_equal(valid[key], reference_validation[key])
                    reference_validation = valid
                    p, y = valid["p_positive"].astype(float), valid["y_true"]
                    if (
                        p.shape != y.shape
                        or p.ndim != 1
                        or not np.isfinite(p).all()
                        or ((p < 0) | (p > 1)).any()
                        or not np.isin(y, [0, 1]).all()
                    ):
                        raise ValueError("Invalid validation probabilities or labels")
                    p = np.clip(p, clip, 1 - clip)
                    z = np.log(p) - np.log1p(-p)
                    params = result["parameters"]
                    slope, intercept = params["slope"], params["intercept"]
                    if params["mode"] != mode or params["validation_rows"] != len(y):
                        raise ValueError("Calibration fit metadata disagrees")
                    if not 0.001 <= slope <= 1000 or not -20 <= intercept <= 20:
                        raise ValueError("Calibration parameters outside declared bounds")
                    if mode == "temperature" and intercept != 0:
                        raise ValueError("Temperature calibration acquired an intercept")
                    after = slope * z + intercept
                    before_nll = float(np.mean(np.logaddexp(0, z) - y * z))
                    after_nll = float(np.mean(np.logaddexp(0, after) - y * after))
                    np.testing.assert_allclose(
                        [before_nll, after_nll],
                        [params["validation_nll_before"], params["validation_nll_after"]],
                        rtol=0,
                        atol=1e-10,
                    )
                    if after_nll > before_nll + 1e-9:
                        raise ValueError("Fitted validation NLL increased")
                    residual = expit(after) - y
                    gradients = [float(np.mean(residual * z))]
                    values, bounds = [slope], [(0.001, 1000)]
                    if mode == "affine":
                        gradients.append(float(residual.mean()))
                        values.append(intercept)
                        bounds.append((-20, 20))
                    projected = [
                        abs(v - np.clip(v - g, low, high))
                        for v, g, (low, high) in zip(values, gradients, bounds)
                    ]
                    if max(projected) > 1e-4:
                        raise ValueError("Calibration violates approximate bounded optimality")
                    test = np.clip(raw["p_positive"].astype(float), clip, 1 - clip)
                    expected = expit(slope * (np.log(test) - np.log1p(-test)) + intercept)
                    error = float(np.max(np.abs(expected - calibrated["p_positive"])))
                    if not np.isfinite(error) or error > 1e-12:
                        raise ValueError("Saved calibration does not reproduce test predictions")
                    records.append(
                        {
                            "artifact": str(folder),
                            "max_probability_error": error,
                            "validation_nll_before": before_nll,
                            "validation_nll_after": after_nll,
                            "max_projected_gradient": max(projected),
                            "prediction_sha256": digest(folder / "predictions.npz"),
                        }
                    )
    expected_fits = np.prod([len(config[k]) for k in ["datasets", "models", "seeds", "variants"]])
    if len(records) != expected_fits or len(records) != manifest["completed_postprocessing_fits"]:
        raise ValueError("Incomplete calibration fit count")
    payload = {
        "status": "passed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent saved-parameter replay, raw checkpoint/prediction hashes, unchanged test metadata, original validation membership, shared validation arrays across variants, validation objective and approximate bounded first-order optimality. Validation predictions are not independently refitted; source timestamp authenticity and causal outcomes are outside this audit.",
        "fits": len(records),
        "max_probability_error": max(r["max_probability_error"] for r in records),
        "max_projected_gradient": max(r["max_projected_gradient"] for r in records),
        "checks": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("Passed", len(records), "validation calibration fits")


if __name__ == "__main__":
    main()
