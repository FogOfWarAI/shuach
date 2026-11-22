# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Independently reload completed tuned predictors on sampled rows using CPU."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from audit_tuning_integrity import source_identity
from scipy.special import expit, softmax

from gamematch.data.heroes import file_hash
from gamematch.runner import code_hash
from gamematch.tuning.data import load_bundle, partition
from gamematch.tuning.engine import model_factory, predict_raw, reload_model

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(2)
    groups = {}
    for run_name in ["hpo-v1", "hpo-v1-baselines", "specialist-hpo-v2"]:
        run = ROOT / "runs" / run_name
        if not (run / "protocol.json").exists():
            continue
        specialist = run_name == "specialist-hpo-v2"
        source_identity(run, specialist=specialist)
        pattern = "*/final/seed-*/result.json" if specialist else "*/*/final/seed-*/result.json"
        for path in sorted(run.glob(pattern)):
            result = read(path)
            name = result["dataset"]
            world = result["seed"] if name.startswith("synthetic-") else 0
            groups.setdefault((name, world), []).append((path, result))
    checks = []
    for (name, world), records in sorted(groups.items()):
        bundle = load_bundle(ROOT, name, world)
        samples = {}
        for split in ["valid", "test"]:
            data = bundle.parts[split]
            indices = np.sort(
                np.random.default_rng(20261114).choice(
                    len(data.target), min(64, len(data.target)), replace=False
                )
            )
            part = partition(
                [x[indices].numpy() for x in data.inputs],
                data.target[indices],
                data.row_id[indices],
                {k: v[indices] for k, v in data.metadata.items()},
                None if data.tabular is None else data.tabular[indices],
            )
            samples[split] = (indices, part)
        for path, result in records:
            cfg = {**result["configuration"], "device": "cpu", "batch_size": 37}
            if cfg["tabular"]:
                model = reload_model(
                    path.parent,
                    bundle.task,
                    result["model"],
                    bundle.specification,
                    result["configuration"],
                )
            else:
                saved = torch.load(
                    path.with_name("checkpoint.pt"), map_location="cpu", weights_only=True
                )
                for key, expected in {
                    "configuration": result["configuration"],
                    "task": bundle.task,
                    "model": result["model"],
                    "specification": bundle.specification,
                    "seed": result["seed"],
                }.items():
                    if saved[key] != expected:
                        raise ValueError(f"Checkpoint identity mismatch: {path}: {key}")
                model = model_factory(
                    bundle.task, result["model"], bundle.specification, result["configuration"]
                )
                model.load_state_dict(saved["state_dict"])
                model.eval()
            for split, (indices, part) in samples.items():
                saved_path = path.with_name(
                    "validation.npz" if split == "valid" else "predictions.npz"
                )
                with np.load(saved_path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["row_id"][indices], part.row_id)
                    raw = predict_raw(model, part, bundle.task, cfg)
                    raw_error = float(np.max(np.abs(raw - saved["raw"][indices])))
                    if not np.isfinite(raw).all() or raw_error > 5e-5:
                        raise ValueError(
                            f"CPU raw prediction mismatch: {path}: {split}: {raw_error}"
                        )
                    output_error = None
                    if split == "test":
                        if bundle.task == "regression":
                            output = (
                                part.metadata["current_mu"]
                                + raw * bundle.specification["delta_scale"]
                                + bundle.specification["delta_mean"]
                            )
                            expected = saved["prediction"][indices]
                        elif bundle.task == "binary":
                            output, expected = expit(raw), saved["p_positive"][indices]
                        else:
                            output = softmax(raw / result["temperature"], axis=1)
                            expected = saved["probabilities"][indices]
                        output_error = float(np.max(np.abs(output - expected)))
                        if not np.isfinite(output).all() or output_error > 5e-5:
                            raise ValueError(f"CPU output mismatch: {path}: {output_error}")
                checks.append(
                    {
                        "result": str(path.relative_to(ROOT)),
                        "split": split,
                        "rows": len(indices),
                        "raw_max_absolute_error": raw_error,
                        "output_max_absolute_error": output_error,
                        "checkpoint_sha256": file_hash(
                            path.with_name(
                                "checkpoint.joblib" if cfg["tabular"] else "checkpoint.pt"
                            )
                        ),
                    }
                )
            del model
        del bundle, samples
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "Completed final fits present at invocation only. Current-library CPU reload, two sampled partitions per fit, up to 64 rows each and batch size 37; no refitting or selection. This does not claim cross-dependency-version artifact portability.",
        "source_sha256": code_hash(),
        "final_fits_checked": len(checks) // 2,
        "absolute_tolerance": 5e-5,
        "max_raw_error": max((c["raw_max_absolute_error"] for c in checks), default=0),
        "max_output_error": max((c["output_max_absolute_error"] or 0 for c in checks), default=0),
        "checks": checks,
    }
    (ROOT / "reports/tuned-predictor-reload-audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}))


if __name__ == "__main__":
    main()
