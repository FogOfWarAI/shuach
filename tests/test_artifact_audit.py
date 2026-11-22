# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Corrupt result files must fail the independent evidence audit."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "artifact_auditor", Path(__file__).resolve().parents[1] / "scripts/audit_results.py"
)
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


class ArtifactAuditTests(unittest.TestCase):
    def test_corrupt_predictions_and_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            snapshot = run / "source-snapshot/gamematch"
            snapshot.mkdir(parents=True)
            (run / "manifest.json").write_text(
                json.dumps({"status": "completed", "code_sha256": AUDITOR.source_hash(snapshot)})
            )
            job = run / "seed-0"
            job.mkdir()
            metrics = {"log_loss": float(-np.log(0.8)), "accuracy": 1.0, "brier": 0.04}
            base = {
                "row_id": np.array([1, 2]),
                "y_true": np.array([0, 1]),
                "p_positive": np.array([0.2, 0.8]),
            }
            (job / "metrics.json").write_text(json.dumps({"metrics": metrics}))
            np.savez(job / "predictions.npz", **base)
            self.assertEqual(AUDITOR.audit(run)["audited_model_jobs"], 1)
            for changed in [
                {"p_positive": np.array([np.nan, 0.8])},
                {"p_positive": np.array([-0.2, 1.2])},
                {"p_positive": np.array([[0.2], [0.8]])},
                {"y_true": np.array([0, 0.5])},
                {"y_true": np.array([0, np.nan])},
                {"row_id": np.array([1])},
            ]:
                with self.subTest(changed=changed):
                    np.savez(job / "predictions.npz", **(base | changed))
                    with self.assertRaises(ValueError):
                        AUDITOR.audit(run)
            np.savez(job / "predictions.npz", **base)
            (job / "metrics.json").write_text(
                json.dumps({"metrics": metrics | {"log_loss": float("nan")}})
            )
            with self.assertRaises(ValueError):
                AUDITOR.audit(run)
