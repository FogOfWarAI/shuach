# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Reporting must preserve experiment evidence and reject ambiguous summaries."""

import json
import tempfile
import unittest
from pathlib import Path

from gamematch.cli import main
from gamematch.reporting import create_run_report
from gamematch.runner import render_report


class ReportingTests(unittest.TestCase):
    def test_all_tracks_preserve_inputs_and_separate_decision_rows(self):
        cases = [
            ("final", {"datasets": ["heroes"]}, "heroes", "mlp"),
            ("pilot", {"datasets": ["heroes"]}, "heroes", "mlp"),
            ("temporal_predator_outcome", {"datasets": ["dbd"]}, "dbd", "mlp"),
            ("temporal_completed_player_history", {}, "overwatch", "history_deepsets"),
            ("temporal_future_observed_rating", {}, "dota2-future-rating", "gru"),
            (
                "temporal_delayed_feedback",
                {"models": [], "separate_domains": [False]},
                "overwatch",
                "calibrated_prior",
            ),
            ("synthetic_independent_decisions", {"mechanisms": ["additive"]}, "additive", "mlp"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            for index, (protocol, extra, dataset, model) in enumerate(cases):
                with self.subTest(protocol=protocol):
                    run = Path(temp) / str(index)
                    directory = run / f"{dataset}-seed-0" / model
                    directory.mkdir(parents=True)
                    config = {"protocol": protocol, "models": [model], "seeds": [0], **extra}
                    (run / "manifest.json").write_text(
                        json.dumps({"config": config, "status": "completed"})
                    )
                    metric = "mae" if protocol == "temporal_future_observed_rating" else "log_loss"
                    record = {"model": model, "seed": 0, "metrics": {metric: 0.4, "n": 10}}
                    if protocol in {"pilot", "final", "temporal_predator_outcome"}:
                        record["dataset"] = dataset
                    if protocol == "synthetic_independent_decisions":
                        record["metrics"] = {"evaluation": record["metrics"]}
                        rows = [
                            {
                                "mechanism": dataset,
                                "model": model,
                                "world_seed": 0,
                                "variant": variant,
                                "solver": solver,
                                "mean_true_imbalance": 0.2,
                                "mean_decision_regret": 0.1,
                                "mean_solver_gap": 0.0,
                            }
                            for variant in ["raw", "temperature2", "offset1"]
                            for solver in ["random", "sampled8", "exact"]
                        ]
                        (directory / "decision-metrics.json").write_text(json.dumps(rows))
                    (directory / "metrics.json").write_text(json.dumps(record))
                    (run / "results.json").write_text('["original results"]\n')
                    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
                    main(["report", str(run)])
                    self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))
                    summary = json.loads((run / "report/summary.json").read_text())
                    self.assertEqual(summary["model_jobs_present"], 1)
                    self.assertEqual(summary["missing_model_jobs"], [])
                    self.assertEqual(summary["summaries"][0]["metrics"][metric]["mean"], 0.4)
                    self.assertEqual(
                        len(summary["decision_summaries"]),
                        9 if protocol == "synthetic_independent_decisions" else 0,
                    )
                    if protocol == "temporal_delayed_feedback":
                        (run / "manifest.json").write_text(
                            json.dumps({"config": {"protocol": protocol}, "status": "running"})
                        )
                        create_run_report(run)
                        default = json.loads((run / "report/summary.json").read_text())
                        self.assertEqual(default["model_jobs_declared"], 7)
                        self.assertEqual(len(default["missing_model_jobs"]), 6)

    def test_incomplete_seeds_and_invalid_inputs_do_not_replace_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            config = {
                "protocol": "temporal_completed_player_history",
                "models": ["history_deepsets"],
                "seeds": [0, 1],
            }
            (run / "manifest.json").write_text(json.dumps({"config": config, "status": "running"}))
            sentinel = b'["irreplaceable aggregate"]\n'
            (run / "results.json").write_bytes(sentinel)
            directory = run / "seed-0-history_deepsets"
            directory.mkdir()
            record = {"model": "history_deepsets", "seed": 0, "metrics": {"log_loss": 0.3}}
            (directory / "metrics.json").write_text(json.dumps(record))
            create_run_report(run)
            saved = (run / "report/summary.json").read_bytes()
            summary = json.loads(saved)
            self.assertIsNone(summary["summaries"][0]["metrics"]["log_loss"]["mean"])
            self.assertEqual(summary["missing_model_jobs"], [["overwatch", "history_deepsets", 1]])
            self.assertIn("pending", (run / "report/report.md").read_text())
            with self.assertRaises(ValueError):
                render_report(run)
            for destination in [run, directory]:
                with self.assertRaises(ValueError):
                    create_run_report(run, destination)
            duplicate = run / "duplicated"
            duplicate.mkdir()
            (duplicate / "metrics.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                create_run_report(run)
            record["model"] = "undeclared-model"
            (duplicate / "metrics.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                create_run_report(run)
            self.assertEqual((run / "results.json").read_bytes(), sentinel)
            self.assertEqual((run / "report/summary.json").read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
