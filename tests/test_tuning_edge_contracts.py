# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Small boundary fixtures for tuning targets, calibration and artifact reuse."""

import copy
import hashlib
import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from gamematch.tuning.data import Bundle, load_bundle, partition
from gamematch.tuning.engine import (
    evaluate,
    fit,
    model_factory,
    predict_raw,
    primary_loss,
    reload_model,
)
from gamematch.tuning.search import candidates, decision_evaluation, run


class TuningEdgeContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        torch.set_num_threads(1)

    def config(self, name="mlp"):
        return {
            **candidates("dbd-2022", name)[0],
            "device": "cpu",
            "threads": 1,
            "epochs": 2,
            "batch_size": 8,
            "hidden_dim": 8,
        }

    def test_invalid_targets_fail_before_creating_training_artifacts(self):
        cases = [
            ("binary", {"vocab_size": 10}, np.array([0.0, 0.5, 1.0])),
            ("binary", {"vocab_size": 10}, np.array([0.0, np.nan, 1.0])),
            ("multiclass", {"features": 2, "classes": 3}, np.array([0.0, 1.5, 2.0])),
            ("multiclass", {"features": 2, "classes": 3}, np.array([0, 3, 2])),
            ("history", {"classes": 3}, np.array([0, -1, 2])),
            ("regression", {"delta_scale": 2.0}, np.array([0.0, np.inf, 2.0])),
            ("regression", {"delta_scale": 2.0}, np.array([[0.0], [1.0], [2.0]])),
        ]
        for index, (task, spec, bad) in enumerate(cases):
            for split in ("train", "valid"):
                with self.subTest(task=task, case=index, split=split):
                    good = np.array([0, 1, 0]) if task != "regression" else np.zeros(3)
                    parts = {
                        name: partition(
                            [np.ones((3, 2), dtype=np.float32)],
                            bad if name == split else good,
                            np.arange(3),
                            {},
                        )
                        for name in ("train", "valid")
                    }
                    output = self.root / f"invalid-{index}-{split}"
                    with (
                        patch(
                            "gamematch.tuning.engine.model_factory",
                            side_effect=AssertionError("Model construction must not be reached"),
                        ),
                        self.assertRaises(ValueError),
                    ):
                        fit(
                            task,
                            "mlp",
                            spec,
                            parts["train"],
                            parts["valid"],
                            self.config(),
                            4,
                            output,
                        )
                    self.assertFalse(output.exists())

    def test_misaligned_target_rows_fail_before_training(self):
        good = partition([np.ones((3, 2), dtype=np.float32)], np.array([0, 1, 2]), np.arange(3), {})
        bad = partition([np.ones((2, 2), dtype=np.float32)], np.array([0, 1, 2]), np.arange(3), {})
        output = self.root / "unaligned"
        with self.assertRaisesRegex(ValueError, "align"):
            fit(
                "multiclass",
                "mlp",
                {"features": 2, "classes": 3},
                bad,
                good,
                self.config(),
                4,
                output,
            )
        self.assertFalse(output.exists())

    def test_history_tuning_rejects_a_silently_changed_output_space(self):
        spec = {
            "history_features": 5,
            "static_features": 13,
            "team_size": 2,
            "history_length": 3,
            "classes": 2,
        }
        cfg = {"width": 8, "dropout": 0.1}
        with self.assertRaisesRegex(ValueError, "three-class"):
            model_factory("history", "history_deepsets", spec, cfg)
        model = model_factory("history", "history_deepsets", {**spec, "classes": 3}, cfg)
        with torch.no_grad():
            logits = model(
                torch.zeros(4, 2, 2, 3, 5),
                torch.zeros(4, 2, 2, 13),
                torch.zeros(4, 2, 2, 3, dtype=torch.bool),
            )
        self.assertEqual(tuple(logits.shape), (4, 3))

    def test_primary_losses_reject_broadcasting_and_fractional_classes(self):
        with self.assertRaisesRegex(ValueError, "integer labels"):
            primary_loss("history", np.array([0.0, 1.5]), np.zeros((2, 3)), {})
        for task, spec in [("binary", {}), ("regression", {"delta_scale": 2.0})]:
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, "shape"):
                primary_loss(task, np.array([0, 1]), np.zeros((2, 1)), spec)
        self.assertAlmostEqual(primary_loss("binary", np.array([0, 1]), np.zeros(2), {}), np.log(2))
        self.assertAlmostEqual(
            primary_loss("history", np.array([0, 2]), np.zeros((2, 3)), {}), np.log(3)
        )
        self.assertEqual(
            primary_loss(
                "regression", np.array([0.0, 1.0]), np.array([1.0, 3.0]), {"delta_scale": 2.0}
            ),
            3.0,
        )

    def test_ordinal_checkpoint_preserves_probabilities_at_a_new_batch_size(self):
        rng = np.random.default_rng(31)
        train = partition(
            [rng.normal(size=(20, 2)).astype(np.float32)], np.arange(20) % 5, np.arange(20), {}
        )
        valid = partition(
            [rng.normal(size=(10, 2)).astype(np.float32)],
            np.arange(10) % 5,
            np.arange(100, 110),
            {},
        )
        cfg, spec = self.config("ordinal"), {"features": 2, "classes": 5}
        model, result = fit("multiclass", "ordinal", spec, train, valid, cfg, 13, self.root)
        restored = reload_model(self.root, "multiclass", "ordinal", spec, cfg)
        before = predict_raw(model, valid, "multiclass", cfg)
        after = predict_raw(restored, valid, "multiclass", {**cfg, "batch_size": 3})
        np.testing.assert_allclose(np.exp(before), np.exp(after), atol=1e-7, rtol=1e-6)
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(
            set(result["artifact_sha256"]), {"checkpoint.pt", "validation.npz", "history.json"}
        )
        for name, digest in result["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((self.root / name).read_bytes()).hexdigest(), digest)
        self.assertFalse((self.root / "predictions.npz").exists())
        with np.load(self.root / "validation.npz") as saved:
            np.testing.assert_array_equal(saved["row_id"], valid.row_id)
            np.testing.assert_array_equal(saved["y_true"], valid.target)
        with self.assertRaisesRegex(ValueError, "identity"):
            reload_model(self.root, "multiclass", "ordinal", {**spec, "classes": 4}, cfg)

    def regression_bundle(self):
        spec = {"delta_mean": 3.0, "delta_scale": 2.0}

        def part(x, start, current=None):
            x = np.asarray(x, dtype=np.float64)[:, None]
            target = 2 * x[:, 0] + 0.5
            metadata = (
                {}
                if current is None
                else {
                    "current_mu": np.asarray(current),
                    "y_true": np.asarray(current) + 3 + 2 * target,
                }
            )
            return partition([], target, np.arange(start, start + len(x)), metadata, x)

        return Bundle(
            "future-rating",
            "regression",
            {
                "train": part([-2, -1, 0, 1, 2], 0),
                "valid": part([0.25, 0.75], 100),
                "test": part([-0.5, 1.5], 200, [50.0, 80.0]),
            },
            spec,
            {"fixture": "known normalized linear residual"},
        )

    def test_regression_reloads_and_restores_the_original_rating_units(self):
        bundle = self.regression_bundle()
        cfg = {"tabular": True, "threads": 1, "alpha": 0.0}
        _, result = fit(
            "regression",
            "ridge",
            bundle.specification,
            bundle.parts["train"],
            bundle.parts["valid"],
            cfg,
            7,
            self.root,
        )
        restored = reload_model(self.root, "regression", "ridge", bundle.specification, cfg)
        self.assertEqual(result["selection_metric"], "mae")
        self.assertEqual(set(result["artifact_sha256"]), {"checkpoint.joblib", "validation.npz"})
        evaluated = evaluate(restored, bundle, cfg, self.root)
        self.assertIsNone(evaluated["temperature"])
        with np.load(self.root / "predictions.npz") as saved:
            np.testing.assert_allclose(saved["prediction"], [52.0, 90.0], atol=1e-10)
            np.testing.assert_array_equal(saved["y_true"], bundle.parts["test"].metadata["y_true"])

    def test_neural_regression_losses_reload_and_ignore_missing_snapshots(self):
        rng = np.random.default_rng(119)

        def part(n, offset):
            snapshots = rng.normal(size=(n, 12, 2)).astype(np.float32)
            mask = np.ones((n, 12), dtype=bool)
            mask[:, -3:] = False
            lineups = np.tile(np.arange(1, 11).reshape(1, 2, 5), (n, 1, 1))
            return partition(
                [snapshots, lineups, mask],
                rng.normal(size=n).astype(np.float32),
                np.arange(offset, offset + n),
                {},
            )

        train, valid = part(12, 0), part(8, 100)
        spec = {"features": 2, "hero_vocab": 11, "delta_scale": 2.5, "delta_mean": 0.3}
        for loss in ["mse", "mae", "huber"]:
            with self.subTest(loss=loss):
                cfg = {
                    **candidates("future-rating", "gru")[0],
                    "device": "cpu",
                    "threads": 1,
                    "width": 8,
                    "layers": 1,
                    "batch_size": 4,
                    "epochs": 2,
                    "loss": loss,
                }
                folder = self.root / loss
                model, result = fit("regression", "gru", spec, train, valid, cfg, 23, folder)
                restored = reload_model(folder, "regression", "gru", spec, cfg)
                before = predict_raw(model, valid, "regression", cfg)
                changed = copy.deepcopy(valid)
                changed.inputs[0][~changed.inputs[2]] = 99999
                after = predict_raw(restored, changed, "regression", {**cfg, "batch_size": 3})
                np.testing.assert_allclose(before, after, atol=1e-7, rtol=1e-6)
                self.assertEqual(result["selection_metric"], "mae")
                self.assertTrue(np.isfinite(before).all())

    def test_temperature_and_predictions_do_not_depend_on_test_labels(self):
        validation = partition(
            [
                np.array(
                    [[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 2, 0], [0, 1, 2], [2, 0, 1]],
                    dtype=np.float32,
                )
            ],
            np.array([0, 1, 2, 0, 1, 2]),
            np.arange(6),
            {},
        )
        test = partition(
            [np.array([[2, 0, 0], [0, 1, 0], [0, 0, 3]], dtype=np.float32)],
            np.array([0, 1, 2]),
            np.arange(10, 13),
            {},
        )
        bundle = Bundle(
            "fixture", "multiclass", {"valid": validation, "test": test}, {"classes": 3}, {}
        )
        cfg = {"tabular": False, "batch_size": 2, "device": "cpu"}
        results, probabilities = [], []
        for i, labels in enumerate([np.array([0, 1, 2]), np.array([2, 0, 1])]):
            bundle.parts["test"].target = labels
            output = self.root / str(i)
            output.mkdir()
            results.append(evaluate(nn.Identity(), bundle, cfg, output))
            with np.load(output / "predictions.npz") as saved:
                probabilities.append(saved["probabilities"].copy())
        self.assertEqual(results[0]["temperature"], results[1]["temperature"])
        np.testing.assert_array_equal(*probabilities)
        self.assertNotEqual(results[0]["metrics"]["log_loss"], results[1]["metrics"]["log_loss"])

    def test_future_rating_transforms_ignore_heldout_and_masked_values(self):
        snapshots = np.arange(9 * 12 * 6, dtype=np.float32).reshape(9, 12, 6)
        valid = np.ones((9, 12), dtype=bool)
        valid[0, -3:] = False
        data = SimpleNamespace(
            snapshots=snapshots,
            valid=valid,
            current_mu=np.full(9, 25.0),
            target_mu=25.0 + np.arange(9),
            lineups=np.tile(np.arange(1, 11).reshape(1, 2, 5), (9, 1, 1)),
            row_id=np.arange(9),
            player_id=np.arange(9),
            match_id=np.arange(9),
            time=np.arange(9),
            manifest={"fixture": "train transform isolation"},
        )
        masks = {name: np.arange(9) // 3 == i for i, name in enumerate(["train", "valid", "test"])}

        def bundle(source):
            with (
                patch("gamematch.tuning.data.load_skill", return_value=(copy.deepcopy(source), {})),
                patch("gamematch.tuning.data.skill_splits", return_value=(masks, [3.0, 6.0])),
            ):
                return load_bundle(self.root, "future-rating")

        before = bundle(data)
        changed = copy.deepcopy(data)
        changed.snapshots[3:] += 10000
        changed.snapshots[~valid] = -99999
        changed.target_mu[3:] += 1000
        changed.lineups[3:] += 100
        after = bundle(changed)
        self.assertEqual(before.specification, after.specification)
        self.assertEqual(before.provenance, after.provenance)
        np.testing.assert_array_equal(before.parts["train"].target, after.parts["train"].target)
        for first, second in zip(before.parts["train"].inputs, after.parts["train"].inputs):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(after.parts["test"].inputs[1].any())
        self.assertTrue(
            np.all(
                after.parts["test"].metadata["y_true"] - before.parts["test"].metadata["y_true"]
                == 1000
            )
        )

    def test_assignment_regret_uses_hidden_outcomes_not_the_predictor_objective(self):
        class SumDifference(nn.Module):
            def forward(self, roster):
                sums = roster.float().sum(-1)
                return (sums[:, 0] - sums[:, 1]) / 10

        template = np.array(
            [
                [side, tuple(i for i in range(10) if i not in side)]
                for side in combinations(range(10), 5)
            ]
        )
        truth = np.full((1, 252), 0.9)
        truth[0, -1] = 0.5  # The hidden optimum has left sum 40, unlike the fitted scorer.
        environment = self.root / "oracle.npz"
        np.savez(
            environment,
            lobbies=np.arange(1, 11)[None],
            candidate_template=template,
            true_probability=truth,
            oracle_cost=np.zeros(1),
        )
        bundle = Bundle(
            "synthetic-fixture",
            "binary",
            {},
            {"vocabulary": list(range(1, 11))},
            {"decision_environment": str(environment)},
        )
        result = decision_evaluation(
            SumDifference(), bundle, {"device": "cpu", "batch_size": 37}, self.root
        )
        self.assertAlmostEqual(result["mean_decision_regret"], 0.4)
        with np.load(self.root / "decisions.npz") as saved:
            selected = int(saved["selected"][0])
            self.assertIn(int((template[selected, 0] + 1).sum()), [27, 28])
            self.assertNotEqual(selected, 251)

    def test_search_resumes_a_completed_fit_without_retraining_or_changing_selection(self):
        bundle = self.regression_bundle()
        reference = self.root / "runs/skill-v1/seed-0-gru"
        reference.mkdir(parents=True)
        np.savez(
            reference / "predictions.npz",
            row_id=bundle.parts["test"].row_id,
            y_true=bundle.parts["test"].metadata["y_true"],
        )
        cfg = {"tabular": True, "threads": 1, "alpha": 0.0}
        output = self.root / "search"
        with (
            patch("gamematch.tuning.search.MATRIX", {"future-rating": ["ridge"]}),
            patch("gamematch.tuning.search.candidates", return_value=[cfg]),
            patch("gamematch.tuning.search.load_bundle", return_value=bundle),
        ):
            run(self.root, output)
            folder = output / "future-rating/ridge"
            selection = (folder / "selection.json").read_bytes()
            result = folder / "final/seed-0/result.json"
            original = json.loads(result.read_text())
            result.unlink()  # Simulate interruption after checkpoint/validation persisted.
            failed_status = {"status": "failed", "dataset": "future-rating", "model": "ridge"}
            (folder / "status.json").write_text(json.dumps(failed_status))
            with patch(
                "gamematch.tuning.search.fit",
                side_effect=AssertionError("Completed fit must be reused"),
            ):
                run(self.root, output)
            self.assertEqual((folder / "selection.json").read_bytes(), selection)
            restored = json.loads(result.read_text())
            self.assertEqual(restored["metrics"], original["metrics"])
            self.assertEqual(restored["selection_sha256"], original["selection_sha256"])
            # A changed locked choice must fail before any new final evaluation.
            changed = json.loads(selection)
            changed["validation_mean"] += 1.0
            (folder / "selection.json").write_text(json.dumps(changed))
            (folder / "status.json").write_text(json.dumps(failed_status))
            with (
                patch(
                    "gamematch.tuning.search.evaluate",
                    side_effect=AssertionError("Test access must not occur"),
                ),
                self.assertRaisesRegex(ValueError, "Selection changed"),
            ):
                run(self.root, output)


if __name__ == "__main__":
    unittest.main()
