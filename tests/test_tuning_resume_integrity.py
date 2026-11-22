# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Small real fits exercise public search locks and interrupted recovery."""

import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from gamematch.data.heroes import file_hash
from gamematch.models.tabular import hero_counts
from gamematch.runner import write_json
from gamematch.tuning import engine, search
from gamematch.tuning.data import Bundle, partition


class TuningResumeIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "run"
        rng = np.random.default_rng(42)
        parts = {}
        for name, n in [("train", 32), ("valid", 12), ("test", 12)]:
            x = rng.integers(1, 12, (n, 2, 5), dtype=np.int64)
            parts[name] = partition(
                [x],
                np.arange(n, dtype=np.float32) % 2,
                np.array([f"{name}-{i}" for i in range(n)]),
                {"group_id": np.arange(n) % 3},
                hero_counts(x, 12),
            )
        self.bundle = Bundle("uci-dota2", "binary", parts, {"vocab_size": 12}, {"fixture": True})
        self.configs = [
            {
                **cfg,
                "device": "cpu",
                "epochs": 1,
                "patience": 1,
                "hidden_dim": 8,
                "embedding_dim": 4,
                "batch_size": 16,
                "threads": 1,
            }
            for cfg in search.candidates("uci-dota2", "mlp")[:2]
        ]
        self.target = self.output / "uci-dota2/mlp"

    def tearDown(self):
        self.temp.cleanup()

    def run_search(self, model="mlp", configs=None):
        with (
            patch.object(search, "MATRIX", {"uci-dota2": [model]}),
            patch.object(search, "candidates", return_value=configs or self.configs),
            patch.object(search, "load_bundle", return_value=self.bundle),
            patch.object(search, "audit_primary_split", return_value={"fixture_reference": True}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            search.run(self.root, self.output)

    def snapshot(self):
        return {
            p: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in self.output.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }

    def restore(self, snapshot):
        for path, (data, mtime) in snapshot.items():
            if not path.exists() or path.read_bytes() != data or path.stat().st_mtime_ns != mtime:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                os.utime(path, ns=(mtime, mtime))

    def test_completed_resume_preserves_every_artifact_without_fit_or_inference(self):
        self.run_search()
        self.assertEqual(
            json.loads((self.output / "protocol.json").read_text())["schema_version"], 2
        )
        before = self.snapshot()
        with (
            patch.object(search, "fit", side_effect=AssertionError("refit")),
            patch.object(search, "evaluate", side_effect=AssertionError("reevaluate")),
            patch.object(search, "predict_raw", side_effect=AssertionError("forward")),
            patch.object(engine, "predict_raw", side_effect=AssertionError("forward")),
        ):
            self.run_search()
        self.assertEqual(before, self.snapshot())

    def test_interrupted_final_fit_is_reused_and_only_unfinished_seeds_train(self):
        with patch.object(search, "evaluate", side_effect=RuntimeError("interruption")):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                self.run_search()
        final = self.target / "final/seed-0"
        self.assertTrue((final / "metrics.json").exists())
        self.assertFalse((final / "result.json").exists())
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in final.iterdir()}
        with patch.object(search, "fit", wraps=engine.fit) as fits:
            self.run_search()
        self.assertEqual([call.args[6] for call in fits.call_args_list], [1, 2, 3, 4])
        self.assertTrue(
            all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())
        )
        self.assertEqual(
            json.loads((self.target / "status.json").read_text())["status"], "completed"
        )

    def test_search_metadata_cannot_mix_seeds_models_configuration_or_test_flags(self):
        self.run_search()
        before = self.snapshot()
        path = self.target / "search/trial-00/seed-100/metrics.json"
        original = json.loads(path.read_text())
        for key, value in [
            ("seed", 101),
            ("model", "fm"),
            ("test_evaluated", True),
            ("configuration", {**self.configs[0], "dropout": 0.9}),
        ]:
            with self.subTest(field=key):
                self.restore(before)
                write_json(path, {**original, key: value})
                with patch.object(search, "fit", side_effect=AssertionError("refit")):
                    with self.assertRaisesRegex(ValueError, "fit identity"):
                        self.run_search()

    def test_checkpoint_metadata_is_checked_even_after_its_hash_is_refreshed(self):
        directory = self.root / "fit"
        with contextlib.redirect_stdout(io.StringIO()):
            engine.fit(
                "binary",
                "mlp",
                self.bundle.specification,
                self.bundle.parts["train"],
                self.bundle.parts["valid"],
                self.configs[0],
                100,
                directory,
            )
        checkpoint_path = directory / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        record = json.loads((directory / "metrics.json").read_text())
        for key, value in [
            ("seed", 101),
            ("task", "regression"),
            ("model", "fm"),
            ("specification", {"vocab_size": 99}),
            ("configuration", {**self.configs[0], "dropout": 0.9}),
        ]:
            with self.subTest(field=key):
                torch.save({**checkpoint, key: value}, checkpoint_path)
                changed = copy.deepcopy(record)
                changed["artifact_sha256"]["checkpoint.pt"] = file_hash(checkpoint_path)
                write_json(directory / "metrics.json", changed)
                with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                    search.audit_fit(directory, self.bundle, "mlp", self.configs[0], 100)

    def test_completed_missing_fit_result_prediction_or_selection_is_rejected(self):
        self.run_search()
        before = self.snapshot()
        for relative in [
            "selection.json",
            "final/seed-0/metrics.json",
            "final/seed-0/result.json",
            "final/seed-0/predictions.npz",
            "final/seed-0/checkpoint.pt",
            "search/trial-00/seed-100/metrics.json",
        ]:
            with self.subTest(missing=relative):
                self.restore(before)
                (self.target / relative).unlink()
                with patch.object(search, "fit", side_effect=AssertionError("refit")):
                    with self.assertRaises(ValueError):
                        self.run_search()

    def test_final_identity_predictions_and_all_metrics_are_recomputed(self):
        self.run_search()
        before = self.snapshot()
        directory = self.target / "final/seed-0"
        result_path = directory / "result.json"
        original = json.loads(result_path.read_text())
        fit_record = json.loads((directory / "metrics.json").read_text())
        identity = {
            key: original[key]
            for key in [
                "dataset",
                "task",
                "specification",
                "world",
                "protocol_sha256",
                "data_manifest_sha256",
                "selection_sha256",
                "fit_metrics_sha256",
            ]
        }
        for key, value in [
            ("seed", 999),
            ("model", "fm"),
            ("test_evaluated", False),
            ("configuration", {}),
            ("metrics", {**original["metrics"], "log_loss": 123}),
        ]:
            with self.subTest(field=key):
                with self.assertRaises(ValueError):
                    search.audit_final(
                        directory,
                        self.bundle,
                        fit_record,
                        identity,
                        False,
                        {**original, key: value},
                    )
        for field in ["y_true", "row_id", "group_id", "p_positive"]:
            with self.subTest(prediction_field=field):
                self.restore(before)
                with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                    arrays = dict(saved)
                if field in {"row_id", "group_id"}:
                    arrays[field] = arrays[field][::-1]
                else:
                    arrays[field] = 1 - arrays[field]
                np.savez_compressed(directory / "predictions.npz", **arrays)
                changed = copy.deepcopy(original)
                changed["evaluation_artifact_sha256"]["predictions.npz"] = file_hash(
                    directory / "predictions.npz"
                )
                with self.assertRaises(ValueError):
                    search.audit_final(directory, self.bundle, fit_record, identity, False, changed)

    def test_data_identity_is_read_only_on_resume_and_detects_transform_or_split_changes(self):
        directory = self.root / "data"
        self.bundle.provenance["preprocessor"] = StandardScaler().fit(np.arange(12).reshape(6, 2))
        search.save_data_identity(directory, self.bundle, {})
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
        search.save_data_identity(directory, self.bundle, {})
        self.assertTrue(
            all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())
        )
        original = self.bundle.parts["train"].inputs[0].clone()
        self.bundle.parts["train"].inputs[0][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "Data manifest changed"):
            search.save_data_identity(directory, self.bundle, {})
        self.bundle.parts["train"].inputs[0].copy_(original)
        self.bundle.parts["test"].target[0] = 1 - self.bundle.parts["test"].target[0]
        with self.assertRaisesRegex(ValueError, "split identities"):
            search.save_data_identity(directory, self.bundle, {})
        self.assertTrue(
            all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())
        )

    def test_tabular_single_final_seed_uses_the_same_resume_guards(self):
        configs = search.candidates("uci-dota2", "logistic")[:2]
        self.run_search("logistic", configs)
        target = self.output / "uci-dota2/logistic"
        self.assertEqual(len(list((target / "final").glob("*/result.json"))), 1)
        before = self.snapshot()
        with patch.object(search, "fit", side_effect=AssertionError("refit")):
            self.run_search("logistic", configs)
        self.assertEqual(before, self.snapshot())

    def test_validation_labels_raw_scores_and_locked_seed_inventory_are_checked(self):
        self.run_search()
        before = self.snapshot()
        directory = self.target / "search/trial-00/seed-100"
        original = json.loads((directory / "metrics.json").read_text())
        for field in ["y_true", "raw"]:
            with self.subTest(field=field):
                self.restore(before)
                with np.load(directory / "validation.npz", allow_pickle=False) as saved:
                    arrays = dict(saved)
                arrays[field] = 1 - arrays[field] if field == "y_true" else arrays[field] + 2
                np.savez_compressed(directory / "validation.npz", **arrays)
                changed = copy.deepcopy(original)
                changed["artifact_sha256"]["validation.npz"] = file_hash(
                    directory / "validation.npz"
                )
                write_json(directory / "metrics.json", changed)
                with self.assertRaisesRegex(ValueError, "Validation"):
                    search.audit_fit(directory, self.bundle, "mlp", self.configs[0], 100)
        self.restore(before)
        unexpected = directory / "predictions.npz"
        unexpected.write_bytes(b"unexpected test artifact")
        with self.assertRaisesRegex(ValueError, "Search fit contains test artifacts"):
            self.run_search()
        unexpected.unlink()
        extra = self.target / "final/seed-99"
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, "seed inventory"):
            self.run_search()

    def test_regression_and_categorical_final_scores_can_resume_without_inference(self):
        rng = np.random.default_rng(13)
        for task in ["regression", "multiclass", "history"]:
            with self.subTest(task=task):
                parts = {}
                for split, count in [("train", 30), ("valid", 12), ("test", 12)]:
                    x = rng.normal(size=(count, 4)).astype(np.float32)
                    y = x[:, 0] * 0.2 if task == "regression" else np.arange(count) % 3
                    metadata = {"time": np.arange(count)}
                    if task == "regression":
                        metadata.update(current_mu=np.full(count, 25.0), y_true=25 + y * 2 + 0.1)
                    parts[split] = partition(
                        [x], y, np.array([f"{split}-{i}" for i in range(count)]), metadata, x
                    )
                spec = (
                    {"features": 4, "delta_scale": 2.0, "delta_mean": 0.1}
                    if task == "regression"
                    else {"features": 4, "classes": 3}
                )
                bundle = Bundle("fixture", task, parts, spec, {})
                name = "ridge" if task == "regression" else "logistic"
                cfg = (
                    {"threads": 1, "tabular": True, "alpha": 1.0}
                    if name == "ridge"
                    else {"threads": 1, "tabular": True, "C": 1.0}
                )
                directory = self.root / task / "final/seed-0"
                model, record = engine.fit(
                    task, name, spec, parts["train"], parts["valid"], cfg, 0, directory
                )
                audited = search.audit_fit(directory, bundle, name, cfg, 0)
                result = {
                    **record,
                    **engine.evaluate(model, bundle, cfg, directory),
                    "test_evaluated": True,
                    "reload_max_abs_error": 0.0,
                }
                result["evaluation_artifact_sha256"] = {
                    "predictions.npz": file_hash(directory / "predictions.npz")
                }
                with patch.object(engine, "predict_raw", side_effect=AssertionError("forward")):
                    search.audit_final(directory, bundle, audited, {}, False, result)


if __name__ == "__main__":
    unittest.main()
