# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Tiny binary-history fits exercise locked artifacts and independent audits."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.special import softmax

from gamematch.data.heroes import file_hash
from gamematch.data.indexed_history import IndexedHistoryStore
from gamematch.evaluation.binary import binary_metrics

SPEC = importlib.util.spec_from_file_location(
    "league_neural_driver", Path(__file__).resolve().parents[1] / "scripts/run_league_neural.py"
)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class LeagueNeuralTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        rng = np.random.default_rng(120)
        n = 20
        performance = rng.normal(size=(9, 24)).astype(np.float32)
        performance[0] = 0
        arrays = {
            "performance": performance,
            "history-index": rng.integers(0, 9, (n, 2, 5, 10), dtype=np.int32),
            "appearance-available": np.zeros(9),
            "match-start": np.arange(n, dtype=float) + 10,
            "static": rng.normal(size=(n, 2, 5, 6)).astype(np.float32),
            "target": np.arange(n, dtype=np.int64) % 2,
            "row-id": np.array([f"fixture-{index}" for index in range(n)]),
            "train-rows": np.arange(12),
            "valid-rows": np.arange(12, 16),
            "test-rows": np.arange(16, 20),
        }
        for name, values in arrays.items():
            np.save(self.data / (name + ".npy"), values)
        DRIVER.write_json(
            self.data / "manifest.json",
            {
                "team_size": 5,
                "history_length": 10,
                "files_sha256": {p.name: file_hash(p) for p in self.data.glob("*.npy")},
                "normalizer": {
                    "history_mean": [0] * 25,
                    "history_scale": [1] * 25,
                    "static_mean": [0] * 6,
                    "static_scale": [1] * 6,
                },
            },
        )
        self.store = IndexedHistoryStore(self.data)
        self.rows = {part: arrays[part + "-rows"] for part in ["train", "valid", "test"]}
        self.config = {
            **DRIVER.CONFIG,
            "width": 8,
            "batch_size": 8,
            "epochs": 2,
            "patience": 2,
            "threads": 1,
        }
        self.protocol = "a" * 64

    def tearDown(self):
        self.temporary.cleanup()

    def fit(self, name="history_deepsets", columns=None, tag="fit"):
        columns = columns or [0, 1, 24]
        directory = self.root / tag
        with contextlib.redirect_stdout(io.StringIO()):
            DRIVER.fit(
                name,
                self.store,
                self.rows["train"],
                self.rows["valid"],
                columns,
                self.config,
                11,
                directory,
                self.protocol,
            )
        model, selection = DRIVER.reload_selected(
            directory, name, columns, self.config, 11, self.protocol
        )
        return directory, model, selection

    def test_binary_models_use_only_train_valid_and_reload(self):
        for name in ["history_deepsets", "owo"]:
            for columns in [[0, 1, 24], list(range(25))]:
                with self.subTest(model=name, features=len(columns)):
                    with patch.object(self.store, "batch", wraps=self.store.batch) as batches:
                        directory, model, selection = self.fit(
                            name, columns, f"{name}-{len(columns)}"
                        )
                        self.assertTrue(
                            all(
                                not np.isin(call.args[0], self.rows["test"]).any()
                                for call in batches.call_args_list
                            )
                        )
                    self.assertFalse(selection["test_evaluated"])
                    with np.load(directory / "validation.npz", allow_pickle=False) as saved:
                        self.assertEqual(saved["raw"].shape, (4, 2))
                    error = DRIVER.audit_validation(
                        directory,
                        model,
                        selection,
                        self.store,
                        self.rows["valid"],
                        columns,
                        self.config,
                    )
                    self.assertLessEqual(error, DRIVER.RELOAD_TOLERANCE)
                    self.assertFalse((directory / "predictions.npz").exists())

    def test_invalid_fit_inputs_and_locked_selection_are_not_overwritten(self):
        directory = self.root / "invalid"
        for train, valid, config in [
            (self.rows["train"], self.rows["valid"], {**self.config, "epochs": 0}),
            (self.rows["train"], np.array([0, 12]), self.config),
            (np.array([-1, 0]), self.rows["valid"], self.config),
        ]:
            with self.assertRaises(ValueError):
                DRIVER.fit(
                    "history_deepsets",
                    self.store,
                    train,
                    valid,
                    [0, 1, 24],
                    config,
                    11,
                    directory,
                    self.protocol,
                )
            self.assertFalse(directory.exists())
        directory, _, _ = self.fit()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
        with self.assertRaisesRegex(ValueError, "reloaded"):
            DRIVER.fit(
                "history_deepsets",
                self.store,
                self.rows["train"],
                self.rows["valid"],
                [0, 1, 24],
                self.config,
                11,
                directory,
                self.protocol,
            )
        self.assertTrue(
            all((p.read_bytes(), p.stat().st_mtime_ns) == state for p, state in before.items())
        )

    def test_validation_identity_score_and_calibration_are_recomputed(self):
        directory, model, selection = self.fit()
        original = {p.name: p.read_bytes() for p in directory.iterdir()}

        def restore():
            for name, values in original.items():
                (directory / name).write_bytes(values)

        with np.load(directory / "validation.npz", allow_pickle=False) as saved:
            arrays = dict(saved)
        np.savez_compressed(
            directory / "validation.npz", **{**arrays, "y_true": 1 - arrays["y_true"]}
        )
        changed = {
            **selection,
            "artifact_sha256": {
                **selection["artifact_sha256"],
                "validation.npz": file_hash(directory / "validation.npz"),
            },
        }
        DRIVER.write_json(directory / "selection.json", changed)
        model, changed = DRIVER.reload_selected(
            directory, "history_deepsets", [0, 1, 24], self.config, 11, self.protocol
        )
        with self.assertRaisesRegex(ValueError, "labels"):
            DRIVER.audit_validation(
                directory, model, changed, self.store, self.rows["valid"], [0, 1, 24], self.config
            )
        restore()
        for key, value, message in [
            ("validation_log_loss", selection["validation_log_loss"] + 0.1, "minimize"),
            ("temperature", selection["temperature"] + 1.0, "temperature"),
        ]:
            changed = {
                **selection,
                key: value,
                "artifact_sha256": dict(selection["artifact_sha256"]),
            }
            checkpoint = torch.load(directory / "checkpoint.pt", weights_only=True)
            checkpoint["selection"] = {k: v for k, v in changed.items() if k != "artifact_sha256"}
            torch.save(checkpoint, directory / "checkpoint.pt")
            changed["artifact_sha256"]["checkpoint.pt"] = file_hash(directory / "checkpoint.pt")
            DRIVER.write_json(directory / "selection.json", changed)
            model, changed = DRIVER.reload_selected(
                directory, "history_deepsets", [0, 1, 24], self.config, 11, self.protocol
            )
            with self.assertRaisesRegex(ValueError, message):
                DRIVER.audit_validation(
                    directory,
                    model,
                    changed,
                    self.store,
                    self.rows["valid"],
                    [0, 1, 24],
                    self.config,
                )
            restore()
        DRIVER.write_json(directory / "selection.json", {**selection, "artifact_sha256": {}})
        with self.assertRaisesRegex(ValueError, "inventory"):
            DRIVER.reload_selected(
                directory, "history_deepsets", [0, 1, 24], self.config, 11, self.protocol
            )

    def test_final_identity_labels_probabilities_and_metrics_are_checked(self):
        directory, model, selection = self.fit()
        rows = self.rows["test"]
        raw = DRIVER.predict(model, self.store, rows, [0, 1, 24], 3)
        p = softmax(raw.astype(float) / selection["temperature"], axis=1)[:, 1]
        arrays = {
            "row_id": self.store.row_id[rows],
            "y_true": self.store.target[rows],
            "raw": raw,
            "p_positive": p,
            "time": self.store.time[rows],
            "history_count": self.store.history[rows].astype(bool).sum(-1),
        }
        prediction = directory / "predictions.npz"
        np.savez_compressed(prediction, **arrays)
        cell = "outcome_history/history_deepsets/seed-11"
        record = {
            "cell": cell,
            "variant": "outcome_history",
            "model": "history_deepsets",
            "columns": [0, 1, 24],
            "configuration": self.config,
            "seed": 11,
            "protocol_sha256": self.protocol,
            "selection_sha256": file_hash(directory / "selection.json"),
            "prediction_sha256": file_hash(prediction),
            "metrics": binary_metrics(arrays["y_true"], p),
            "raw_metrics": binary_metrics(
                arrays["y_true"], softmax(raw.astype(float), axis=1)[:, 1]
            ),
            "reload_max_abs_error": 0.0,
            "test_evaluated": True,
        }
        DRIVER.audit_result(directory, selection, self.store, rows, cell, self.protocol, record)
        DRIVER.write_json(directory / "result.json", record)
        DRIVER.audit_result(directory, selection, self.store, rows, cell, self.protocol)
        for key, value in [("variant", "wrong"), ("model", "wrong"), ("seed", 33)]:
            with self.assertRaisesRegex(ValueError, "identity"):
                DRIVER.audit_result(
                    directory,
                    selection,
                    self.store,
                    rows,
                    cell,
                    self.protocol,
                    {**record, key: value},
                )
        original = prediction.read_bytes()
        for changed, message in [
            ({"y_true": 1 - arrays["y_true"]}, "labels"),
            ({"time": arrays["time"] + 1}, "times"),
            ({"p_positive": np.clip(p + 0.01, 0, 1)}, "probability"),
        ]:
            np.savez_compressed(prediction, **{**arrays, **changed})
            with self.assertRaisesRegex(ValueError, message):
                DRIVER.audit_result(
                    directory,
                    selection,
                    self.store,
                    rows,
                    cell,
                    self.protocol,
                    {**record, "prediction_sha256": file_hash(prediction)},
                )
            prediction.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "metrics"):
            DRIVER.audit_result(
                directory,
                selection,
                self.store,
                rows,
                cell,
                self.protocol,
                {**record, "metrics": {**record["metrics"], "log_loss": 123.0}},
            )

    def test_completed_resume_preserves_evidence_and_rejects_changed_protocol(self):
        output = self.root / "main-run"
        config = {**self.config, "epochs": 1, "patience": 1}
        with patch.multiple(
            DRIVER, ROOT=self.root, DATA=self.data, OUT=output, CONFIG=config, SEEDS=[11]
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                DRIVER.main()
            self.assertEqual(DRIVER.read(output / "status.json")["status"], "completed")
            self.assertEqual(len(list(output.glob("*/*/seed-*/result.json"))), 4)
            before = {
                p: (p.read_bytes(), p.stat().st_mtime_ns)
                for p in output.rglob("*")
                if p.is_file() and p.name != "status.json"
            }
            with patch.object(DRIVER, "fit", side_effect=AssertionError("Unexpected refit")):
                DRIVER.main()
            self.assertTrue(
                all((p.read_bytes(), p.stat().st_mtime_ns) == state for p, state in before.items())
            )
            status = (output / "status.json").read_bytes()
            protocol = DRIVER.read(output / "protocol.json")
            DRIVER.write_json(output / "protocol.json", {**protocol, "classes": 3})
            with self.assertRaisesRegex(ValueError, "protocol"):
                DRIVER.main()
            self.assertFalse(DRIVER.RUN_IDENTITY_VALIDATED)
            self.assertEqual((output / "status.json").read_bytes(), status)


if __name__ == "__main__":
    unittest.main()
