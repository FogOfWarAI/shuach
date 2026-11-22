# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import softmax

import gamematch
from gamematch.data.heroes import file_hash
from gamematch.data.indexed_history import IndexedHistoryStore
from gamematch.indexed_inference import IndexedHistoryPredictor
from gamematch.runner import write_json

SPEC = importlib.util.spec_from_file_location(
    "indexed_inference_fixture_driver",
    Path(__file__).resolve().parents[1] / "scripts/run_league_neural.py",
)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class IndexedInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        rng = np.random.default_rng(19)
        performance = rng.normal(size=(5, 24)).astype(np.float32)
        performance[0] = 0
        arrays = {
            "performance": performance,
            "history-index": rng.integers(0, 5, (16, 2, 5, 10), dtype=np.int32),
            "appearance-available": np.zeros(5),
            "match-start": np.arange(16.0) + 10,
            "static": rng.normal(size=(16, 2, 5, 6)).astype(np.float32),
            "target": np.arange(16, dtype=np.int64) % 2,
            "row-id": np.array([f"row-{i}" for i in range(16)]),
            "train-rows": np.arange(8),
            "valid-rows": np.arange(8, 12),
            "test-rows": np.arange(12, 16),
        }
        arrays["history-index"][0] = 0
        for name, value in arrays.items():
            np.save(self.data / (name + ".npy"), value, allow_pickle=False)
        write_json(
            self.data / "manifest.json",
            {
                "classes": 2,
                "team_size": 5,
                "history_length": 10,
                "files_sha256": {p.name: file_hash(p) for p in self.data.glob("*.npy")},
                "normalizer": {
                    "history_mean": [0.0] * 25,
                    "history_scale": [1.0] * 25,
                    "static_mean": [0.0] * 6,
                    "static_scale": [1.0] * 6,
                },
            },
        )
        self.store = IndexedHistoryStore(self.data)
        self.config = {
            **DRIVER.CONFIG,
            "width": 8,
            "epochs": 1,
            "patience": 1,
            "batch_size": 4,
            "threads": 1,
        }
        self.protocol = {
            "data_manifest_sha256": file_hash(self.data / "manifest.json"),
            "configuration": self.config,
            "models": ["history_deepsets", "owo"],
            "seeds": [11],
            "classes": 2,
            "variants": {"outcome_history": [0, 1, 24], "performance_history": list(range(25))},
        }
        write_json(self.root / "protocol.json", self.protocol)
        shutil.copytree(
            Path(gamematch.__file__).parent / "models",
            self.root / "source-snapshot/gamematch/models",
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def fitted(self, name="history_deepsets", columns=None):
        columns = columns or [0, 1, 24]
        directory = self.root / name / str(len(columns)) / "seed-11"
        with contextlib.redirect_stdout(io.StringIO()):
            model, selected = DRIVER.fit(
                name,
                self.store,
                np.arange(8),
                np.arange(8, 12),
                columns,
                self.config,
                11,
                directory,
                file_hash(self.root / "protocol.json"),
            )
        return directory, model, selected

    def test_reload_both_heads_information_sets_and_row_order(self):
        rows = np.array([15, 0, 12, 15, 1])
        for name in ["history_deepsets", "owo"]:
            for columns in [[0, 1, 24], list(range(25))]:
                with self.subTest(model=name, width=len(columns)):
                    directory, model, selected = self.fitted(name, columns)
                    expected = DRIVER.predict(model, self.store, rows, columns, 5)
                    predictor = IndexedHistoryPredictor(directory, self.data, threads=1)
                    np.testing.assert_allclose(
                        predictor.predict_logits(rows, 2), expected, atol=1e-6, rtol=0
                    )
                    np.testing.assert_allclose(
                        predictor.predict_proba(rows, 2),
                        softmax(expected.astype(float) / selected["temperature"], axis=1),
                        atol=1e-6,
                    )
                    np.testing.assert_allclose(
                        predictor.predict_proba(rows, 2, calibrated=False),
                        softmax(expected.astype(float), axis=1),
                        atol=1e-6,
                    )
                    self.assertEqual(predictor.predict_proba(np.array([], dtype=int)).shape, (0, 2))

    def test_invalid_rows_and_devices_rejected(self):
        directory, _, _ = self.fitted()
        predictor = IndexedHistoryPredictor(directory, self.data, threads=1)
        for rows in [np.array([-1]), np.array([16]), np.array([1.5]), np.array([[1]])]:
            with self.assertRaises(ValueError):
                predictor.predict_proba(rows)
        for batch in [0, True, 1.5]:
            with self.assertRaises(ValueError):
                predictor.predict_proba(np.array([1]), batch)
        for arguments in [{"device": "other"}, {"threads": 0}, {"threads": True}]:
            with self.assertRaises(ValueError):
                IndexedHistoryPredictor(directory, self.data, **arguments)

    def test_changed_store_checkpoint_and_source_rejected(self):
        directory, _, _ = self.fitted()
        manifest = self.data / "manifest.json"
        original = manifest.read_bytes()
        manifest.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "original feature store"):
            IndexedHistoryPredictor(directory, self.data)
        manifest.write_bytes(original)
        checkpoint = directory / "checkpoint.pt"
        original_checkpoint = checkpoint.read_bytes()
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            IndexedHistoryPredictor(directory, self.data)
        checkpoint.write_bytes(original_checkpoint)
        source = self.root / "source-snapshot/gamematch/models/history.py"
        source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "implementation differs"):
            IndexedHistoryPredictor(directory, self.data)

    def test_nonpositive_temperature_rejected(self):
        directory, _, _ = self.fitted()
        path = directory / "selection.json"
        selected = json.loads(path.read_text())
        selected["temperature"] = 0
        write_json(path, selected)
        with self.assertRaisesRegex(ValueError, "positive validation temperature"):
            IndexedHistoryPredictor(directory, self.data)
