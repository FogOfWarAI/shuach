# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Reject incomplete or internally inconsistent feature/readout result archives."""

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "rich_exporter", SCRIPTS / "export_rich_history_results.py"
)
EXPORT = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(EXPORT)
finally:
    sys.path.pop(0)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


class RichExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gamematch-rich-export-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "runs/rich-history-v1"
        self.base.mkdir(parents=True)
        source = self.base / "source-snapshot/gamematch/module.py"
        source.parent.mkdir(parents=True)
        source.write_text("# small source identity fixture\n")
        driver = self.base / "source-driver.py"
        driver.write_text("# small driver identity fixture\n")
        cache = self.root / ".cache/rich-history-v1.npz"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"cache identity fixture, no dataset")
        manifest = {"sha256": EXPORT.file_hash(cache), "identity": {"fixture": "builder"}}
        write(cache.with_suffix(".json"), manifest)
        self.cell = copy.deepcopy(EXPORT.CELLS[0])
        self.configuration = {
            "device": "cpu",
            "width": 4,
            "heads": 2,
            "layers": 1,
            "dropout": 0.1,
            "norm_first": False,
            "epochs": 2,
        }
        self.protocol = {
            "seeds": list(EXPORT.SEEDS),
            "cells": copy.deepcopy(EXPORT.CELLS),
            "fits": 30,
            "configuration": self.configuration,
            "source_sha256": hashlib.sha256(b"module.py" + source.read_bytes()).hexdigest(),
            "driver_sha256": EXPORT.file_hash(driver),
            "cache_sha256": EXPORT.file_hash(cache),
            "cache_manifest_sha256": EXPORT.file_hash(cache.with_suffix(".json")),
            "cache_generation_identity": manifest["identity"],
        }
        self.folder = self.base / EXPORT.cell_name(self.cell) / "seed-11"
        self.folder.mkdir(parents=True)
        self.cfg = {**self.configuration, "pooling": "global"}
        self.specification = {
            "history_features": 5,
            "static_features": 13,
            "team_size": 6,
            "history_length": 10,
            "classes": 3,
        }
        model = EXPORT.model_factory("history", "history_deepsets", self.specification, self.cfg)
        self.checkpoint = {
            "state_dict": model.state_dict(),
            "task": "history",
            "model": "history_deepsets",
            "specification": self.specification,
            "configuration": self.cfg,
            "seed": 11,
            "best_epoch": 2,
        }
        torch.save(self.checkpoint, self.folder / "checkpoint.pt")
        self.valid = {
            "row_id": np.arange(6),
            "y_true": np.array([0, 1, 2, 0, 1, 2]),
            "raw": np.array(
                [[2, 1, 0], [1, 3, 0], [0, 1, 1], [2, 3, 0], [1, 2, 1], [0, 1, 3]], dtype=np.float32
            ),
        }
        self.validation_reference = {k: v.copy() for k, v in self.valid.items() if k != "raw"}
        np.savez(self.folder / "validation.npz", **self.valid)
        temperature = EXPORT.temperature_fit(self.valid["raw"], self.valid["y_true"])
        self.predictions = {
            "row_id": np.arange(6, 12),
            "y_true": self.valid["y_true"].copy(),
            "time": np.arange(6) * 86400,
            "domain": np.arange(6) % 2,
            "history_count": np.ones((6, 2, 6), dtype=int),
            "raw": self.valid["raw"].copy(),
            "probabilities": softmax(self.valid["raw"] / temperature, axis=1),
        }
        self.reference = {
            k: v.copy() for k, v in self.predictions.items() if k not in {"raw", "probabilities"}
        }
        np.savez(self.folder / "predictions.npz", **self.predictions)
        score = EXPORT.primary_loss("history", self.valid["y_true"], self.valid["raw"], {})
        self.training = {
            "configuration": self.cfg,
            "seed": 11,
            "model": "history_deepsets",
            "best_epoch": 2,
            "epochs_completed": 2,
            "validation_primary": score,
            "selection_metric": "raw_log_loss",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "seconds": 1.0,
            "test_evaluated": False,
        }
        write(self.folder / "metrics.json", self.training)
        self.history = [
            {"epoch": 1, "validation_primary": score + 0.1},
            {"epoch": 2, "validation_primary": score},
        ]
        write(self.folder / "history.json", self.history)
        measured, raw = EXPORT.metrics(self.predictions), EXPORT.metrics(self.predictions, raw=True)
        self.result = {
            **self.training,
            "test_evaluated": True,
            "cell": self.cell,
            "protocol_sha256": "fixture-protocol",
            "temperature": temperature,
            "reload_max_abs_error": 0.0,
            "metrics": {k: measured[k] for k in EXPORT.METRIC_KEYS},
            "uncalibrated_metrics": {k: raw[k] for k in EXPORT.METRIC_KEYS},
            "classification_diagnostics": {k: measured[k] for k in EXPORT.DIAGNOSTIC_KEYS},
        }
        write(self.folder / "result.json", self.result)

    def audit(self):
        return EXPORT.audit_fit(
            self.folder,
            self.cell,
            11,
            self.protocol,
            "fixture-protocol",
            self.reference,
            self.validation_reference,
        )

    def test_valid_archive_and_fit_pass(self):
        EXPORT.verify_archive(self.protocol, self.base, self.root)
        data, result, hashes = self.audit()
        self.assertEqual(result["seed"], 11)
        self.assertEqual(data["raw"].shape, (6, 3))
        self.assertEqual(len(hashes), 6)

    def test_three_distinct_seeds_and_all_ten_cells_are_required(self):
        for change in [
            {"seeds": [11]},
            {"seeds": [11, 11, 33]},
            {"fits": 10},
            {"cells": EXPORT.CELLS[:-1]},
            {"cells": EXPORT.CELLS[:-1] + [EXPORT.CELLS[0]]},
        ]:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "three distinct"):
                EXPORT.verify_archive({**self.protocol, **change}, self.base, self.root)

    def test_cell_completion_needs_three_results_and_matching_status(self):
        parent = self.folder.parent
        completed = {
            "status": "completed",
            "completed_seeds": EXPORT.SEEDS,
            "expected_seeds": EXPORT.SEEDS,
        }
        write(parent / "status.json", completed)
        self.assertEqual(EXPORT.cell_completion(parent), ([11], False))
        for seed in [22, 33]:
            write(parent / f"seed-{seed}/result.json", {})
        self.assertEqual(EXPORT.cell_completion(parent), (EXPORT.SEEDS, True))
        write(parent / "status.json", {**completed, "completed_seeds": [11]})
        with self.assertRaisesRegex(ValueError, "seed identities"):
            EXPORT.cell_completion(parent)
        (parent / "status.json").unlink()
        self.assertEqual(EXPORT.cell_completion(parent), (EXPORT.SEEDS, False))

    def test_changed_frozen_source_driver_and_cache_are_rejected(self):
        for path in [
            self.base / "source-snapshot/gamematch/module.py",
            self.base / "source-driver.py",
            self.root / ".cache/rich-history-v1.npz",
            self.root / ".cache/rich-history-v1.json",
        ]:
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with (
                self.subTest(path=path.name),
                self.assertRaisesRegex(ValueError, "identity changed"),
            ):
                EXPORT.verify_archive(self.protocol, self.base, self.root)
            path.write_bytes(original)

    def test_cache_manifest_must_agree_with_protocol(self):
        changed = {**self.protocol, "cache_generation_identity": {"fixture": "other"}}
        with self.assertRaisesRegex(ValueError, "Cache manifest"):
            EXPORT.verify_archive(changed, self.base, self.root)

    def test_test_and_validation_row_identity_are_checked(self):
        for name, original in [
            ("predictions.npz", self.predictions),
            ("validation.npz", self.valid),
        ]:
            for key in ["row_id", "y_true"]:
                altered = {**original, key: original[key][::-1]}
                np.savez(self.folder / name, **altered)
                with self.subTest(name=name, key=key), self.assertRaises(AssertionError):
                    self.audit()
                np.savez(self.folder / name, **original)

    def test_validation_score_and_selected_epoch_are_checked(self):
        changed = {**self.valid, "raw": self.valid["raw"] + np.array([0.5, 0, 0])}
        np.savez(self.folder / "validation.npz", **changed)
        with self.assertRaisesRegex(ValueError, "selection score"):
            self.audit()
        np.savez(self.folder / "validation.npz", **self.valid)
        write(self.folder / "history.json", [self.history[0]])
        with self.assertRaisesRegex(ValueError, "epoch history"):
            self.audit()
        write(
            self.folder / "history.json",
            [
                {
                    **self.history[0],
                    "validation_primary": self.training["validation_primary"] - 0.1,
                },
                self.history[1],
            ],
        )
        with self.assertRaisesRegex(ValueError, "selected epoch"):
            self.audit()

    def test_consistently_rewritten_test_temperature_is_rejected(self):
        temperature = self.result["temperature"] * 1.5
        rewritten = {
            **self.predictions,
            "probabilities": softmax(self.predictions["raw"] / temperature, axis=1),
        }
        measured = EXPORT.metrics(rewritten)
        result = {
            **self.result,
            "temperature": temperature,
            "metrics": {k: measured[k] for k in EXPORT.METRIC_KEYS},
            "classification_diagnostics": {k: measured[k] for k in EXPORT.DIAGNOSTIC_KEYS},
        }
        np.savez(self.folder / "predictions.npz", **rewritten)
        write(self.folder / "result.json", result)
        with self.assertRaisesRegex(ValueError, "Temperature cannot be recovered"):
            self.audit()

    def test_missing_or_falsified_metrics_are_rejected(self):
        for field in ["metrics", "uncalibrated_metrics", "classification_diagnostics"]:
            for replacement in [{}, {k: 0 for k in self.result[field]}]:
                write(self.folder / "result.json", {**self.result, field: replacement})
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "metric"):
                    self.audit()

    def test_checkpoint_identity_nonfinite_weights_and_reload_failure_are_rejected(self):
        torch.save({**self.checkpoint, "seed": 22}, self.folder / "checkpoint.pt")
        with self.assertRaisesRegex(ValueError, "Checkpoint identity"):
            self.audit()
        altered = copy.deepcopy(self.checkpoint)
        next(iter(altered["state_dict"].values())).view(-1)[0] = float("nan")
        torch.save(altered, self.folder / "checkpoint.pt")
        with self.assertRaisesRegex(ValueError, "nonfinite weights"):
            self.audit()
        torch.save(self.checkpoint, self.folder / "checkpoint.pt")
        write(self.folder / "result.json", {**self.result, "reload_max_abs_error": 0.01})
        with self.assertRaisesRegex(ValueError, "reload"):
            self.audit()

    def test_prediction_shape_and_nonfinite_logits_are_rejected(self):
        for raw in [np.zeros((6, 4)), np.full((6, 3), np.nan)]:
            np.savez(self.folder / "predictions.npz", **{**self.predictions, "raw": raw})
            with self.assertRaisesRegex(ValueError, "finite, aligned three-class"):
                self.audit()

    def test_positive_and_negative_interaction_directions(self):
        terms = [
            ("rich_team", 1),
            ("original_team", -1),
            ("rich_global", -1),
            ("original_global", 1),
        ]
        for rich_team, expected in [(0.4, -0.4), (1.2, 0.4)]:
            losses = {
                "rich_team": rich_team,
                "original_team": 1.0,
                "rich_global": 0.7,
                "original_global": 0.9,
            }
            arrays = {}
            for name, loss in losses.items():
                p = np.array([[np.exp(-loss), (1 - np.exp(-loss)) / 2, (1 - np.exp(-loss)) / 2]])
                arrays[name] = [
                    {"y_true": np.array([0]), "probabilities": p, "raw": np.log(p)}
                    for _ in EXPORT.SEEDS
                ]
            for raw in [False, True]:
                with self.subTest(expected=expected, raw=raw):
                    np.testing.assert_allclose(
                        EXPORT.loss_contrast(arrays, terms, raw=raw), expected, atol=1e-12
                    )


if __name__ == "__main__":
    unittest.main()
