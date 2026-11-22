# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Export admission checks for local LoL fit artifacts, without research extras."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_league_results.py"
SPEC = importlib.util.spec_from_file_location("league_exporter", SCRIPT)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


class LeagueExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gamematch-league-export-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "runs/league-baselines-v1"
        source = self.base / "source-snapshot/gamematch/module.py"
        source.parent.mkdir(parents=True)
        source.write_text("# frozen source fixture\n")
        driver = self.base / "source-driver.py"
        driver.write_text("# frozen driver fixture\n")
        self.manifest = self.root / "manifest.json"
        write(self.manifest, {"fixture": True})
        self.splits = {
            "train": np.arange(20),
            "valid": np.arange(20, 28),
            "test": np.arange(28, 36),
        }
        np.savez(self.base / "split-identity.npz", **self.splits)
        self.protocol = {
            "variants": copy.deepcopy(EXPORT.COLUMNS),
            "spaces": copy.deepcopy(EXPORT.SPACES),
            "seed": 20261114,
            "counts": {k: len(v) for k, v in self.splits.items()},
            "source_sha256": hashlib.sha256(b"module.py" + source.read_bytes()).hexdigest(),
            "driver_sha256": EXPORT.file_hash(driver),
            "data_manifest_sha256": EXPORT.file_hash(self.manifest),
        }
        self.folder = self.base / "outcome_history/logistic"
        self.folder.mkdir(parents=True)
        self.validation_reference = {"row_id": self.splits["valid"], "y_true": np.tile([0, 1], 4)}
        self.reference = {
            "row_id": self.splits["test"],
            "y_true": np.tile([0, 1], 4),
            "time": np.arange(8) * 86400,
            "history_count": np.ones((8, 2, 5), dtype=int),
        }
        rng = np.random.default_rng(41)
        train, valid, test = (
            rng.normal(size=(20, 52)),
            rng.normal(size=(8, 52)),
            rng.normal(size=(8, 52)),
        )
        labels = np.tile([0, 1], 10)
        self.candidates, test_probability = [], []
        for index, config in enumerate(EXPORT.SPACES["logistic"]):
            estimator = make_pipeline(
                StandardScaler(), LogisticRegression(**config, random_state=20261114, max_iter=2000)
            )
            estimator.fit(train, labels)
            trial = self.folder / f"trial-{index}"
            trial.mkdir()
            probability = estimator.predict_proba(valid)[:, 1]
            joblib.dump(estimator, trial / "checkpoint.joblib")
            np.savez(trial / "validation.npz", **self.validation_reference, p_positive=probability)
            record = {
                "variant": "outcome_history",
                "model": "logistic",
                "trial": index,
                "configuration": config,
                "protocol_sha256": "fixture-protocol",
                "features": 52,
                "seed": 20261114,
                "test_evaluated": False,
                "iterations": EXPORT.require_logistic_convergence(estimator[-1]),
                "metrics": EXPORT.binary_metrics(self.validation_reference["y_true"], probability),
                "checkpoint_sha256": EXPORT.file_hash(trial / "checkpoint.joblib"),
                "validation_sha256": EXPORT.file_hash(trial / "validation.npz"),
                "reload_max_abs_error": 0.0,
            }
            write(trial / "validation.json", record)
            self.candidates.append(record)
            test_probability.append(estimator.predict_proba(test)[:, 1])
        self.selection = min(self.candidates, key=lambda r: (r["metrics"]["log_loss"], r["trial"]))
        write(self.folder / "selection.json", self.selection)
        probability = test_probability[self.selection["trial"]]
        self.predictions = {k: v for k, v in self.reference.items() if k != "history_count"}
        self.predictions["p_positive"] = probability
        np.savez(self.folder / "predictions.npz", **self.predictions)
        self.result = {
            **{
                key: self.selection[key]
                for key in ["variant", "model", "trial", "configuration", "seed"]
            },
            "protocol_sha256": "fixture-protocol",
            "test_evaluated": True,
            "selection_sha256": EXPORT.file_hash(self.folder / "selection.json"),
            "predictions_sha256": EXPORT.file_hash(self.folder / "predictions.npz"),
            "metrics": EXPORT.binary_metrics(self.reference["y_true"], probability),
            "predicted_positive_fraction": float((probability >= 0.5).mean()),
            "reload_max_abs_error": 0.0,
            "test_reload_max_abs_error": 0.0,
        }
        write(self.folder / "result.json", self.result)

    def audit(self):
        return EXPORT.load_fit(
            self.folder,
            "fixture-protocol",
            self.reference,
            False,
            "outcome_history",
            "logistic",
            20261114,
            self.protocol,
            self.validation_reference,
        )

    def verify(self, protocol=None):
        EXPORT.verify_protocol(
            self.base, protocol or self.protocol, False, self.manifest, self.splits
        )

    def test_valid_three_candidate_selection_and_test_artifacts_pass(self):
        self.verify()
        data, raw, hashes = self.audit()
        np.testing.assert_array_equal(data["p_positive"], raw["p_positive"])
        self.assertEqual(len(hashes), 12)

    def test_helpers_import_without_matplotlib(self):
        code = """
import importlib.abc
import importlib.util
import sys
class BlockResearch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'plotting_fonts'}:
            raise AssertionError('Research dependency imported by audit helper')
sys.meta_path.insert(0, BlockResearch())
spec = importlib.util.spec_from_file_location('audit', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
"""
        subprocess.run([sys.executable, "-c", code, str(SCRIPT)], check=True, capture_output=True)

    def test_source_driver_manifest_and_split_tampering_are_rejected(self):
        for path in [
            self.base / "source-snapshot/gamematch/module.py",
            self.base / "source-driver.py",
            self.manifest,
        ]:
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with (
                self.subTest(path=path.name),
                self.assertRaisesRegex(ValueError, "identity changed"),
            ):
                self.verify()
            path.write_bytes(original)
        np.savez(
            self.base / "split-identity.npz", **{**self.splits, "valid": self.splits["valid"][::-1]}
        )
        with self.assertRaises(AssertionError):
            self.verify()

    def test_declared_budget_and_information_sets_cannot_shrink(self):
        for changes in [
            {"spaces": {"logistic": EXPORT.SPACES["logistic"][:1]}},
            {"seed": 11},
            {"variants": {"outcome_history": [0, 1]}},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.verify({**self.protocol, **changes})

    def test_missing_trial_or_checkpoint_prevents_export(self):
        missing = self.folder / "trial-2/validation.json"
        original = missing.read_bytes()
        missing.unlink()
        with self.assertRaises(FileNotFoundError):
            self.audit()
        missing.write_bytes(original)
        (self.folder / "trial-1/checkpoint.joblib").unlink()
        with self.assertRaises(FileNotFoundError):
            self.audit()

    def test_selected_candidate_must_be_validation_minimum(self):
        worst = max(self.candidates, key=lambda r: r["metrics"]["log_loss"])
        self.assertNotEqual(worst["trial"], self.selection["trial"])
        write(self.folder / "selection.json", worst)
        write(
            self.folder / "result.json",
            {**self.result, "selection_sha256": EXPORT.file_hash(self.folder / "selection.json")},
        )
        with self.assertRaisesRegex(ValueError, "lowest validation"):
            self.audit()

    def test_all_candidate_validation_metrics_and_identities_are_recomputed(self):
        path = self.folder / "trial-2/validation.json"
        record = EXPORT.read(path)
        write(path, {**record, "metrics": {**record["metrics"], "log_loss": 0.0}})
        with self.assertRaisesRegex(ValueError, "saved metric"):
            self.audit()
        write(path, record)
        validation_path = self.folder / "trial-2/validation.npz"
        with np.load(validation_path, allow_pickle=False) as values:
            altered = dict(values)
        altered["row_id"] = altered["row_id"][::-1]
        np.savez(validation_path, **altered)
        write(path, {**record, "validation_sha256": EXPORT.file_hash(validation_path)})
        with self.assertRaises(AssertionError):
            self.audit()

    def test_checkpoint_hash_and_recorded_hyperparameters_are_checked(self):
        checkpoint = self.folder / "trial-1/checkpoint.joblib"
        original = checkpoint.read_bytes()
        checkpoint.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            self.audit()
        checkpoint.write_bytes(original)
        estimator = joblib.load(checkpoint)
        estimator[-1].set_params(C=999)
        joblib.dump(estimator, checkpoint)
        record_path = self.folder / "trial-1/validation.json"
        write(
            record_path,
            {**EXPORT.read(record_path), "checkpoint_sha256": EXPORT.file_hash(checkpoint)},
        )
        with self.assertRaisesRegex(ValueError, "hyperparameters"):
            self.audit()

    def test_final_seed_trial_metrics_and_diagnostics_are_checked(self):
        for changes in [
            {"seed": 11},
            {"trial": 99},
            {"metrics": {}},
            {"metrics": {**self.result["metrics"], "accuracy": 1.0}},
            {"metrics": {**self.result["metrics"], "n": [8, 8]}},
            {"predicted_positive_fraction": 2.0},
            {"test_reload_max_abs_error": 1.0},
        ]:
            write(self.folder / "result.json", {**self.result, **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.audit()

    def test_test_checksum_and_row_identity_are_checked(self):
        path = self.folder / "predictions.npz"
        np.savez(path, **{**self.predictions, "row_id": self.predictions["row_id"][::-1]})
        with self.assertRaisesRegex(ValueError, "predictions changed"):
            self.audit()
        write(
            self.folder / "result.json",
            {**self.result, "predictions_sha256": EXPORT.file_hash(path)},
        )
        with self.assertRaises(AssertionError):
            self.audit()

    def test_neural_completion_gate_requires_distinct_declared_seeds(self):
        for seed in [11, 22]:
            write(self.base / f"outcome_history/owo/seed-{seed}/result.json", {})
        folders, seeds, completed = EXPORT.cell_folders(
            self.base, "outcome_history", "owo", True, None
        )
        self.assertEqual(seeds, [11, 22, 33])
        self.assertEqual(completed, 2)
        write(folders[-1] / "result.json", {})
        self.assertEqual(EXPORT.cell_folders(self.base, "outcome_history", "owo", True, None)[2], 3)
        protocol = {**self.protocol, "models": ["history_deepsets", "owo"], "classes": 2}
        for seeds in [[11], [11, 11, 33]]:
            with self.assertRaisesRegex(ValueError, "neural budget"):
                EXPORT.verify_protocol(
                    self.base, {**protocol, "seeds": seeds}, True, self.manifest, self.splits
                )


class NeuralSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gamematch-league-neural-export-")
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.configuration = {"width": 4, "dropout": 0.1, "epochs": 2}
        self.protocol = {"configuration": self.configuration}
        self.valid = {
            "row_id": np.arange(6),
            "y_true": np.array([0, 1, 0, 1, 0, 1]),
            "raw": np.array([[2, 0], [1, 2], [1, 0], [2, 1], [1, 0], [0, 3]], dtype=np.float32),
        }
        self.reference = {key: value.copy() for key, value in self.valid.items() if key != "raw"}
        estimator = EXPORT.build_neural_model(
            "history_deepsets", EXPORT.COLUMNS["outcome_history"], self.configuration
        )
        score = EXPORT.binary_metrics(
            self.valid["y_true"], softmax(self.valid["raw"].astype(float), axis=1)[:, 1]
        )["log_loss"]
        self.selection = {
            "model": "history_deepsets",
            "columns": EXPORT.COLUMNS["outcome_history"],
            "configuration": self.configuration,
            "seed": 11,
            "protocol_sha256": "fixture",
            "test_evaluated": False,
            "epochs_completed": 2,
            "best_epoch": 2,
            "validation_log_loss": score,
            "temperature": EXPORT.temperature_fit(self.valid["raw"], self.valid["y_true"]),
            "parameters": sum(p.numel() for p in estimator.parameters()),
            "seconds": 1.0,
        }
        self.checkpoint = {
            "state_dict": estimator.state_dict(),
            "selection": copy.deepcopy(self.selection),
        }
        torch.save(self.checkpoint, self.folder / "checkpoint.pt")
        np.savez(self.folder / "validation.npz", **self.valid)
        self.history = [
            {"epoch": 1, "validation_log_loss": score + 0.1},
            {"epoch": 2, "validation_log_loss": score},
        ]
        write(self.folder / "history.json", self.history)
        self.refresh_hashes()
        write(self.folder / "selection.json", self.selection)
        self.test_reference = {
            "row_id": np.arange(6, 12),
            "y_true": self.valid["y_true"].copy(),
            "time": np.arange(6) * 86400,
            "history_count": np.ones((6, 2, 5), dtype=int),
        }
        probability = softmax(
            self.valid["raw"].astype(float) / self.selection["temperature"], axis=1
        )[:, 1]
        self.predictions = {
            **self.test_reference,
            "raw": self.valid["raw"].copy(),
            "p_positive": probability,
        }
        np.savez(self.folder / "predictions.npz", **self.predictions)
        self.result = {
            "cell": "outcome_history/history_deepsets/seed-11",
            "variant": "outcome_history",
            "model": "history_deepsets",
            "columns": EXPORT.COLUMNS["outcome_history"],
            "configuration": self.configuration,
            "seed": 11,
            "protocol_sha256": "fixture",
            "selection_sha256": EXPORT.file_hash(self.folder / "selection.json"),
            "prediction_sha256": EXPORT.file_hash(self.folder / "predictions.npz"),
            "metrics": EXPORT.binary_metrics(self.test_reference["y_true"], probability),
            "raw_metrics": EXPORT.binary_metrics(
                self.test_reference["y_true"],
                softmax(self.valid["raw"].astype(float), axis=1)[:, 1],
            ),
            "reload_max_abs_error": 0.0,
            "test_evaluated": True,
        }
        write(self.folder / "result.json", self.result)

    def refresh_hashes(self):
        self.selection["artifact_sha256"] = {
            name: EXPORT.file_hash(self.folder / name)
            for name in ["checkpoint.pt", "validation.npz", "history.json"]
        }

    def audit(self):
        return EXPORT.audit_neural_selection(
            self.folder,
            self.selection,
            self.protocol,
            "fixture",
            self.reference,
            "outcome_history",
            "history_deepsets",
            11,
        )

    def test_valid_neural_selection_and_checkpoint_pass(self):
        self.assertEqual(len(self.audit()), 3)

    def load_fit(self):
        return EXPORT.load_fit(
            self.folder,
            "fixture",
            self.test_reference,
            True,
            "outcome_history",
            "history_deepsets",
            11,
            self.protocol,
            self.reference,
        )

    def test_neural_fit_recomputes_calibrated_and_raw_metrics(self):
        data, raw, hashes = self.load_fit()
        self.assertEqual(len(hashes), 6)
        self.assertFalse(np.array_equal(data["p_positive"], raw["p_positive"]))
        for changes in [
            {"metrics": {}},
            {"raw_metrics": {}},
            {"seed": 22},
            {"metrics": {**self.result["metrics"], "log_loss": 0.0}},
        ]:
            write(self.folder / "result.json", {**self.result, **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load_fit()

    def test_neural_prediction_metadata_and_temperature_transform_are_checked(self):
        for changes in [
            {"history_count": np.zeros((6, 2, 5), dtype=int)},
            {"raw": np.zeros((6, 3))},
            {"p_positive": np.full(6, 0.5)},
        ]:
            predictions = {**self.predictions, **changes}
            np.savez(self.folder / "predictions.npz", **predictions)
            write(
                self.folder / "result.json",
                {
                    **self.result,
                    "prediction_sha256": EXPORT.file_hash(self.folder / "predictions.npz"),
                    "metrics": EXPORT.binary_metrics(
                        predictions["y_true"], predictions["p_positive"]
                    ),
                },
            )
            with (
                self.subTest(changes=list(changes)),
                self.assertRaises((ValueError, AssertionError)),
            ):
                self.load_fit()

    def test_missing_artifact_hash_and_wrong_columns_are_rejected(self):
        self.selection["artifact_sha256"].pop("validation.npz")
        with self.assertRaisesRegex(ValueError, "artifact inventory"):
            self.audit()
        self.refresh_hashes()
        self.selection["columns"] = [0]
        with self.assertRaisesRegex(ValueError, "neural selection"):
            self.audit()

    def test_temperature_must_be_recovered_from_validation(self):
        self.selection["temperature"] *= 1.5
        self.checkpoint["selection"]["temperature"] = self.selection["temperature"]
        torch.save(self.checkpoint, self.folder / "checkpoint.pt")
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "Temperature cannot be recovered"):
            self.audit()

    def test_validation_rows_score_and_best_epoch_are_checked(self):
        np.savez(
            self.folder / "validation.npz", **{**self.valid, "row_id": self.valid["row_id"][::-1]}
        )
        self.refresh_hashes()
        with self.assertRaises(AssertionError):
            self.audit()
        np.savez(self.folder / "validation.npz", **self.valid)
        self.refresh_hashes()
        self.selection["validation_log_loss"] = 0.0
        with self.assertRaisesRegex(ValueError, "LogLoss cannot be reproduced"):
            self.audit()
        self.selection["validation_log_loss"] = self.history[1]["validation_log_loss"]
        self.selection["best_epoch"] = 1
        with self.assertRaisesRegex(ValueError, "selected epoch"):
            self.audit()

    def test_checkpoint_selection_and_nonfinite_weights_are_rejected(self):
        self.checkpoint["selection"]["seed"] = 22
        torch.save(self.checkpoint, self.folder / "checkpoint.pt")
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.audit()
        self.checkpoint["selection"]["seed"] = 11
        next(iter(self.checkpoint["state_dict"].values())).view(-1)[0] = float("nan")
        torch.save(self.checkpoint, self.folder / "checkpoint.pt")
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "nonfinite weights"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
