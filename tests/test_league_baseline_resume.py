# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Locked searches fail without rewriting evidence; unfinished fits can restart."""

import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPEC = importlib.util.spec_from_file_location(
    "league_baseline_resume_driver",
    Path(__file__).resolve().parents[1] / "scripts/run_league_baselines.py",
)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class StopBeforeTraining(Exception):
    pass


class NextModelCell(Exception):
    pass


def snapshot(directory):
    return {
        str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


class LeagueBaselineResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gamematch-baseline-resume-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output, self.data = self.root / "run", self.root / "data"
        self.data.mkdir()
        self.rows = {"train": np.arange(24), "valid": np.arange(24, 32), "test": np.arange(32, 40)}
        for name, indices in self.rows.items():
            np.save(self.data / f"{name}-rows.npy", indices)
        DRIVER.write_json(self.data / "manifest.json", {"fixture": True})
        self.store = SimpleNamespace(
            row_id=np.arange(40), target=np.arange(40) % 2, time=np.arange(40) * 86400
        )
        self.features = np.random.default_rng(508).normal(size=(40, 52)).astype(np.float32)
        # Freeze the real driver's identity, stopping before any fitting loop.
        with patch.object(DRIVER, "validate_locked_evidence", side_effect=StopBeforeTraining):
            with self.assertRaises(StopBeforeTraining):
                self.run_main()
        self.protocol_hash = DRIVER.file_hash(self.output / "protocol.json")
        self.folder = self.output / "outcome_history/logistic"
        self.folder.mkdir(parents=True)
        self.records = []
        for index, c in enumerate([0.01, 0.1, 1.0]):
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=20261114)
            ).fit(self.features[self.rows["train"]], self.store.target[self.rows["train"]])
            trial = self.folder / f"trial-{index}"
            trial.mkdir()
            identity = {
                "variant": "outcome_history",
                "model": "logistic",
                "trial": index,
                "configuration": {"C": c},
                "seed": 20261114,
                "protocol_sha256": self.protocol_hash,
                "features": 52,
            }
            self.records.append(self.save_candidate(trial, identity, model))
        self.choice = min(self.records, key=lambda r: (r["metrics"]["log_loss"], r["trial"]))
        DRIVER.write_json(self.folder / "selection.json", self.choice)
        model = joblib.load(self.folder / f"trial-{self.choice['trial']}/checkpoint.joblib")
        test = self.rows["test"]
        probability = DRIVER.positive_probability(model, self.features[test])
        np.savez_compressed(
            self.folder / "predictions.npz",
            row_id=self.store.row_id[test],
            y_true=self.store.target[test],
            time=self.store.time[test],
            p_positive=probability,
        )
        identity = {
            k: self.choice[k]
            for k in ["variant", "model", "trial", "configuration", "seed", "protocol_sha256"]
        }
        identity["selection_sha256"] = DRIVER.file_hash(self.folder / "selection.json")
        result = {
            **identity,
            "test_evaluated": True,
            "reload_max_abs_error": 0.0,
            "metrics": DRIVER.binary_metrics(self.store.target[test], probability),
            "predicted_positive_fraction": float((probability >= 0.5).mean()),
            "predictions_sha256": DRIVER.file_hash(self.folder / "predictions.npz"),
        }
        DRIVER.audit_final(
            self.folder,
            identity,
            self.choice,
            self.features[test],
            self.store.row_id[test],
            self.store.target[test],
            self.store.time[test],
            result,
        )
        DRIVER.write_json(self.folder / "result.json", result)

    def run_main(self):
        # Stop at the next model cell: these recovery tests need no LightGBM install.
        optional = SimpleNamespace(LGBMClassifier=Mock(side_effect=NextModelCell))
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.dict(sys.modules, {"lightgbm": optional}),
            patch.object(DRIVER, "OUT", self.output),
            patch.object(DRIVER, "DATA", self.data),
            patch.object(DRIVER, "IndexedHistoryStore", return_value=self.store),
            patch.object(
                DRIVER, "aggregate", side_effect=lambda _, rows, columns: self.features[rows]
            ),
        ):
            DRIVER.main()

    def save_candidate(self, trial, identity, model):
        valid = self.rows["valid"]
        probability = DRIVER.positive_probability(model, self.features[valid])
        joblib.dump(model, trial / "checkpoint.joblib")
        np.savez_compressed(
            trial / "validation.npz",
            row_id=self.store.row_id[valid],
            y_true=self.store.target[valid],
            p_positive=probability,
        )
        record = {
            **identity,
            "test_evaluated": False,
            "iterations": DRIVER.require_logistic_convergence(model[-1])
            if identity["model"] == "logistic"
            else int(model.best_iteration_),
            "metrics": DRIVER.binary_metrics(self.store.target[valid], probability),
            "checkpoint_sha256": DRIVER.file_hash(trial / "checkpoint.joblib"),
            "validation_sha256": DRIVER.file_hash(trial / "validation.npz"),
        }
        DRIVER.audit_candidate(
            trial,
            identity,
            self.features[valid],
            self.store.row_id[valid],
            self.store.target[valid],
            record,
        )
        DRIVER.write_json(trial / "validation.json", record)
        return record

    def assert_rejected_without_write(self, pattern):
        before = snapshot(self.output)
        with patch.object(
            LogisticRegression, "fit", side_effect=AssertionError("Unexpected refit")
        ):
            with self.assertRaisesRegex(ValueError, pattern):
                self.run_main()
        self.assertEqual(snapshot(self.output), before)
        self.assertFalse(DRIVER.RUN_IDENTITY_VALIDATED)

    def test_missing_locked_candidate_never_retrains_or_rewrites(self):
        trial = self.folder / f"trial-{(self.choice['trial'] + 1) % 3}"
        for phase in ["completed", "selected"]:
            if phase == "selected":
                (self.folder / "result.json").unlink()
            for name in ["validation.json", "checkpoint.joblib", "validation.npz"]:
                with self.subTest(phase=phase, missing=name):
                    path = trial / name
                    contents = path.read_bytes()
                    path.unlink()
                    self.assert_rejected_without_write("Locked candidate evidence is incomplete")
                    path.write_bytes(contents)

    def test_result_cannot_reconstruct_a_missing_selection(self):
        (self.folder / "selection.json").unlink()
        self.assert_rejected_without_write("lost its locked selection")
        self.assertFalse((self.folder / "selection.json").exists())

    def test_changed_locked_choice_or_artifact_fails_before_writes(self):
        paths = [
            self.folder / "trial-0/checkpoint.joblib",
            self.folder / "trial-0/validation.json",
            self.folder / "selection.json",
            self.folder / "result.json",
        ]
        for path in paths:
            with self.subTest(changed=path.name):
                contents = path.read_bytes()
                if path.suffix == ".joblib":
                    path.write_bytes(contents + b"changed")
                else:
                    record = DRIVER.read(path)
                    if path.name == "validation.json":
                        record["metrics"]["log_loss"] += 0.01
                    elif path.name == "selection.json":
                        record = self.records[(self.choice["trial"] + 1) % 3]
                    else:
                        record["selection_sha256"] = "0" * 64
                    DRIVER.write_json(path, record)
                self.assert_rejected_without_write("changed|identity")
                path.write_bytes(contents)

    def test_completed_resume_reuses_locked_artifacts(self):
        before = snapshot(self.folder)
        with patch.object(
            LogisticRegression, "fit", side_effect=AssertionError("Unexpected refit")
        ):
            with self.assertRaises(NextModelCell):
                self.run_main()
        self.assertEqual(snapshot(self.folder), before)

    def test_unselected_interruption_reuses_valid_trial_and_restarts_partial_fit(self):
        for name in ["selection.json", "result.json", "predictions.npz"]:
            (self.folder / name).unlink()
        reused = snapshot(self.folder / "trial-0")
        (self.folder / "trial-1/validation.json").unlink()
        (self.folder / "trial-1/checkpoint.joblib").write_bytes(b"interrupted save")
        shutil.rmtree(self.folder / "trial-2")
        calls, fit = [], LogisticRegression.fit

        def count_fit(model, *args, **kwargs):
            calls.append(model.C)
            return fit(model, *args, **kwargs)

        with patch.object(LogisticRegression, "fit", count_fit):
            with self.assertRaises(NextModelCell):
                self.run_main()
        self.assertEqual(calls, [0.1, 1.0])
        self.assertEqual(snapshot(self.folder / "trial-0"), reused)
        self.assertEqual(DRIVER.read(self.folder / "selection.json"), self.choice)
        self.assertTrue(DRIVER.read(self.folder / "result.json")["test_evaluated"])

    def test_self_consistent_logistic_checkpoint_must_match_estimator_settings(self):
        for setting, value in [("C", 1.0), ("random_state", 99), ("max_iter", 1000)]:
            with self.subTest(setting=setting):
                trial = self.root / setting
                shutil.copytree(self.folder / "trial-0", trial)
                record = DRIVER.read(trial / "validation.json")
                model = joblib.load(trial / "checkpoint.joblib")
                setattr(model[-1], setting, value)
                joblib.dump(model, trial / "checkpoint.joblib")
                record["checkpoint_sha256"] = DRIVER.file_hash(trial / "checkpoint.joblib")
                valid = self.rows["valid"]
                with self.assertRaisesRegex(ValueError, "Logistic parameters"):
                    DRIVER.audit_candidate(
                        trial,
                        {k: record[k] for k in ["model", "configuration", "seed"]},
                        self.features[valid],
                        self.store.row_id[valid],
                        self.store.target[valid],
                        record,
                    )

    @unittest.skipUnless(importlib.util.find_spec("lightgbm"), "optional LightGBM unavailable")
    def test_lightgbm_candidate_audit_remains_compatible(self):
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=8,
            num_leaves=3,
            min_child_samples=2,
            n_jobs=1,
            random_state=20261114,
            verbosity=-1,
        )
        train, valid = self.rows["train"], self.rows["valid"]
        model.fit(
            self.features[train],
            self.store.target[train],
            eval_set=[(self.features[valid], self.store.target[valid])],
            callbacks=[lgb.early_stopping(2, verbose=False)],
        )
        trial = self.root / "lightgbm"
        trial.mkdir()
        identity = {
            **{
                key: self.records[0][key]
                for key in ["variant", "trial", "seed", "protocol_sha256", "features"]
            },
            "model": "lightgbm",
            "configuration": {"num_leaves": 3},
        }
        record = self.save_candidate(trial, identity, model)
        DRIVER.audit_candidate(
            trial,
            {k: identity[k] for k in ["model", "configuration", "seed"]},
            self.features[valid],
            self.store.row_id[valid],
            self.store.target[valid],
        )
        self.assertEqual(record["iterations"], model.best_iteration_)


if __name__ == "__main__":
    unittest.main()
