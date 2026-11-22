# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Small synthetic end-to-end fixtures exercise public workflows and reloads."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from gamematch.asymmetric_runner import run_asymmetric
from gamematch.cli import main as cli_main
from gamematch.data.asymmetric import load_asymmetric
from gamematch.data.skill import SkillData
from gamematch.decision_runner import run_decisions
from gamematch.history_runner import cached_history, run_history
from gamematch.inference import AsymmetricPredictor, HistoryPredictor, SkillPredictor
from gamematch.rating_runner import run_ratings
from gamematch.skill_runner import run_skill
from gamematch.tuning.data import load_bundle
from gamematch.tuning.search import candidates, save_data_identity
from gamematch.tuning.search import run as run_tuning


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        raw = self.root / "datasets/raw"
        raw.mkdir(parents=True)
        rng = np.random.default_rng(21)
        records = []
        for i in range(300):
            players = [f"p{n}" for n in rng.choice(40, 12, replace=False)]
            records.append(
                {
                    "time": 1600000000 + i * 20,
                    "duration": 5,
                    "competitive": bool(i % 2),
                    "result": ["LOSS", "DRAW", "WIN"][i % 3],
                    "teams": {"blue": players[:6], "red": players[6:]},
                    "compositions": None,
                }
            )
        (raw / "overwatch.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=400, freq="min", tz="UTC").astype(
                    str
                ),
                "match_id": [f"m{i}" for i in range(400)],
                "player_id": [f"p{i % 20}" for i in range(400)],
                "environment_id": [f"map{i % 3}" for i in range(400)],
                "avatar_id": [f"hero{i % 7}" for i in range(400)],
                "hunting_success": np.arange(400) % 5,
                "game_duration": np.full(400, 10.0),
                "total_xp": np.full(400, 20),
            }
        )
        frame.to_csv(raw / "dbd-2022.csv", index=False)
        frame.rename(columns={"player_id": "predator_id"}).to_csv(raw / "dbd-2023.csv", index=False)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, name, value):
        p = self.root / f"{name}.json"
        p.write_text(json.dumps(value))
        return p

    def cli(self, command, config, output):
        cli_main(
            [
                command,
                "--config",
                str(config),
                "--workspace",
                str(self.root),
                "--output",
                str(output),
            ]
        )

    def assert_completed(self, output, count, pattern):
        self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "completed")
        results = list(output.glob(pattern))
        self.assertEqual(len(results), count)
        for path in results:
            with np.load(path.with_name("predictions.npz"), allow_pickle=False) as data:
                self.assertEqual(len(np.unique(data["row_id"])), len(data["y_true"]))

    def test_ratings_complete_resume_and_detect_data_changes(self):
        cfg = self.config(
            "rating",
            {
                "protocol": "temporal_delayed_feedback",
                "models": ["elo", "trueskill", "openskill", "openskill_bt_full", "openskill_tm_full"],
                "separate_domains": [False, True],
            },
        )
        out = self.root / "rating-run"
        self.cli("outcome-benchmark", cfg, out)
        self.assert_completed(out, 11, "*/metrics.json")
        before = (out / "elo-pooled/metrics.json").stat().st_mtime_ns
        run_ratings(cfg, self.root, out)
        self.assertEqual(before, (out / "elo-pooled/metrics.json").stat().st_mtime_ns)
        source = self.root / "datasets/raw/overwatch.jsonl"
        content = source.read_text()
        source.write_text(content.replace('"LOSS"', '"WIN"', 1))
        with self.assertRaises(ValueError):
            run_ratings(cfg, self.root, out)

    def test_tuning_locks_validation_selection_and_rejects_changed_data_on_resume(self):
        bundle = load_bundle(self.root, "dbd-2022")
        reference = self.root / "runs/asymmetric-v1/dbd-2022/seed-0-logistic"
        reference.mkdir(parents=True)
        np.savez_compressed(
            reference / "predictions.npz",
            row_id=bundle.parts["test"].row_id,
            y_true=bundle.parts["test"].target,
        )
        configs = [
            {**c, "epochs": 2, "hidden_dim": 8, "device": "cpu", "batch_size": 64}
            for c in candidates("dbd-2022", "mlp")[:2]
        ]
        out = self.root / "tuning-run"
        with (
            patch("gamematch.tuning.search.MATRIX", {"dbd-2022": ["mlp"]}),
            patch("gamematch.tuning.search.candidates", return_value=configs),
        ):
            run_tuning(self.root, out)
            target = out / "dbd-2022/mlp"
            selection = json.loads((target / "selection.json").read_text())
            self.assertFalse(selection["test_used"])
            self.assertEqual(selection["selection_seeds"], [100, 101])
            for path in target.glob("search/*/*/metrics.json"):
                self.assertFalse(json.loads(path.read_text())["test_evaluated"])
                self.assertFalse(path.with_name("predictions.npz").exists())
            finals = list(target.glob("final/*/result.json"))
            self.assertEqual(len(finals), 5)
            for path in finals:
                result = json.loads(path.read_text())
                self.assertTrue(result["test_evaluated"])
                self.assertLessEqual(result["reload_max_abs_error"], 1e-6)
                self.assertLessEqual(
                    (target / "selection.json").stat().st_mtime_ns, path.stat().st_mtime_ns
                )
            before = {p: p.stat().st_mtime_ns for p in finals}
            run_tuning(self.root, out)
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in finals})
            preprocessor = out / "dbd-2022/data-world-0/preprocessor.joblib"
            serialized = preprocessor.read_bytes()
            source = self.root / "datasets/raw/dbd-2022.csv"
            frame = pd.read_csv(source)
            frame.loc[0, "total_xp"] = 21
            frame.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "Data manifest changed"):
                run_tuning(self.root, out)
            self.assertEqual(serialized, preprocessor.read_bytes())

    def test_history_all_architectures_train_save_and_calibrate(self):
        cfg = self.config(
            "history",
            {
                "protocol": "temporal_completed_player_history",
                "history_length": 3,
                "models": ["static_deepsets", "history_deepsets", "owo_transformer", "owo", "dawn"],
                "seeds": [0],
                "training": {
                    "epochs": 2,
                    "patience": 1,
                    "batch_size": 32,
                    "width": 8,
                    "heads": 2,
                    "layers": 1,
                    "dropout": 0.1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "threads": 1,
                    "device": "cpu",
                },
            },
        )
        out = self.root / "history-run"
        self.cli("outcome-benchmark", cfg, out)
        self.assert_completed(out, 5, "seed-*/metrics.json")
        run_history(cfg, self.root, out)
        for p in out.glob("seed-*/metrics.json"):
            self.assertGreater(json.loads(p.read_text())["temperature"], 0)
            raw, _ = cached_history(self.root, json.loads(cfg.read_text()))
            with np.load(out / "split-manifest.npz", allow_pickle=False) as split:
                selected = np.flatnonzero(split["test"])[:17]
            predictor = HistoryPredictor(p.parent, batch_size=7)
            actual = predictor.predict(
                raw.history[selected],
                raw.static[selected],
                raw.valid[selected],
                raw.domain[selected],
            )
            with np.load(p.with_name("predictions.npz"), allow_pickle=False) as saved:
                np.testing.assert_allclose(actual, saved["probabilities"][:17], atol=2e-6)
            reversed_rows = predictor.predict(
                raw.history[selected][::-1],
                raw.static[selected][::-1],
                raw.valid[selected][::-1],
                raw.domain[selected][::-1],
            )
            np.testing.assert_allclose(reversed_rows, actual[::-1], atol=2e-6)
            with self.assertRaises(ValueError):
                predictor.predict(
                    raw.history[selected, ..., :-1],
                    raw.static[selected],
                    raw.valid[selected],
                    raw.domain[selected],
                )

    def test_asymmetric_workflow_avoids_object_archives(self):
        models = ["prior", "history_frequency", "logistic", "mlp", "ordinal"]
        try:
            import lightgbm  # noqa: F401

            models.append("lightgbm")
        except (ImportError, OSError):
            pass
        cfg = self.config(
            "asymmetric",
            {
                "protocol": "temporal_predator_outcome",
                "datasets": ["dbd-2022", "dbd-2023"],
                "models": models,
                "seeds": [0],
                "training": {
                    "epochs": 2,
                    "patience": 1,
                    "batch_size": 64,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0001,
                    "hidden_dim": 8,
                    "max_trees": 5,
                    "threads": 1,
                },
            },
        )
        out = self.root / "asymmetric-run"
        self.cli("outcome-benchmark", cfg, out)
        self.assert_completed(out, 2 * len(models), "*/seed-*/metrics.json")
        data = load_asymmetric("dbd-2022", self.root / "datasets")
        self.assertEqual(data.row_ids.dtype.kind, "U")
        self.assertEqual(data.player_ids.dtype.kind, "U")
        run_asymmetric(cfg, self.root, out)
        for p in out.glob("*/seed-*/metrics.json"):
            raw = load_asymmetric(p.parent.parent.name, self.root / "datasets")
            with np.load(p.with_name("predictions.npz"), allow_pickle=False) as saved:
                indices = [np.flatnonzero(raw.row_ids == row)[0] for row in saved["row_id"][:17]]
                predictor = AsymmetricPredictor(p.parent, batch_size=7)
                frame = raw.features.iloc[indices]
                actual = predictor.predict(frame)
                np.testing.assert_allclose(actual, saved["probabilities"][:17], atol=2e-6)
                if predictor.name == "history_frequency":
                    with self.assertRaisesRegex(ValueError, "batch size"):
                        AsymmetricPredictor(p.parent, batch_size=True).predict(frame)
                    for value in [np.nan, -0.1, 0.7]:
                        with (
                            self.subTest(value=value),
                            self.assertRaisesRegex(ValueError, "fraction"),
                        ):
                            predictor.predict(frame.assign(previous_kill_fraction_0=value))

    def test_reloaded_calibrators_require_a_positive_finite_temperature(self):
        directory = self.root / "run/seed-0-prior"
        directory.mkdir(parents=True)
        (directory.parent / "manifest.json").write_text(
            json.dumps({"normalizer": {}, "config": {}})
        )
        for predictor in [AsymmetricPredictor, HistoryPredictor]:
            for value in [0, -1, float("inf"), float("nan"), True]:
                (directory / "metrics.json").write_text(
                    json.dumps({"model": "prior", "temperature": value})
                )
                with (
                    self.subTest(predictor=predictor, value=value),
                    self.assertRaisesRegex(ValueError, "temperature"),
                ):
                    predictor(directory)

    def test_asymmetric_tabular_outputs_preserve_unobserved_count_column(self):
        raw = self.root / "datasets/raw/dbd-2022.csv"
        frame = pd.read_csv(raw)
        frame["hunting_success"] = frame["hunting_success"].replace(2, 3)
        frame.to_csv(raw, index=False)
        config = self.config(
            "missing-count",
            {
                "protocol": "temporal_predator_outcome",
                "datasets": ["dbd-2022"],
                "models": ["logistic"],
                "seeds": [0],
                "training": {"threads": 1},
            },
        )
        out = self.root / "missing-count-run"
        run_asymmetric(config, self.root, out)
        fit = out / "dbd-2022/seed-0-logistic"
        data = load_asymmetric("dbd-2022", self.root / "datasets")
        predictor = AsymmetricPredictor(fit)
        with np.load(fit / "predictions.npz", allow_pickle=False) as saved:
            index = pd.Index(data.row_ids).get_indexer(saved["row_id"])
            probability = predictor.predict(data.features.iloc[index])
            np.testing.assert_array_equal(predictor.model.classes_, [0, 1, 3, 4])
            self.assertEqual(probability.shape, (len(index), 5))
            np.testing.assert_allclose(probability, saved["probabilities"])
            self.assertTrue((probability[:, 2] < probability.max(axis=1)).all())

    def test_synthetic_factorial_outputs_have_zero_exact_solver_gap(self):
        cfg = self.config(
            "decision",
            {
                "protocol": "synthetic_independent_decisions",
                "mechanisms": ["cooperative"],
                "population": 20,
                "split_rows": {"train": 200, "valid": 50, "test": 50},
                "lobbies": 4,
                "models": ["logistic", "fm", "mlp", "deepsets", "neuralac"],
                "seeds": [0],
                "training": {"epochs": 1, "batch_size": 64, "threads": 1},
            },
        )
        out = self.root / "runs/decisions-v1"
        self.cli("decision-benchmark", cfg, out)
        self.assert_completed(out, 5, "*/*/metrics.json")
        results = json.loads((out / "results.json").read_text())
        self.assertEqual(len(results), 45)
        for row in results:
            self.assertGreaterEqual(row["mean_decision_regret"], -1e-12)
            if row["solver"] == "exact":
                self.assertEqual(row["mean_solver_gap"], 0)
        run_decisions(cfg, self.root, out)
        # An oracle/environment change must invalidate tuning reuse even when
        # factual train/validation/test examples remain byte-for-byte identical.
        bundle = load_bundle(self.root, "synthetic-cooperative", 0)
        identity = self.root / "tuning-synthetic-identity"
        save_data_identity(identity, bundle, {})
        environment = out / "cooperative-seed-0/decision-environment.npz"
        with environment.open("ab") as stream:
            stream.write(b"changed-environment")
        changed = load_bundle(self.root, "synthetic-cooperative", 0)
        with self.assertRaisesRegex(ValueError, "Data manifest changed"):
            save_data_identity(identity, changed, {})

    def test_skill_workflow_uses_purged_targets_and_reloadable_checkpoints(self):
        rng = np.random.default_rng(12)
        n = 300
        snapshots = rng.normal(size=(n, 12, 6)).astype(np.float32)
        valid = np.ones((n, 12), dtype=bool)
        lineups = np.array(
            [rng.choice(np.arange(1, 21), 10, replace=False).reshape(2, 5) for _ in range(n)]
        )
        lineups[255:, 0, 0] = 99
        current = 25 + rng.normal(size=n)
        data = SkillData(
            snapshots,
            valid,
            lineups,
            current,
            current + snapshots[:, 0, 0] * 0.2,
            np.arange(n, dtype=float),
            np.arange(n, dtype=float) + 3,
            np.arange(n) % 20,
            np.arange(n),
            np.array([f"row{i}" for i in range(n)]),
            np.arange(n) % 3 == 0,
            {"fixture": "synthetic test only"},
        )
        metadata = {"start_time": np.arange(n), "valid": np.ones(n, dtype=bool)}
        cfg = self.config(
            "skill",
            {
                "protocol": "temporal_future_observed_rating",
                "models": ["current", "mean_delta", "ridge", "mlp", "gru", "transformer", "mmrnet"],
                "seeds": [0],
                "training": {
                    "epochs": 2,
                    "patience": 1,
                    "batch_size": 64,
                    "width": 8,
                    "heads": 2,
                    "layers": 1,
                    "dropout": 0.1,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0001,
                    "threads": 1,
                    "device": "cpu",
                },
            },
        )
        out = self.root / "skill-run"
        with patch("gamematch.skill_runner.load_skill", return_value=(data, metadata)):
            self.cli("skill-benchmark", cfg, out)
            run_skill(cfg, self.root, out)
        self.assert_completed(out, 7, "seed-*/metrics.json")
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertNotIn(99, manifest["hero_vocabulary"])
        baseline = json.loads((out / "seed-0-current/metrics.json").read_text())["metrics"]["mae"]
        ridge = json.loads((out / "seed-0-ridge/metrics.json").read_text())["metrics"]["mae"]
        self.assertLess(ridge, baseline)
        with np.load(out / "split-manifest.npz", allow_pickle=False) as split:
            selected = np.flatnonzero(split["test"])[:17]
        for p in out.glob("seed-*/predictions.npz"):
            predictor = SkillPredictor(p.parent, batch_size=7)
            actual = predictor.predict(
                data.snapshots[selected],
                data.valid[selected],
                data.lineups[selected],
                data.current_mu[selected],
            )
            with np.load(p, allow_pickle=False) as saved:
                np.testing.assert_allclose(actual, saved["prediction"][:17], atol=2e-6)
            reversed_rows = predictor.predict(
                data.snapshots[selected][::-1],
                data.valid[selected][::-1],
                data.lineups[selected][::-1],
                data.current_mu[selected][::-1],
            )
            np.testing.assert_allclose(reversed_rows, actual[::-1], atol=2e-6)
            with self.assertRaises(ValueError):
                predictor.predict(
                    data.snapshots[selected],
                    data.valid[selected],
                    data.lineups[selected].astype(float) + 0.5,
                    data.current_mu[selected],
                )


if __name__ == "__main__":
    unittest.main()
