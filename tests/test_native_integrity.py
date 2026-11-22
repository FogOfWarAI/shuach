# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Regression cases for durable native-run completion and early validation."""

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import test_semantics as fixtures

from gamematch.artifacts import completed_fit, fit_files, seal_artifacts, verify_completions
from gamematch.asymmetric_runner import run_asymmetric
from gamematch.decision_runner import run_decisions
from gamematch.history_runner import run_history
from gamematch.rating_runner import run_ratings
from gamematch.runner import TrainConfig, fit_one, run_benchmark, write_json
from gamematch.skill_runner import run_skill


class NativeIntegrityTests(unittest.TestCase):
    def test_all_native_entries_reject_damaged_fits_before_mutation(self):
        runners = (
            run_benchmark,
            run_history,
            run_ratings,
            run_asymmetric,
            run_skill,
            run_decisions,
        )
        for runner in runners:
            for damage in (
                "missing_checkpoint",
                "changed_prediction",
                "missing_metrics",
                "changed_normalizer",
            ):
                with (
                    self.subTest(runner=runner.__name__, damage=damage),
                    tempfile.TemporaryDirectory() as temp,
                ):
                    root = Path(temp)
                    output = root / "output"
                    directory = output / "fit"
                    directory.mkdir(parents=True)
                    write_json(
                        output / "manifest.json", {"fingerprint": "known", "status": "completed"}
                    )
                    for name in fit_files("prior"):
                        (directory / name).write_bytes(b"original")
                    seal_artifacts(directory, fit_files("prior"), output)
                    if damage == "missing_checkpoint":
                        (directory / "checkpoint.json").unlink()
                    elif damage == "missing_metrics":
                        (directory / "metrics.json").unlink()
                    elif damage == "changed_normalizer":
                        write_json(
                            output / "manifest.json",
                            {"fingerprint": "known", "normalizer": {"history_mean": [100.0]}},
                        )
                    else:
                        (directory / "predictions.npz").write_bytes(b"changed")
                    config = root / "config.json"
                    write_json(
                        config,
                        {
                            "protocol": "pilot",
                            "datasets": ["uci-dota2"],
                            "models": ["prior"],
                            "seeds": [0],
                        },
                    )
                    before = {
                        p: (p.read_bytes(), p.stat().st_mtime_ns)
                        for p in output.rglob("*")
                        if p.is_file()
                    }
                    with self.assertRaisesRegex(
                        ValueError, "completed artifact|artifacts changed|identity"
                    ):
                        runner(config, root, output)
                    self.assertEqual(
                        before,
                        {
                            p: (p.read_bytes(), p.stat().st_mtime_ns)
                            for p in output.rglob("*")
                            if p.is_file()
                        },
                    )

    def test_completed_fit_identity_and_escaping_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "fit"
            directory.mkdir()
            write_json(root / "manifest.json", {"fingerprint": "known"})
            (directory / "checkpoint.json").write_text("{}")
            seal_artifacts(directory, ("checkpoint.json",), root)
            write_json(
                root / "manifest.json",
                {
                    "fingerprint": "known",
                    "status": "completed",
                    "error": "previous interrupted task",
                    "updated_utc": "later",
                },
            )
            self.assertTrue(completed_fit(directory, ("checkpoint.json",), root))
            sibling = root / "other-seed"
            sibling.mkdir()
            for name in ("checkpoint.json", "completion.json"):
                (sibling / name).write_bytes((directory / name).read_bytes())
            with self.assertRaisesRegex(ValueError, "identity"):
                completed_fit(sibling, ("checkpoint.json",), root)
            with self.assertRaisesRegex(ValueError, "identity"):
                completed_fit(directory, ("other.json",), root)
            write_json(root / "manifest.json", {"fingerprint": "different"})
            with self.assertRaisesRegex(ValueError, "identity"):
                verify_completions(root)
            write_json(directory / "completion.json", [])
            with self.assertRaisesRegex(ValueError, "identity"):
                verify_completions(root)
            with self.assertRaisesRegex(ValueError, "escaping"):
                seal_artifacts(directory, ("../../outside",), root)

    def test_public_hero_resume_checks_fits_and_restarts_unsealed_work(self):
        fixture = fixtures.PipelineTests()
        fixture.setUp()
        try:
            config = fixture.root / "config.json"
            output = fixture.root / "run"
            write_json(
                config,
                {"protocol": "pilot", "datasets": ["uci-dota2"], "models": ["prior"], "seeds": [0]},
            )
            run_benchmark(config, fixture.root, output)
            directory = output / "uci-dota2/seed-0-prior"
            files = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
            with patch("gamematch.runner.fit_one", side_effect=AssertionError("Unexpected refit")):
                run_benchmark(config, fixture.root, output)
            self.assertEqual(
                files, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
            )
            (directory / "completion.json").unlink()
            with patch("gamematch.runner.fit_one", wraps=fit_one) as fitting:
                run_benchmark(config, fixture.root, output)
            self.assertEqual(fitting.call_count, 1)
            verify_completions(output)
        finally:
            fixture.tearDown()

    def test_invalid_training_and_model_overrides_fail_before_loading(self):
        invalid = (
            {"epochs": 1.5},
            {"batch_size": True},
            {"threads": 1.5},
            {"hidden_dim": 3.5},
            {"learning_rate": float("nan")},
            {"weight_decay": float("inf")},
            {"tree_learning_rate": False},
            {"logistic_c": "1.0"},
        )
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                TrainConfig(**settings).validate()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.json"
            write_json(
                config,
                {
                    "protocol": "pilot",
                    "datasets": ["uci-dota2"],
                    "models": ["prior"],
                    "seeds": [0],
                    "model_training": {"prior": {"epochs": 1.5}},
                },
            )
            with patch("gamematch.runner.prepare", side_effect=AssertionError("Data loaded")):
                with self.assertRaisesRegex(ValueError, "epochs"):
                    run_benchmark(config, root, root / "output")
            self.assertFalse((root / "output").exists())

    def test_concurrent_json_writes_and_failed_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "state.json"
            with ThreadPoolExecutor(max_workers=8) as workers:
                list(
                    workers.map(
                        lambda n: write_json(target, {"value": n, "payload": "x" * 1024}), range(64)
                    )
                )
            self.assertIn(json.loads(target.read_text())["value"], range(64))
            original = target.read_bytes()
            with patch.object(Path, "replace", side_effect=OSError("write failed")):
                with self.assertRaises(OSError):
                    write_json(target, {"value": "new"})
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(root.iterdir()), [target])
