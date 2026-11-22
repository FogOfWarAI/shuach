# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Contract failures, real estimator wiring and dispatch without fabricated inputs."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from gamematch.catalog import (
    HERO_MODELS,
    HISTORY_MODELS,
    MODELS,
    OUTCOME,
    check_models,
    compatibility,
    model_catalog,
)
from gamematch.cli import main as cli_main
from gamematch.models.heroes import MODEL_NAMES
from gamematch.models.history import make_history_model
from gamematch.models.outcomes import (
    HistoryOutcomeSpec,
    build_history_outcome_model,
    validate_history_batch,
)
from gamematch.outcomes import plan_outcomes, run_outcomes


class CatalogTests(unittest.TestCase):
    def test_counts_and_actual_hero_registry_agree(self):
        catalog = model_catalog()
        self.assertEqual((catalog["base_count"], catalog["variant_count"]), (28, 4))
        self.assertEqual(set(HERO_MODELS), set(MODEL_NAMES))
        self.assertEqual(len(MODELS), len({card["name"] for card in catalog["models"]}))
        outcome = {row["name"] for row in model_catalog(OUTCOME)["models"]}
        self.assertTrue({"owo", "dawn", "neuralac", "elo", "ordinal"} <= outcome)
        self.assertNotIn("mmrnet", outcome)

    def test_hero_ids_do_not_become_player_histories(self):
        report = check_models("hero-roster", ["neuralac", "owo", "dawn"])
        self.assertFalse(report["input_compatible"])
        self.assertTrue(report["models"][0]["input_compatible"])
        self.assertIn("player_history", report["models"][1]["reason"])

    def test_binary_capability_is_not_a_completed_dawn_dataset_driver(self):
        report = check_models("league-history", ["owo", "dawn"])
        self.assertTrue(report["input_compatible"])
        self.assertFalse(report["native_runner_ready"])
        self.assertEqual(report["models"][0]["execution"], "external_driver")
        self.assertEqual(report["models"][1]["execution"], "model_api_only")
        self.assertTrue(report["models"][1]["warnings"])

    def test_future_rating_cannot_be_relabelled_as_outcome(self):
        self.assertFalse(compatibility("mmrnet", "hero-roster")["input_compatible"])
        self.assertTrue(compatibility("mmrnet", "future-rating")["input_compatible"])

    def test_native_overwatch_includes_owo_and_dawn(self):
        self.assertTrue(
            check_models("overwatch-history", list(HISTORY_MODELS))["native_runner_ready"]
        )

    def test_unknown_and_duplicate_requests_fail(self):
        for profile, models in [
            ("unknown", ["owo"]),
            ("hero-roster", ["unknown"]),
            ("hero-roster", ["mlp", "mlp"]),
            ("hero-roster", []),
        ]:
            with self.subTest(profile=profile, models=models), self.assertRaises(ValueError):
                check_models(profile, models)

    def test_cli_reports_json_and_nonzero_on_incompatibility(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            cli_main(["models", "--task", OUTCOME])
        self.assertIn("dawn", {row["name"] for row in json.loads(stream.getvalue())["models"]})
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli_main(["check-models", "--profile", "hero-roster", "--models", "owo"])
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(json.loads(stream.getvalue())["input_compatible"])


class OutcomeDispatchTests(unittest.TestCase):
    def test_all_native_routes_preserve_config_and_paths(self):
        cases = [
            ("final", ["neuralac"], ["uci-dota2"], "gamematch.runner.run_benchmark"),
            (
                "temporal_completed_player_history",
                ["dawn"],
                None,
                "gamematch.history_runner.run_history",
            ),
            (
                "temporal_predator_outcome",
                ["ordinal"],
                ["dbd-2022"],
                "gamematch.asymmetric_runner.run_asymmetric",
            ),
            ("temporal_delayed_feedback", ["elo"], None, "gamematch.rating_runner.run_ratings"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for protocol, models, datasets, function in cases:
                config = {"protocol": protocol, "models": models, "seeds": [0]}
                if datasets:
                    config["datasets"] = datasets
                source = root / "config.json"
                source.write_text(json.dumps(config))
                before = source.read_bytes()
                with self.subTest(protocol=protocol), patch(function, return_value="report") as run:
                    self.assertEqual(run_outcomes(source, root, root / "run"), "report")
                    run.assert_called_once_with(
                        source.resolve(), root.resolve(), (root / "run").resolve()
                    )
                    self.assertEqual(source.read_bytes(), before)

    def test_incompatible_request_fails_before_output_or_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "config.json"
            source.write_text(
                json.dumps(
                    {
                        "protocol": "final",
                        "datasets": ["uci-dota2"],
                        "models": ["owo"],
                        "seeds": [0],
                    }
                )
            )
            with patch("gamematch.runner.run_benchmark") as run, self.assertRaises(ValueError):
                run_outcomes(source, root, root / "output")
            run.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_wrong_dataset_and_silent_class_overrides_fail(self):
        base = {"protocol": "final", "models": ["mlp"], "datasets": ["uci-dota2"], "seeds": [0]}
        for change in [
            {"datasets": ["overwatch"]},
            {"classes": 3},
            {"training": {"team_size": 6}},
            {"training": None},
            {"task": "future_observed_rating"},
            {"input_profile": "league-history"},
            {"dataset": "overwatch"},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                plan_outcomes({**base, **change})
        with self.assertRaises(ValueError):
            plan_outcomes(
                {
                    "protocol": "temporal_completed_player_history",
                    "models": ["owo"],
                    "seeds": [0],
                    "datasets": ["leagueData"],
                }
            )

    def test_dry_run_rejects_malformed_training_settings_before_data_access(self):
        config = {"protocol": "pilot", "datasets": ["uci-dota2"], "models": ["mlp"], "seeds": [0]}
        for settings in (
            {"training": {"epochs": 1.5}},
            {"model_training": []},
            {"model_training": {"mlp": None}},
            {"model_training": {"mlp": {"learning_rate": float("inf")}}},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                plan_outcomes({**config, **settings})

    def test_dry_run_does_not_need_data_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "config.json"
            source.write_text(
                json.dumps(
                    {
                        "protocol": "temporal_completed_player_history",
                        "models": ["owo", "dawn"],
                        "seeds": [0],
                        "source": "does-not-exist.jsonl",
                    }
                )
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                cli_main(["outcome-benchmark", "--config", str(source), "--dry-run"])
            self.assertTrue(json.loads(output.getvalue())["native_runner_ready"])
            self.assertEqual(list(root.iterdir()), [source])


class HistoryOutcomeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.spec = HistoryOutcomeSpec(3, 4, (5, 5), 2, ("loss", "win"), domains=1)
        self.h = torch.randn(3, 2, 5, 2, 3)
        self.s = torch.randn(3, 2, 5, 4)
        self.mask = torch.ones(3, 2, 5, 2, dtype=torch.bool)
        self.domain = torch.zeros(3, dtype=torch.long)

    def test_all_history_models_train_with_binary_and_three_class_heads(self):
        for players, labels, domains in [(5, ("loss", "win"), 1), (6, ("loss", "draw", "win"), 2)]:
            spec = HistoryOutcomeSpec(3, 4, (players, players), 2, labels, domains)
            history, static = torch.randn(3, 2, players, 2, 3), torch.randn(3, 2, players, 4)
            mask = torch.ones(3, 2, players, 2, dtype=torch.bool)
            domain = torch.arange(3) % domains
            for name in HISTORY_MODELS:
                with self.subTest(name=name, classes=len(labels)):
                    model = build_history_outcome_model(name, spec, width=8, heads=2, layers=1)
                    logits = model(history, static, mask, domain)
                    self.assertEqual(logits.shape, (3, len(labels)))
                    loss = torch.nn.functional.cross_entropy(logits, torch.arange(3) % len(labels))
                    loss.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_explicit_reference_settings_reproduce_existing_history_factory(self):
        spec = HistoryOutcomeSpec(3, 4, (6, 6), 2, ("loss", "draw", "win"), domains=2)
        config = {"width": 8, "heads": 2, "layers": 1, "dropout": 0.0}
        inputs = (
            torch.randn(2, 2, 6, 2, 3),
            torch.randn(2, 2, 6, 4),
            torch.ones(2, 2, 6, 2, dtype=torch.bool),
            torch.tensor([0, 1]),
        )
        for name in HISTORY_MODELS:
            with self.subTest(name=name):
                old = make_history_model(name, 3, 4, 6, 2, config).eval()
                new = build_history_outcome_model(name, spec, pooling="global", **config).eval()
                new.load_state_dict(old.state_dict(), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(old(*inputs), new(*inputs), atol=0, rtol=0)

    def test_member_permutation_and_empty_histories(self):
        for name in ("owo", "dawn"):
            with self.subTest(name=name):
                model = build_history_outcome_model(
                    name, self.spec, width=8, heads=2, layers=1
                ).eval()
                mask = torch.zeros_like(self.mask)
                first = model(self.h, self.s, mask, self.domain)
                second = model(self.h.flip(2), self.s.flip(2), mask.flip(2), self.domain)
                self.assertTrue(torch.isfinite(first).all())
                torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)

    def test_domain_is_required_and_must_be_in_vocabulary(self):
        with self.assertRaises(ValueError):
            build_history_outcome_model("dawn", HistoryOutcomeSpec(3, 4, (5, 5), 2, ("a", "b")))
        for domain in (None, torch.ones(3, dtype=torch.long), torch.zeros(3)):
            with self.subTest(domain=domain), self.assertRaises(ValueError):
                validate_history_batch(self.spec, self.h, self.s, self.mask, domain=domain)

    def test_unequal_teams_and_malformed_labels_rejected(self):
        for teams, labels in [
            ((1, 4), ("loss", "win")),
            ((5, 5), ("win", "win")),
            ((5, 5), ("win",)),
            ((True, True), ("a", "b")),
        ]:
            with self.subTest(teams=teams, labels=labels), self.assertRaises(ValueError):
                HistoryOutcomeSpec(3, 4, teams, 2, labels)

    def test_target_shape_type_and_class_bounds_rejected(self):
        for target in [
            torch.tensor([0, 2, 0]),
            torch.tensor([0, 1, -1]),
            torch.zeros(3),
            torch.zeros(3, 1, dtype=torch.long),
        ]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_history_batch(
                    self.spec, self.h, self.s, self.mask, domain=self.domain, target=target
                )

    def test_nonfinite_inputs_and_nonboolean_masks_rejected(self):
        for field in ("history", "static", "mask"):
            history, static, mask = self.h.clone(), self.s.clone(), self.mask.clone()
            if field == "history":
                history[0, 0, 0, 0, 0] = float("nan")
                mask[0, 0, 0, 0] = False
            elif field == "static":
                static[0, 0, 0, 0] = float("inf")
            else:
                mask = mask.float()
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_history_batch(self.spec, history, static, mask, domain=self.domain)

    def test_tied_or_future_feedback_fails_and_missing_timing_is_explicit(self):
        report = validate_history_batch(self.spec, self.h, self.s, self.mask, domain=self.domain)
        self.assertFalse(report["temporal_evidence_checked"])
        now = torch.full((3,), 100)
        for time in (100, 101):
            with self.subTest(time=time), self.assertRaises(ValueError):
                validate_history_batch(
                    self.spec,
                    self.h,
                    self.s,
                    self.mask,
                    domain=self.domain,
                    prediction_time=now,
                    history_available_at=torch.full(self.mask.shape, time),
                )
        report = validate_history_batch(
            self.spec,
            self.h,
            self.s,
            self.mask,
            domain=self.domain,
            prediction_time=now,
            history_available_at=torch.full(self.mask.shape, 99),
        )
        self.assertTrue(report["temporal_evidence_checked"])
        with self.assertRaises(ValueError):
            validate_history_batch(
                self.spec, self.h, self.s, self.mask, domain=self.domain, prediction_time=now
            )


if __name__ == "__main__":
    unittest.main()
