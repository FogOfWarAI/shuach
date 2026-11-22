# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from gamematch.data.heroes import HeroVocabulary, group_ids, group_uniform, prepare, read_records
from gamematch.evaluation.binary import binary_metrics
from gamematch.matching.lobby import exact_balance, partitions
from gamematch.models import make_model
from gamematch.runner import TrainConfig, fit_one, run_benchmark


class SemanticTests(unittest.TestCase):
    def test_grouping_blocks_permutation_and_side_leakage(self):
        roster = np.arange(1, 11).reshape(1, 2, 5)
        versions = np.concatenate([roster, roster[:, :, ::-1], roster[:, ::-1, :]])
        self.assertEqual(len(set(group_ids(versions))), 1)
        self.assertEqual(len(set(group_uniform(group_ids(versions), "test"))), 1)

    def test_vocabulary_uses_training_identifiers_only(self):
        vocab = HeroVocabulary().fit(np.array([1, 5, 9]))
        np.testing.assert_array_equal(vocab.transform(np.array([0, 1, 8, 9, 99])), [0, 1, 0, 3, 0])

    def test_empty_fitted_vocabulary_maps_all_heroes_to_unknown(self):
        roster = np.arange(10).reshape(1, 2, 5)
        vocab = HeroVocabulary().fit(roster[:0])
        self.assertEqual(vocab.size, 1)
        np.testing.assert_array_equal(vocab.transform(roster), np.zeros_like(roster))

    def test_models_are_invariant_to_member_order_and_keep_batch_one(self):
        roster = torch.arange(1, 11).reshape(1, 2, 5)
        for name in ["linear", "signed_hero", "fm", "mlp", "deepsets", "set_transformer", "neuralac"]:
            model = make_model(name, 12)
            model.eval()
            with torch.no_grad():
                prediction = model(roster)
                self.assertEqual(prediction.shape, (1,))
                torch.testing.assert_close(prediction, model(roster.flip(-1)))

    def test_signed_hero_shares_effects_across_sides_and_ignores_unknown_ids(self):
        model = make_model("signed_hero", 4)
        with torch.no_grad():
            model.contribution.weight[1:, 0] = torch.tensor([0.2, 0.5, 1.0])
            model.side_bias.fill_(0.1)
        roster = torch.tensor([[[1, 2, 0, 0, 0], [3, 0, 0, 0, 0]]])
        torch.testing.assert_close(model(roster), torch.tensor([-0.2]))
        torch.testing.assert_close(model(roster.flip(1)), torch.tensor([0.4]))

    def test_neuralac_matches_explicit_pair_equations(self):
        model = make_model("neuralac", 12)
        model.eval()
        rosters = torch.arange(1, 11).reshape(1, 2, 5)
        reference = []
        with torch.no_grad():
            for side in range(2):
                team, opponent = rosters[0, side], rosters[0, 1 - side]
                score = model.individual(team).sum()
                for hero in team:
                    v = model.cooperation(hero)
                    others = model.cooperation(team[team != hero])
                    weights = (others * model.cooperation_attention(v)).sum(-1).softmax(-1)
                    score += (weights * model.cooperation_network(v * others).squeeze(-1)).sum()
                    attack = model.attack(hero)
                    defense = model.defense(opponent)
                    weights = (defense * model.competition_attention(attack)).sum(-1).softmax(-1)
                    score += (
                        weights * model.competition_network(attack * defense).squeeze(-1)
                    ).sum()
                reference.append(score)
            torch.testing.assert_close(model(rosters)[0], reference[0] - reference[1])
            torch.testing.assert_close(model(rosters), -model(rosters.flip(1)))
        with self.assertRaises(ValueError):
            model(torch.ones(1, 2, 6, dtype=torch.long))

    def test_sparse_and_neural_count_features_agree(self):
        from gamematch.models.heroes import side_counts
        from gamematch.models.tabular import hero_counts

        rosters = np.array([[[0, 0, 1, 2, 3], [0, 2, 3, 4, 5]]])
        np.testing.assert_array_equal(
            hero_counts(rosters, 6).toarray(), side_counts(torch.from_numpy(rosters), 6).numpy()
        )

    def test_binary_metric_rejects_draws_and_skill_scores(self):
        with self.assertRaises(ValueError):
            binary_metrics([0, 2], [0.2, 0.5])
        with self.assertRaises(ValueError):
            binary_metrics([0, 1], [25, 30])
        self.assertIsNone(binary_metrics([1, 1], [0.8, 0.8])["auc"])
        self.assertIn("ece_5_equal_width", binary_metrics([0, 1], [0.2, 0.8], bins=5))
        with self.assertRaises(ValueError):
            binary_metrics([0, 1], [0.2, 0.8], bins=0)

    def test_candidate_completeness_and_side_semantics(self):
        self.assertEqual(len(list(partitions(range(10), 5, True))), 126)
        self.assertEqual(len(list(partitions(range(10), 5, False))), 252)
        self.assertEqual(len(list(partitions(range(12), 6, True))), 462)
        with self.assertRaises(ValueError):
            list(partitions([1, 1, 2, 3], 2))

    def test_exact_solver_finds_balance_without_oracle_access(self):
        def scorer(candidates):
            return np.array([0.5 if sum(a) == sum(b) else 0.9 for a, b in candidates])

        result = exact_balance([1, 2, 3, 4], 2, scorer, True)
        self.assertEqual(result["predicted_imbalance"], 0)
        a, b = result["teams"]
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a) | set(b), {1, 2, 3, 4})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / "datasets/extracted/uci-dota2"
        source.mkdir(parents=True)
        rng = np.random.default_rng(12)
        rows = []
        for _ in range(600):
            heroes = rng.choice(30, 10, replace=False)
            row = np.zeros(117, dtype=int)
            row[0] = 1 if heroes[:5].mean() > heroes[5:].mean() else -1
            row[1:4] = [1, 1, 1]
            row[4 + heroes[:5]] = 1
            row[4 + heroes[5:]] = -1
            rows.append(row)
        pd.DataFrame(rows).to_csv(source / "dota2Train.csv", header=False, index=False)
        # A pilot must not even open this deliberately invalid official test file.
        (source / "dota2Test.csv").write_text("do not open during a pilot")

    def tearDown(self):
        self.temp.cleanup()

    def test_uci_adapter_rejects_fractional_values_before_integer_conversion(self):
        source = self.root / "datasets/extracted/uci-dota2/dota2Train.csv"
        original = pd.read_csv(source, header=None).astype(float)
        for column, value in [(0, 1.9), (4, 0.4), (1, np.nan)]:
            changed = original.copy()
            changed.iloc[0, column] = value
            changed.to_csv(source, header=False, index=False)
            with self.assertRaisesRegex(ValueError, "finite integers"):
                read_records(source, uci=True)

    def test_nac_adapter_rejects_labels_that_round_to_binary_in_float32(self):
        source = self.root / "nac.csv"
        row = {f"hero{i}": [i] for i in range(1, 11)}
        row["target"] = [1.00000001]
        pd.DataFrame(row).to_csv(source, index=False)
        with self.assertRaisesRegex(ValueError, "binary outcomes"):
            read_records(source)

    def test_pilot_is_disjoint_and_does_not_touch_official_test(self):
        splits, vocab, manifest = prepare("uci-dota2", self.root / "datasets", "pilot")
        self.assertEqual(len(manifest["sources"]), 1)
        groups = [set(s.groups.tolist()) for s in splits.values()]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertGreater(vocab.size, 1)
        self.assertTrue(all(len(s) for s in splits.values()))

    def test_final_keeps_official_test_and_purges_overlapping_train_groups(self):
        source = self.root / "datasets/extracted/uci-dota2"
        rows = pd.read_csv(source / "dota2Train.csv", header=None)
        rows.iloc[:10].to_csv(source / "dota2Test.csv", header=False, index=False)
        splits, _, manifest = prepare("uci-dota2", self.root / "datasets", "final")
        self.assertEqual(len(splits["test"]), 10)
        self.assertEqual(
            manifest["excluded_training_validation_rows_overlapping_official_test"], 10
        )
        self.assertFalse(set(splits["train"].groups) & set(splits["test"].groups))

    def test_training_learns_and_checkpoint_predicts(self):
        splits, vocab, _ = prepare("uci-dota2", self.root / "datasets", "pilot")
        result = fit_one(
            "linear",
            splits,
            vocab,
            TrainConfig(epochs=20, batch_size=64, learning_rate=0.03, patience=10, threads=1),
            0,
            self.root / "training",
        )
        self.assertGreater(result["metrics"]["evaluation"]["auc"], 0.8)
        self.assertTrue((self.root / "training/checkpoint.pt").exists())
        values = np.load(self.root / "training/predictions.npz")
        self.assertEqual(len(values["p_positive"]), len(splits["test"]))

    def test_runner_resume_and_configuration_guard(self):
        from gamematch.inference import HeroPredictor

        models = ["prior", "linear", "signed_hero", "logistic", "fm", "mlp",
                  "deepsets", "set_transformer", "neuralac"]
        try:
            import lightgbm  # noqa: F401

            models.append("lightgbm")
        except (ImportError, OSError):
            pass
        config = {
            "protocol": "pilot",
            "datasets": ["uci-dota2"],
            "models": models,
            "seeds": [0],
            "training": {"epochs": 1, "threads": 1, "max_trees": 5},
        }
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        output = self.root / "run"
        run_benchmark(path, self.root, output)
        self.assertTrue((output / "source-snapshot/gamematch/runner.py").exists())
        first = (output / "uci-dota2/seed-0-prior/metrics.json").stat().st_mtime_ns
        run_benchmark(path, self.root, output)
        self.assertEqual(first, (output / "uci-dota2/seed-0-prior/metrics.json").stat().st_mtime_ns)
        self.assertEqual(len(json.loads((output / "results.json").read_text())), len(models))
        splits, _, _ = prepare("uci-dota2", self.root / "datasets", "pilot")
        for model in models:
            directory = output / f"uci-dota2/seed-0-{model}"
            actual = HeroPredictor(directory, batch_size=7).predict(splits["test"].rosters[:17])
            with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_allclose(actual, saved["p_positive"][:17], atol=2e-6)
        config["seeds"] = [1]
        path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            run_benchmark(path, self.root, output)

    def test_resuming_under_different_dependency_versions_requires_new_output(self):
        from gamematch.configuration import runtime_identity

        config = {"protocol": "pilot", "datasets": ["uci-dota2"], "models": ["prior"], "seeds": [0]}
        path, output = self.root / "config.json", self.root / "run"
        path.write_text(json.dumps(config))
        run_benchmark(path, self.root, output)
        altered = runtime_identity()
        altered["dependencies"]["numpy"] = "different-test-version"
        before = (output / "manifest.json").read_bytes()
        with patch("gamematch.runner.runtime_identity", return_value=altered):
            with self.assertRaisesRegex(ValueError, "runtime"):
                run_benchmark(path, self.root, output)
        self.assertEqual((output / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
