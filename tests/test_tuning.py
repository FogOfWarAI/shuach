# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gamematch.evaluation.binary import binary_metrics
from gamematch.evaluation.multiclass import multiclass_metrics
from gamematch.models.heroes import make_model
from gamematch.models.history import make_history_model
from gamematch.tuning.data import partition
from gamematch.tuning.engine import fit, predict_raw, reload_model
from gamematch.tuning.search import candidates, select_trial


class TuningTests(unittest.TestCase):
    def test_probability_errors_have_explicit_binary_and_categorical_scaling(self):
        binary = binary_metrics([0, 1], [0.2, 0.7])
        self.assertAlmostEqual(binary["probability_mae"], 0.25)
        self.assertAlmostEqual(binary["probability_rmse"] ** 2, 0.065)
        self.assertAlmostEqual(binary["probability_rmse"] ** 2, binary["brier"])
        categorical = multiclass_metrics([0, 2], [[0.5, 0.2, 0.3], [0.1, 0.2, 0.7]])
        self.assertAlmostEqual(categorical["probability_mae"], 1.6 / 6)
        self.assertAlmostEqual(categorical["probability_rmse"] ** 2, 0.52 / 6)
        self.assertAlmostEqual(
            categorical["probability_rmse"] ** 2, categorical["brier_multiclass_sum"] / 3
        )
        perfect = multiclass_metrics([0, 1, 2], np.eye(3))
        for key in ["log_loss", "probability_rmse", "probability_mae"]:
            self.assertEqual(perfect[key], 0.0)

    def test_team_readout_is_member_invariant_and_assignment_sensitive(self):
        torch.manual_seed(19)
        config = {
            "width": 8,
            "heads": 2,
            "layers": 1,
            "dropout": 0.0,
            "pooling": "team",
            "norm_first": True,
        }
        history = torch.randn(2, 2, 3, 4, 5)
        static = torch.randn(2, 2, 3, 7)
        valid = torch.ones(2, 2, 3, 4, dtype=torch.bool)
        valid[0, 0, 0] = False
        domain = torch.tensor([0, 1])
        for name in ["owo", "dawn", "owo_transformer"]:
            model = make_history_model(name, 5, 7, 3, 4, config).eval()
            with torch.no_grad():
                expected = model(history, static, valid, domain=domain)
                order = [2, 0, 1]
                permuted = model(
                    history[:, :, order], static[:, :, order], valid[:, :, order], domain=domain
                )
                swapped = model(history.flip(1), static.flip(1), valid.flip(1), domain=domain)
            torch.testing.assert_close(expected, permuted, atol=2e-6, rtol=1e-5)
            self.assertFalse(torch.allclose(expected, swapped, atol=1e-6))

    def test_selection_uses_validation_and_requires_complete_seeds(self):
        def item(seed, score):
            return {
                "seed": seed,
                "validation_primary": score,
                "test_evaluated": False,
                "unrelated_test_score": 1 - score,
            }

        records = {
            0: [item(100, 0.7), item(101, 0.3)],
            1: [item(100, 0.2)],
            2: [item(100, 0.45), item(101, 0.45)],
        }
        self.assertEqual(select_trial(records, [100, 101]), 2)
        records[2][0]["test_evaluated"] = True
        with self.assertRaises(ValueError):
            select_trial(records, [100, 101])

    def test_dropout_changes_training_but_preserves_checkpoint_keys_and_eval(self):
        x = torch.randint(0, 20, (32, 2, 5))
        for name in ["mlp", "deepsets"]:
            plain = make_model(name, 20, 8, 16, dropout=0)
            regularized = make_model(name, 20, 8, 16, dropout=0.5)
            regularized.load_state_dict(plain.state_dict())
            plain.eval()
            regularized.eval()
            torch.testing.assert_close(plain(x), regularized(x), rtol=0, atol=0)
            regularized.train()
            self.assertFalse(torch.equal(regularized(x), regularized(x)))

    def test_validation_only_fit_and_reload(self):
        rng = np.random.default_rng(4)

        def data(n):
            return partition(
                [rng.integers(0, 20, (n, 2, 5), dtype=np.int64)],
                rng.integers(0, 2, n).astype(np.float32),
                np.arange(n),
                {},
            )

        train, valid = data(40), data(20)
        cfg = {**candidates("uci-dota2", "mlp")[2], "device": "cpu", "epochs": 2, "batch_size": 16}
        with tempfile.TemporaryDirectory() as folder:
            model, result = fit("binary", "mlp", {"vocab_size": 20}, train, valid, cfg, 100, folder)
            self.assertFalse(result["test_evaluated"])
            self.assertNotIn("metrics", result)
            self.assertFalse((Path(folder) / "predictions.npz").exists())
            restored = reload_model(folder, "binary", "mlp", {"vocab_size": 20}, cfg)
            np.testing.assert_array_equal(
                predict_raw(model, valid, "binary", cfg),
                predict_raw(restored, valid, "binary", cfg),
            )


if __name__ == "__main__":
    unittest.main()
