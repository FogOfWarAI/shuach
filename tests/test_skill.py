# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from gamematch.data.skill import skill_splits
from gamematch.models.skill import make_skill_model
from gamematch.skill_runner import regression_metrics


class SkillTests(unittest.TestCase):
    def test_future_target_is_purged_at_training_and_validation_boundary(self):
        data = SimpleNamespace(
            time=np.arange(300, dtype=float), target_available_at=np.arange(300, dtype=float) + 3
        )
        masks, bounds = skill_splits(
            data, {"start_time": np.arange(300), "valid": np.ones(300, dtype=bool)}
        )
        self.assertTrue((data.target_available_at[masks["train"]] < bounds[0]).all())
        self.assertTrue((data.target_available_at[masks["valid"]] < bounds[1]).all())
        self.assertFalse(masks["train"][int(bounds[0]) - 2])
        self.assertFalse(masks["valid"][int(bounds[1]) - 2])

    def test_snapshot_baselines_cannot_read_padded_features(self):
        config = {"width": 8, "layers": 2, "heads": 2, "dropout": 0.1}
        x = torch.randn(2, 12, 6)
        lineup = torch.randint(1, 20, (2, 2, 5))
        valid = torch.zeros(2, 12, dtype=torch.bool)
        valid[:, :5] = True
        changed = x.clone()
        changed[~valid] = 1000
        for name in ["mlp", "gru", "transformer", "mmrnet"]:
            model = make_skill_model(name, 6, 20, config).eval()
            torch.testing.assert_close(model(x, lineup, valid), model(changed, lineup, valid))

    def test_regression_has_no_false_true_skill_semantics(self):
        m = regression_metrics([20.0, 30.0], [21.0, 29.0])
        self.assertEqual(m["mae"], 1.0)
        self.assertEqual(m["rmse"], 1.0)
        self.assertEqual(m["mean_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
