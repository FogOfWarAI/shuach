# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import unittest

import numpy as np
import pandas as pd
import torch

from gamematch.asymmetric_runner import ordinal_metrics
from gamematch.data.asymmetric import build_predator_features
from gamematch.models.ordinal import OrdinalMLP


class AsymmetricTests(unittest.TestCase):
    def test_future_total_experience_and_current_behavior_are_excluded(self):
        frame = pd.DataFrame(
            {
                "timestamp": [
                    "2020-01-01T00:00:00Z",
                    "2020-01-01T00:00:05Z",
                    "2020-01-01T00:00:11Z",
                ],
                "match_id": ["a", "b", "c"],
                "player_id": ["p"] * 3,
                "environment_id": ["map"] * 3,
                "avatar_id": ["avatar"] * 3,
                "hunting_success": [4, 0, 2],
                "game_duration": [10, 10, 10],
                "total_xp": [3, 3, 3],
                "speed": [1, 2, 3],
            }
        )
        prepared, features = build_predator_features(frame, "player_id")
        np.testing.assert_array_equal(prepared["_time"], [1577836800, 1577836805, 1577836811])
        changed = frame.copy()
        changed["total_xp"] = 100000
        changed["speed"] = 999
        _, features2 = build_predator_features(changed, "player_id")
        pd.testing.assert_frame_equal(features, features2)
        self.assertEqual(features.log1p_completed.iloc[1], 0)
        self.assertAlmostEqual(features.log1p_completed.iloc[2], np.log(2), places=6)
        self.assertEqual(features.last_previous_kills.iloc[2], 4)
        changed.loc[1, "hunting_success"] = 4
        _, features3 = build_predator_features(changed, "player_id")
        pd.testing.assert_frame_equal(features, features3)

    def test_fractional_timestamps_timezone_and_delay_contract(self):
        frame = pd.DataFrame(
            {
                "timestamp": ["2020-01-01T08:00:00.125+08:00"],
                "match_id": ["a"],
                "player_id": ["p"],
                "environment_id": ["map"],
                "avatar_id": ["avatar"],
                "hunting_success": [2],
                "game_duration": [10.5],
            }
        )
        prepared, _ = build_predator_features(frame, "player_id", feedback_delay=2)
        self.assertEqual(prepared["_time"].iloc[0], 1577836800.125)
        self.assertEqual(prepared["_available"].iloc[0], 1577836812.625)
        for delay in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                build_predator_features(frame, "player_id", feedback_delay=delay)
        frame["timestamp"] = None
        with self.assertRaises(ValueError):
            build_predator_features(frame, "player_id")

    def test_ordered_likelihood_is_normalized_and_thresholds_monotonic(self):
        model = OrdinalMLP(3, hidden=8)
        x = torch.randn(8, 3)
        logp = model(x)
        torch.testing.assert_close(logp.exp().sum(1), torch.ones(8))
        self.assertTrue((torch.diff(model.thresholds()) > 0).all())
        loss = torch.nn.functional.nll_loss(logp, torch.arange(8) % 5)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_ranked_score_respects_ordered_distance(self):
        correct = ordinal_metrics(np.array([0]), np.array([[1, 0, 0, 0, 0]], float))
        near = ordinal_metrics(np.array([0]), np.array([[0, 1, 0, 0, 0]], float))
        far = ordinal_metrics(np.array([0]), np.array([[0, 0, 0, 0, 1]], float))
        self.assertEqual(correct["ranked_probability_score"], 0)
        self.assertLess(near["ranked_probability_score"], far["ranked_probability_score"])


if __name__ == "__main__":
    unittest.main()
