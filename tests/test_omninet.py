# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import unittest

import numpy as np
import torch

from gamematch.data.events import MatchEvent
from gamematch.data.history import build_histories
from gamematch.models.omninet import MMRNet, OmniEncoder, OwO


class OmniTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_padding_cannot_change_valid_predictions_and_empty_histories_are_finite(self):
        model = OmniEncoder(width=8, heads=2, layers=2).eval()
        x = torch.randn(2, 4, 8)
        valid = torch.tensor([[False, False, True, True], [False] * 4])
        changed = x.clone()
        changed[~valid] = 10000
        a, b = model(x, valid), model(changed, valid)
        torch.testing.assert_close(a, b)
        self.assertTrue(torch.isfinite(a).all())
        self.assertEqual(float(a[1].abs().sum()), 0)
        a.sum().backward()
        self.assertTrue(
            all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        )

    def test_owo_unordered_adaptation_is_invariant_to_member_order(self):
        history = torch.randn(2, 2, 3, 4, 5)
        static = torch.randn(2, 2, 3, 6)
        valid = torch.rand(2, 2, 3, 4) > 0.4
        for domains in (0, 2):
            model = OwO(
                5,
                6,
                team_size=3,
                history_length=4,
                width=8,
                heads=2,
                ordered_slots=False,
                domains=domains,
            ).eval()
            kwargs = {"domain": torch.tensor([0, 1])} if domains else {}
            a = model(history, static, valid, **kwargs)
            b = model(history.flip(2), static.flip(2), valid.flip(2), **kwargs)
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            self.assertTrue(torch.isfinite(a).all())
            a.sum().backward()
            for name in ["toe", "soe", "poe"]:
                self.assertIsNotNone(getattr(model, name).meta_attention.in_proj_weight.grad)

    def test_mmrnet_padding_and_scalar_output(self):
        model = MMRNet(3, 20, steps=4, width=8, heads=2, layers=2).eval()
        x = torch.randn(1, 4, 3)
        lineup = torch.arange(1, 11).reshape(1, 2, 5)
        valid = torch.tensor([[True, True, False, False]])
        pred = model(x, lineup, valid)
        changed = x.clone()
        changed[:, 2:] = 100
        torch.testing.assert_close(pred, model(changed, lineup, valid))
        self.assertEqual(pred.shape, (1,))

    def test_history_builder_uses_only_delivered_feedback(self):
        teams = (("a",), ("b",))
        events = [
            MatchEvent("0", 10, 30, teams, 2, 0),
            MatchEvent("1", 20, 40, teams, 0, 1),
            MatchEvent("2", 30, 50, teams, 1, 0),
            MatchEvent("3", 31, 60, teams, 2, 0),
        ]
        data = build_histories(events, history_length=2)
        self.assertFalse(data.valid[:3].any())
        self.assertTrue(data.valid[3, :, :, -1].all())
        self.assertEqual(data.history[3, 0, 0, -1, 0], 1)
        self.assertEqual(data.history[3, 1, 0, -1, 0], 0)
        # Changing a not-yet-available outcome cannot change the current input.
        changed = events.copy()
        changed[1] = MatchEvent("1", 20, 40, teams, 2, 1)
        after = build_histories(changed, history_length=2)
        np.testing.assert_array_equal(data.history, after.history)
        np.testing.assert_array_equal(data.static, after.static)


if __name__ == "__main__":
    unittest.main()
