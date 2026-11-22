# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import unittest
from dataclasses import replace

import numpy as np

from gamematch.data.events import MatchEvent
from gamematch.data.history import build_histories
from gamematch.data.rich_history import (
    RICH_HISTORY_FEATURES,
    RICH_STATIC_FEATURES,
    build_rich_histories,
)


class RichHistoryTest(unittest.TestCase):
    def events(self):
        teams = (("a", "b"), ("c", "d"))
        return [
            MatchEvent("first", 0, 10, teams, 2, 0, 10),
            MatchEvent("overlap", 5, 20, teams, 0, 1, 15),
            MatchEvent("tied", 10, 30, teams, 1, 0, 20),
            MatchEvent("later", 11, 40, teams, 2, 0, 29),
            MatchEvent("final", 50, 60, teams, 0, 0, 10),
        ]

    def test_delayed_feedback_and_original_fields(self):
        events = self.events()
        rich, base = build_rich_histories(events), build_histories(events)
        np.testing.assert_array_equal(rich.history[..., :5], base.history)
        np.testing.assert_array_equal(rich.static[..., :13], base.static)
        self.assertEqual(rich.history.shape[-1], len(RICH_HISTORY_FEATURES))
        self.assertEqual(rich.static.shape[-1], len(RICH_STATIC_FEATURES))
        seen = RICH_STATIC_FEATURES.index("has_completed_history")
        self.assertFalse(rich.static[:3, ..., seen].any())
        self.assertTrue(rich.static[3, ..., seen].all())
        elo = RICH_STATIC_FEATURES.index("elo_centered")
        self.assertAlmostEqual(float(rich.static[3, 0, 0, elo]), 16 / 400)
        self.assertAlmostEqual(float(rich.history[3, 0, 0, -1, -1]), 0.5)

    def test_current_and_future_results_cannot_change_current_inputs(self):
        events = self.events()
        original = build_rich_histories(events)
        for index in range(len(events)):
            changed = events.copy()
            changed[index] = replace(events[index], outcome=(events[index].outcome + 1) % 3)
            data = build_rich_histories(changed)
            np.testing.assert_array_equal(data.history[: index + 1], original.history[: index + 1])
            np.testing.assert_array_equal(data.static[: index + 1], original.static[: index + 1])

    def test_player_permutation_and_side_swap(self):
        events = self.events()
        original = build_rich_histories(events)
        permuted = build_rich_histories(
            [replace(e, teams=tuple(tuple(reversed(t)) for t in e.teams)) for e in events]
        )
        np.testing.assert_array_equal(permuted.static, original.static[:, :, ::-1])
        np.testing.assert_array_equal(permuted.history, original.history[:, :, ::-1])
        swapped = build_rich_histories(
            [replace(e, teams=e.teams[::-1], outcome=2 - e.outcome) for e in events]
        )
        np.testing.assert_allclose(swapped.static, original.static[:, ::-1], atol=1e-7)
        np.testing.assert_allclose(swapped.history, original.history[:, ::-1], atol=1e-7)

    def test_pair_features_and_expired_activity(self):
        events = self.events()
        events.append(
            replace(events[-1], row_id="expired", time=31 * 86400, available_at=31 * 86400 + 10)
        )
        data = build_rich_histories(events)
        for name in ["teammate_seen_fraction", "opponent_seen_fraction"]:
            self.assertTrue((data.static[3, ..., RICH_STATIC_FEATURES.index(name)] == 1).all())
        self.assertFalse(
            data.static[-1, ..., RICH_STATIC_FEATURES.index("log1p_matches_30d")].any()
        )
        self.assertTrue(
            data.static[-1, ..., RICH_STATIC_FEATURES.index("has_completed_history")].all()
        )

    def test_duplicate_ids_rejected(self):
        events = self.events()
        with self.assertRaisesRegex(ValueError, "Unique"):
            build_rich_histories([events[0], events[0]])


if __name__ == "__main__":
    unittest.main()
