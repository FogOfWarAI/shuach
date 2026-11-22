# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gamematch.data.events import MatchEvent, load_overwatch
from gamematch.evaluation.multiclass import multiclass_metrics
from gamematch.models.ratings import DEFAULT_RATING_MODELS, RATING_MODELS, OpenSkillRating
from gamematch.rating_runner import fit_calibration, replay


class TemporalTests(unittest.TestCase):
    def test_rating_ids_select_the_declared_upstream_backend(self):
        from openskill.models import BradleyTerryFull, PlackettLuce, ThurstoneMostellerFull

        self.assertEqual(DEFAULT_RATING_MODELS, ("elo", "trueskill", "openskill"))
        for name, backend in (
            ("openskill", PlackettLuce),
            ("openskill_bt_full", BradleyTerryFull),
            ("openskill_tm_full", ThurstoneMostellerFull),
        ):
            with self.subTest(name=name):
                self.assertIsInstance(RATING_MODELS[name]().environment, backend)
        with self.assertRaisesRegex(ValueError, "Unknown OpenSkill"):
            OpenSkillRating("unregistered")

    def test_event_rejects_extra_teams_and_nonfinite_feedback_delay(self):
        for teams in [(), (("a",),), (("a",), ("b",), ("c",))]:
            with self.assertRaisesRegex(ValueError, "Exactly two"):
                MatchEvent("invalid", 1, 2, teams, 2, 0)
        for delay in [np.nan, np.inf, -1]:
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                load_overwatch(Path("nonexistent.jsonl"), feedback_delay=delay)

    def test_feedback_at_or_after_prediction_is_inaccessible(self):
        teams = (("a", "b"), ("c", "d"))
        events = [
            MatchEvent("0", 10, 30, teams, 2, 0),
            MatchEvent("1", 20, 50, teams, 0, 0),
            MatchEvent("2", 30, 60, teams, 1, 0),
            MatchEvent("3", 31, 70, teams, 2, 0),
        ]
        scores, history, _ = replay(events, "elo")
        np.testing.assert_array_equal(scores[:3], [0, 0, 0])
        np.testing.assert_array_equal(history[:3, 0], [0, 0, 0])
        self.assertGreater(scores[3], 0)
        self.assertEqual(history[3, 0], 1)

    def test_domain_separation_and_global_observed_counts(self):
        teams = (("a",), ("b",))
        events = [MatchEvent("a", 1, 2, teams, 2, 0), MatchEvent("b", 3, 4, teams, 2, 1)]
        pooled, _, _ = replay(events, "elo", False)
        separate, history, _ = replay(events, "elo", True)
        self.assertGreater(pooled[1], 0)
        self.assertEqual(separate[1], 0)
        self.assertEqual(history[1, 0], 1)
        self.assertEqual(history[1, 3], 0)

    def test_rating_adapters_respond_to_win_and_preserve_orientation(self):
        teams = (("a", "b"), ("c", "d"))
        for name, factory in RATING_MODELS.items():
            with self.subTest(name=name):
                model = factory()
                self.assertAlmostEqual(model.features(teams), 0)
                model.update(teams, 2)
                self.assertGreater(model.features(teams), 0)
                self.assertAlmostEqual(
                    model.features(teams), -model.features(teams[::-1]), places=6
                )
                draw_model = factory()
                draw_model.update(teams, 1)
                self.assertAlmostEqual(draw_model.features(teams), 0, places=6)

    def test_future_labels_cannot_change_calibration(self):
        score = np.linspace(-1, 1, 100)
        domain = np.arange(100) % 2
        outcome = np.arange(100) % 3
        valid = np.arange(100) < 60
        a = fit_calibration(score, domain, outcome, valid)
        changed = outcome.copy()
        changed[~valid] = 2
        b = fit_calibration(score, domain, changed, valid)
        np.testing.assert_array_equal(a[-1].coef_, b[-1].coef_)

    def test_multiclass_retains_draw_and_rejects_invalid_simplex(self):
        result = multiclass_metrics(np.array([0, 1, 2]), np.eye(3))
        self.assertEqual(result["class_counts"], [1, 1, 1])
        self.assertEqual(result["brier_multiclass_sum"], 0)
        with self.assertRaises(ValueError):
            multiclass_metrics(np.array([1]), [[0.2, 0.5, 0.5]])

    def test_overwatch_adapter_deduplicates_and_preserves_draws(self):
        item = {
            "time": 100,
            "duration": 10,
            "competitive": True,
            "result": "DRAW",
            "teams": {"blue": list("abcdef"), "red": list("ghijkl")},
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ow.jsonl"
            path.write_text(json.dumps(item) + "\n" + json.dumps(item) + "\n")
            events, manifest = load_overwatch(path, feedback_delay=50)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].outcome, 1)
            self.assertEqual(events[0].available_at, 160)
            self.assertEqual(manifest["counts"]["exact_duplicates"], 1)


if __name__ == "__main__":
    unittest.main()
