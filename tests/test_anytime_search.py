# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Anytime search must never silently label an approximate result as optimal."""

import unittest

import numpy as np

from gamematch.matching.lobby import exact_balance, local_swap_balance
from gamematch.matching.search import minimize_assignments, minimize_assignments_budgeted


def linear_cost(values):
    return np.abs(np.asarray(values, dtype=float) - 0.5)


class BudgetedSearchTests(unittest.TestCase):
    def test_full_scan_matches_exact_minimizer(self):
        candidates = iter(range(100))
        exact = minimize_assignments(candidates, lambda b: np.asarray(b), linear_cost, batch_size=16)
        budgeted = minimize_assignments_budgeted(
            range(100), lambda b: np.asarray(b), linear_cost, batch_size=16
        )
        self.assertEqual(budgeted["status"], "optimal_over_enumerated_candidates")
        self.assertTrue(budgeted["exhausted"])
        self.assertEqual(budgeted["cost"], exact["cost"])
        self.assertEqual(budgeted["assignment"], exact["assignment"])

    def test_candidate_budget_stops_early_and_is_labelled(self):
        result = minimize_assignments_budgeted(
            range(1000), lambda b: np.asarray(b), linear_cost,
            batch_size=32, max_candidates=100,
        )
        self.assertEqual(result["candidate_count"], 100)
        self.assertEqual(result["status"], "anytime_budget_reached")
        self.assertFalse(result["exhausted"])
        self.assertEqual(result["assignment"], 0)

    def test_time_budget_stops_early(self):
        result = minimize_assignments_budgeted(
            range(100000), lambda b: np.asarray(b), linear_cost,
            batch_size=1, time_budget=0.01,
        )
        self.assertEqual(result["status"], "anytime_budget_reached")
        self.assertGreater(result["candidate_count"], 0)
        self.assertLess(result["candidate_count"], 100000)

    def test_invalid_budgets_rejected(self):
        for kwargs in ({"time_budget": 0}, {"max_candidates": 0}, {"batch_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                minimize_assignments_budgeted(
                    range(3), lambda b: np.asarray(b), linear_cost, **kwargs
                )

    def test_best_so_far_is_returned_when_budget_expires(self):
        # The minimum sits just past the budget; the answer must be the best
        # candidate actually evaluated, never an unscored one.
        scores = np.linspace(1.0, 0.0, 500)
        result = minimize_assignments_budgeted(
            range(500), lambda b: scores[np.asarray(b)], linear_cost,
            batch_size=16, max_candidates=64,
        )
        self.assertEqual(result["candidate_count"], 64)
        self.assertAlmostEqual(result["cost"], float(np.abs(scores[63] - 0.5)))


class LocalSwapBalanceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.skills = self.rng.normal(size=10)

    def predict(self, candidates):
        difference = np.asarray(
            [self.skills[list(a)].mean() - self.skills[list(b)].mean() for a, b in candidates]
        )
        return 1 / (1 + np.exp(-difference))

    def test_approximate_result_is_labelled_and_never_better_than_exact(self):
        exact = exact_balance(range(10), 5, self.predict, exchangeable_sides=True)
        approx = local_swap_balance(range(10), 5, self.predict, seed=3)
        self.assertEqual(approx["status"], "local_swap_converged")
        self.assertIn("approximate", approx["method"])
        self.assertGreaterEqual(
            approx["predicted_imbalance"], exact["predicted_imbalance"] - 1e-9
        )
        self.assertEqual(len(approx["teams"]), 2)
        self.assertEqual(len(approx["teams"][0]), 5)
        self.assertEqual(sorted(approx["teams"][0] + approx["teams"][1]), list(range(10)))

    def test_swap_budget_reports_its_status(self):
        approx = local_swap_balance(range(10), 5, self.predict, seed=3, max_swaps=1)
        self.assertEqual(approx["status"], "swap_budget_reached")
        self.assertEqual(approx["swaps"], 1)

    def test_scales_to_a_lobby_with_too_many_partitions_to_enumerate(self):
        skills = self.rng.normal(size=20)
        def predict(candidates):
            difference = np.asarray(
                [skills[list(a)].mean() - skills[list(b)].mean() for a, b in candidates]
            )
            return 1 / (1 + np.exp(-difference))

        approx = local_swap_balance(range(20), 10, predict, seed=3, time_budget=1.0)
        self.assertLess(approx["candidate_count"], 184756)
        self.assertEqual(len(approx["teams"][0]), 10)
        self.assertLessEqual(approx["predicted_imbalance"], 0.5)

    def test_invalid_scorer_and_lobby_rejected(self):
        with self.assertRaises(ValueError):
            local_swap_balance(range(10), 5, lambda c: np.zeros(len(c)) * np.nan, seed=0)
        with self.assertRaises(ValueError):
            local_swap_balance(range(9), 5, self.predict, seed=0)
        with self.assertRaises(ValueError):
            local_swap_balance(range(10), 5, self.predict, seed=0, time_budget=0)


if __name__ == "__main__":
    unittest.main()
