# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import unittest

import numpy as np

from gamematch.matching.constraints import constrained_assignments
from gamematch.matching.lobby import exact_balance
from gamematch.matching.search import minimize_assignments


class StreamingSearchTests(unittest.TestCase):
    def test_batch_sizes_preserve_global_solution_and_first_tie(self):
        expected = None
        for size in [1, 3, 100]:
            batches = []

            def score(batch):
                batches.append(len(batch))
                return np.asarray([abs(sum(a.teams[0]) - sum(a.teams[1])) for a in batch])

            result = minimize_assignments(
                constrained_assignments(range(6), (3, 3)), score, lambda x: x, size
            )
            self.assertEqual(result["candidate_count"], 20)
            self.assertLessEqual(max(batches), size)
            if expected is None:
                expected = result["assignment"]
            self.assertEqual(result["assignment"], expected)
            self.assertEqual(result["cost"], 1)

    def test_asymmetric_objective_consumes_categorical_predictions(self):
        # Distinct 1v4 semantics: target expected predator count is two.
        candidates = list(constrained_assignments(range(5), (1, 4), parties=[(0, 1)]))
        result = minimize_assignments(
            iter(candidates),
            lambda batch: np.eye(5)[[a.teams[0][0] for a in batch]],
            lambda probabilities: np.abs(probabilities @ np.arange(5) - 2),
            batch_size=2,
        )
        self.assertEqual(result["assignment"].teams[0], (2,))
        self.assertEqual(result["cost"], 0)
        self.assertEqual(result["candidate_count"], 3)

    def test_infeasible_and_invalid_objectives_fail_explicitly(self):
        for candidates, objective, batch_size in [
            ([], lambda x: x, 1),
            ([1], lambda x: [np.nan], 1),
            ([1], lambda x: [1, 2], 1),
            ([1], lambda x: np.asarray([1 + 2j]), 1),
            ([1], lambda x: x, 0),
            ([1], lambda x: x, True),
        ]:
            with self.assertRaises(ValueError):
                minimize_assignments(iter(candidates), lambda x: x, objective, batch_size)

    def test_none_valued_candidate_is_returned_when_it_wins(self):
        result = minimize_assignments([None, "other"], lambda batch: [0, 1], lambda scores: scores)
        self.assertIsNone(result["assignment"])
        self.assertEqual(result["candidate_index"], 0)
        self.assertEqual(result["candidate_count"], 2)

    def test_binary_scorer_must_return_real_probabilities(self):
        with self.assertRaisesRegex(ValueError, "real win probabilities"):
            exact_balance(range(2), 1, lambda pairs: np.full(len(pairs), 0.5 + 2j))


if __name__ == "__main__":
    unittest.main()
