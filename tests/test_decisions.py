# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import unittest

import numpy as np

from gamematch.decision_runner import (
    SyntheticMechanism,
    select_candidates,
    synthetic_splits,
    transformed_probability,
)
from gamematch.evaluation.binary import binary_metrics


class DecisionTests(unittest.TestCase):
    def test_oracle_orientation_and_roster_permutation(self):
        roster = np.arange(1, 11).reshape(1, 2, 5)
        for kind in ["additive", "cooperative"]:
            oracle = SyntheticMechanism(20, 0, kind)
            p = oracle.probability(roster)
            np.testing.assert_allclose(p, 1 - oracle.probability(roster[:, ::-1]))
            np.testing.assert_allclose(p, oracle.probability(roster[:, :, ::-1]))

    def test_probability_shift_changes_decision_without_changing_auc(self):
        p = np.array([[0.3, 0.49]])
        shifted = transformed_probability(p, "offset1")
        raw = select_candidates(p, "exact", np.array([[0, 1]]))[0]
        changed = select_candidates(shifted, "exact", np.array([[0, 1]]))[0]
        self.assertEqual(raw, 1)
        self.assertEqual(changed, 0)
        self.assertEqual(
            binary_metrics([0, 1], p[0])["auc"], binary_metrics([0, 1], shifted[0])["auc"]
        )
        self.assertGreater(abs(p[0, changed] - 0.5), abs(p[0, raw] - 0.5))

    def test_temperature_changes_probabilities_but_not_balance_order(self):
        rng = np.random.default_rng(4)
        p = rng.uniform(0.01, 0.99, (100, 252))
        np.testing.assert_array_equal(
            np.abs(p - 0.5).argmin(1),
            np.abs(transformed_probability(p, "temperature2") - 0.5).argmin(1),
        )

    def test_synthetic_evaluation_rosters_are_group_disjoint(self):
        splits = synthetic_splits(
            20, {"train": 100, "valid": 20, "test": 20}, 1, SyntheticMechanism(20, 1, "cooperative")
        )
        a, b, c = (set(s.groups) for s in splits.values())
        self.assertFalse(a & b or a & c or b & c)


if __name__ == "__main__":
    unittest.main()
