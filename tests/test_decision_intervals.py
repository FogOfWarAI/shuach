# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Decision-regret intervals are reproducible and never fabricated from one world."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gamematch.evaluation.decision_intervals import build_decision_confidence

VARIANTS = ("raw", "temperature2", "offset1")
SOLVERS = ("random", "sampled8", "exact")


def write_world(root, seed, *, lobbies=24):
    rng = np.random.default_rng(seed)
    directory = root / f"additive-seed-{seed}" / "logistic"
    directory.mkdir(parents=True)
    arrays = {}
    for variant in VARIANTS:
        for solver in SOLVERS:
            base = {"random": 0.20, "sampled8": 0.12, "exact": 0.04}[solver]
            arrays[f"{variant}_{solver}_regret"] = np.abs(
                rng.normal(base, 0.01, size=lobbies)
            )
    np.savez_compressed(directory / "decisions.npz", **arrays)


class DecisionIntervalTests(unittest.TestCase):
    def test_intervals_and_contrasts_from_paired_worlds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (0, 1, 2):
                write_world(root, seed)
            payload = build_decision_confidence(root, replicates=200, seed=7)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(len(payload["single_group_intervals"]), 9)
            for row in payload["single_group_intervals"]:
                self.assertEqual(row["status"], "estimated")
                self.assertEqual(row["n_seed_pairs"], 3)
                self.assertEqual(row["n_examples"], 24)
                self.assertEqual(len(row["percentile_interval"]), 2)
                self.assertLess(row["percentile_interval"][0], row["percentile_interval"][1])
                self.assertTrue(row["single_group_interval"])
            self.assertEqual(len(payload["solver_contrasts"]), 3)
            for contrast in payload["solver_contrasts"]:
                self.assertEqual(contrast["comparison"], "exact_minus_random")
                self.assertLess(contrast["delta_first_minus_second"], 0)
            json.dumps(payload, allow_nan=False)

    def test_single_seed_does_not_produce_a_fabricated_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_world(root, 0)
            payload = build_decision_confidence(root, replicates=200)
            self.assertTrue(payload["single_group_intervals"])
            for row in payload["single_group_intervals"]:
                self.assertEqual(row["status"], "insufficient_or_unaligned_seed_replicates")
                self.assertNotIn("percentile_interval", row)
            self.assertEqual(payload["solver_contrasts"], [])

    def test_empty_run_is_explicitly_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = build_decision_confidence(Path(temporary), replicates=200)
            self.assertEqual(payload["single_group_intervals"], [])
            self.assertEqual(payload["solver_contrasts"], [])

    def test_invalid_replicates_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (0, 1):
                write_world(root, seed)
            with self.assertRaises(ValueError):
                build_decision_confidence(root, replicates=10)


if __name__ == "__main__":
    unittest.main()
