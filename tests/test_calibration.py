# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import unittest

import numpy as np
from scipy.special import expit

from gamematch.evaluation.calibration import apply_binary_calibration, fit_binary_calibration


class CalibrationTests(unittest.TestCase):
    def test_recovery_ranking_and_saved_application(self):
        rng = np.random.default_rng(42)
        z = rng.normal(size=20000)
        y = rng.binomial(1, expit(1.8 * z + 0.7))
        fitted = fit_binary_calibration(z, y, "affine")
        self.assertAlmostEqual(fitted["slope"], 1.8, delta=0.1)
        self.assertAlmostEqual(fitted["intercept"], 0.7, delta=0.1)
        self.assertLess(fitted["validation_nll_after"], fitted["validation_nll_before"])
        query = np.linspace(-5, 5, 101)
        p = apply_binary_calibration(query, fitted)
        self.assertTrue((np.diff(p) > 0).all())
        np.testing.assert_allclose(p, expit(fitted["slope"] * query + fitted["intercept"]))
        self.assertGreater(p[50], 0.5)

    def test_temperature_preserves_boundary_and_extreme_logits(self):
        z = np.array([-1000.0, -2, -1, 0, 1, 2, 1000])
        y = np.array([0, 1, 0, 1, 1, 0, 1])
        fitted = fit_binary_calibration(z, y)
        self.assertEqual(fitted["intercept"], 0)
        p = apply_binary_calibration(z, fitted)
        np.testing.assert_array_equal(p >= 0.5, z >= 0)
        self.assertTrue(np.isfinite(p).all())

    def test_invalid_labels_logits_parameters_and_modes_rejected(self):
        for z, y, mode in [
            ([], [], "temperature"),
            ([0, np.nan], [0, 1], "temperature"),
            ([0, 1], [1, 1], "temperature"),
            ([0, 1], [0, 2], "temperature"),
            ([0, 1], [0, 1], "unknown"),
        ]:
            with self.assertRaises(ValueError):
                fit_binary_calibration(z, y, mode)
        for slope, intercept in [(0, 0), (-1, 0), (1, np.nan)]:
            with self.assertRaises(ValueError):
                apply_binary_calibration([0], {"slope": slope, "intercept": intercept})


if __name__ == "__main__":
    unittest.main()
