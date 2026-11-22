# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gamematch.evaluation.comparison import (
    paired_cluster_bootstrap,
    paired_losses,
    per_example_loss,
)


class ComparisonTests(unittest.TestCase):
    def test_identical_predictions_have_exact_zero_interval(self):
        result = paired_cluster_bootstrap(
            np.zeros((3, 20)), np.repeat(np.arange(5), 4), replicates=200
        )
        self.assertEqual(result["percentile_interval"], [0.0, 0.0])

    def test_cluster_bootstrap_targets_row_weighted_mean(self):
        delta = np.array([[1.0, 1.0, 1.0, -1.0]])
        result = paired_cluster_bootstrap(delta, np.array(["a", "a", "a", "b"]), replicates=200)
        self.assertEqual(result["delta_first_minus_second"], 0.5)
        self.assertEqual(result["n_clusters"], 2)

    def test_binary_and_multiclass_loss_semantics_match(self):
        a = {"y_true": np.array([0, 1]), "p_positive": np.array([0.2, 0.7])}
        b = {"y_true": np.array([0, 1]), "probabilities": np.array([[0.8, 0.2], [0.3, 0.7]])}
        np.testing.assert_allclose(per_example_loss(a), per_example_loss(b))

    def test_regression_loss_preserves_fractional_targets(self):
        data = {"y_true": np.array([1.2, 3.7]), "prediction": np.array([1.5, 3.0])}
        np.testing.assert_allclose(per_example_loss(data, "absolute_error"), [0.3, 0.7])
        np.testing.assert_allclose(per_example_loss(data, "squared_error"), [0.09, 0.49])
        with self.assertRaises(ValueError):
            per_example_loss(data, "unsupported")

    def test_per_example_loss_rejects_invalid_outcomes_without_coercion(self):
        invalid = [
            ({"y_true": [0.4, 1], "p_positive": [0.2, 0.8]}, "log_loss"),
            ({"y_true": [0, 1], "p_positive": [-0.1, 1.2]}, "log_loss"),
            ({"y_true": [0, 1], "p_positive": [0.2, np.nan]}, "log_loss"),
            ({"y_true": [0, 1], "p_positive": [[0.2], [0.8]]}, "log_loss"),
            ({"y_true": [0.4, 1], "probabilities": [[0.8, 0.2], [0.2, 0.8]]}, "log_loss"),
            ({"y_true": [-1, 1], "probabilities": [[0.8, 0.2], [0.2, 0.8]]}, "log_loss"),
            ({"y_true": [0, 1], "probabilities": [[0.8, 0.8], [0.2, 0.8]]}, "log_loss"),
            ({"y_true": [0, 1], "probabilities": [[1.1, -0.1], [0.2, 0.8]]}, "log_loss"),
            ({"y_true": [0, 1], "probabilities": [0.2, 0.8]}, "log_loss"),
            ({"y_true": [0, 1], "prediction": [[0.2], [0.8]]}, "absolute_error"),
            ({"y_true": [np.nan, 1], "prediction": [0.2, 0.8]}, "absolute_error"),
            ({"y_true": [0, 1], "prediction": [0.2, np.inf]}, "squared_error"),
            ({"y_true": [], "p_positive": []}, "log_loss"),
        ]
        for data, metric in invalid:
            with self.subTest(data=data, metric=metric), self.assertRaises(ValueError):
                per_example_loss(data, metric)
        endpoints = {"y_true": np.array([0, 1]), "p_positive": np.array([0.0, 1.0])}
        self.assertTrue(np.isfinite(per_example_loss(endpoints)).all())

    def test_pairing_aligns_ids_and_rejects_label_changes_or_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "a.npz", Path(temporary) / "b.npz"
            np.savez(a, row_id=["b", "a"], y_true=[1, 0], p_positive=[0.7, 0.2])
            np.savez(b, row_id=["a", "b"], y_true=[0, 1], p_positive=[0.2, 0.7])
            delta, reference = paired_losses([a], [b])
            np.testing.assert_array_equal(delta, [[0, 0]])
            np.testing.assert_array_equal(reference["row_id"], ["a", "b"])
            np.savez(b, row_id=["a", "b"], y_true=[1, 1], p_positive=[0.2, 0.7])
            with self.assertRaisesRegex(ValueError, "align"):
                paired_losses([a], [b])
            np.savez(b, row_id=["a", "a"], y_true=[0, 1], p_positive=[0.2, 0.7])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                paired_losses([a], [b])
            with self.assertRaises(ValueError):
                paired_losses([], [])

    def test_invalid_resampling_units_and_nonfinite_differences_fail(self):
        for delta, clusters in [
            (np.zeros(3), [0, 0, 0]),
            ([np.nan, 0], [0, 1]),
            (np.zeros(3), [0, 1]),
            (np.zeros((0, 2)), [0, 1]),
            (np.zeros((2, 0)), []),
            (np.zeros(2), [[0], [1]]),
        ]:
            with self.assertRaises(ValueError):
                paired_cluster_bootstrap(delta, clusters)

    def test_pairing_rejects_cross_seed_labels_and_changed_cluster_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"{i}.npz" for i in range(4)]
            original = dict(
                row_id=["a", "b"], y_true=[0, 1], group_id=["x", "y"], p_positive=[0.2, 0.7]
            )
            for path in paths:
                np.savez(path, **original)
            # Both models agree within each seed, but the same IDs have different
            # targets across seeds. Such experiments are not a paired fixed test.
            for path in paths[2:]:
                np.savez(path, **(original | {"y_true": [1, 0]}))
            with self.assertRaisesRegex(ValueError, "y_true"):
                paired_losses(paths[::2], paths[1::2])
            np.savez(paths[1], **(original | {"group_id": ["x", "x"]}))
            with self.assertRaisesRegex(ValueError, "group_id"):
                paired_losses(paths[:1], paths[1:2])

    def test_bootstrap_rejects_noninteger_or_too_small_replicate_counts(self):
        for count in [True, 99, 200.5]:
            with self.assertRaises(ValueError):
                paired_cluster_bootstrap([1.0, 2.0], [0, 1], replicates=count)


if __name__ == "__main__":
    unittest.main()
