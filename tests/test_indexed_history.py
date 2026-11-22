# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gamematch.data.indexed_history import IndexedHistoryStore, prior_observation_indices


class IndexedHistoryTest(unittest.TestCase):
    def test_strict_boundary_overlap_and_right_alignment(self):
        players = [0, 0, 0, 0, 1]
        starts = [0, 5, 10, 21, 50]
        ends = [20, 10, 15, 30, 60]
        indices, counts, latest = prior_observation_indices(players, starts, ends, length=2)
        np.testing.assert_array_equal(indices, [[0, 0], [0, 0], [0, 0], [3, 1], [0, 0]])
        np.testing.assert_array_equal(counts, [0, 0, 0, 3, 0])
        self.assertEqual(latest[3], 20)

    def test_matches_independent_brute_force(self):
        rng = np.random.default_rng(710)
        for _ in range(20):
            players = rng.integers(0, 10, 100)
            starts = rng.integers(0, 70, 100)
            ends = starts + rng.integers(0, 25, 100)
            indices, counts, _ = prior_observation_indices(players, starts, ends, 7)
            for i in range(100):
                prior = [j for j in range(100) if players[j] == players[i] and ends[j] < starts[i]]
                prior.sort(key=lambda j: (ends[j], j))
                expected = [0] * (7 - min(7, len(prior))) + [j + 1 for j in prior[-7:]]
                np.testing.assert_array_equal(indices[i], expected)
                self.assertEqual(counts[i], len(prior))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            prior_observation_indices([1], [3], [2])
        with self.assertRaises(ValueError):
            prior_observation_indices([], [], [])
        with self.assertRaises(ValueError):
            prior_observation_indices([1], [1], [2], length=0)


class IndexedStoreTest(unittest.TestCase):
    def make_store(self, path, future=False):
        arrays = {
            "performance": np.array([[0, 0], [0.2, 0.8], [0.4, 0.6]], dtype=np.float32),
            "history-index": np.array(
                [[[[0, 0]], [[0, 0]]], [[[0, 1]], [[0, 2]]], [[[0, 1]], [[0, 2]]]], dtype=np.int32
            ),
            "appearance-available": np.array([0.0, 3.0 if future else 0.5, 0.5]),
            "match-start": np.array([0.0, 1.0, 2.0]),
            "static": np.zeros((3, 2, 1, 1), dtype=np.float32),
            "target": np.array([0, 1, 0]),
            "row-id": np.array(["a", "b", "c"]),
            "train-rows": np.array([0]),
            "valid-rows": np.array([1]),
            "test-rows": np.array([2]),
        }
        for name, value in arrays.items():
            np.save(path / (name + ".npy"), value, allow_pickle=False)
        manifest = {
            "team_size": 1,
            "history_length": 2,
            "files_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.npy")
            },
            "normalizer": {
                "history_mean": [0, 0, 0],
                "history_scale": [1, 1, 1],
                "static_mean": [0],
                "static_scale": [1],
            },
        }
        (path / "manifest.json").write_text(json.dumps(manifest))
        return IndexedHistoryStore(path)

    def test_batch_expansion_padding_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(Path(tmp))
            (history, static, valid, domain), target = store.batch([2, 0, 1])
            self.assertEqual(history.shape, (3, 2, 1, 2, 3))
            np.testing.assert_array_equal(target, [0, 0, 1])
            self.assertFalse(history[1].any())
            np.testing.assert_array_equal(
                history[0, 0, 0, -1, :2], np.array([0.2, 0.8], dtype=np.float32)
            )
            self.assertAlmostEqual(float(history[0, 0, 0, -1, 2]), np.log1p(1.5 / 86400))
            self.assertEqual(store.batch([1], [0, 2])[0][0].shape[-1], 2)
            self.assertEqual(static.shape, (3, 2, 1, 1))
            self.assertFalse(domain.any())
            self.assertEqual(int(valid.sum()), 4)

    def test_future_reference_rejected_at_batch_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(Path(tmp), future=True)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                store.batch([1])

    def test_selected_columns_preserve_numpy_indexing_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(Path(tmp))
            rows = [2, 0, 1, 2]
            # Include a nontrivial normalizer and the no-history row so that
            # selecting the age feature still preserves scaling and padding.
            store.mean[:] = [0.3, -0.2, 1.5]
            store.scale[:] = [2.0, 0.5, 3.0]
            full, target = store.batch(rows)
            for columns in (
                [0, 2],
                [2, 0, 2, -2],
                [0, 1, 2],
                np.array([], dtype=int),
                np.array([True, False, True]),
                slice(None, None, -1),
                2,
                (0, 2),
                [[0, 2], [2, 1]],
            ):
                with self.subTest(columns=columns):
                    selected, actual_target = store.batch(rows, columns)
                    np.testing.assert_array_equal(selected[0], full[0][..., columns])
                    for actual, expected in zip(selected[1:], full[1:]):
                        np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(actual_target, target)

    def test_checksum_and_invalid_indices_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            store = self.make_store(path)
            for rows in [[-1], [3], [0.5]]:
                with self.assertRaises(ValueError):
                    store.batch(rows)
            np.save(path / "target.npy", np.array([1, 1, 1]))
            with self.assertRaisesRegex(ValueError, "checksum"):
                IndexedHistoryStore(path)


if __name__ == "__main__":
    unittest.main()
