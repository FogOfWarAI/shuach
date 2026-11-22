# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gamematch.data.heroes import file_hash
from gamematch.data.skill import load_skill


class SkillDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "datasets/derived/dota2-reference-v2"
        self.raw = self.root / "datasets/extracted/dota2-matches-v2"
        self.reference.mkdir(parents=True)
        self.raw.mkdir(parents=True)
        self.examples = {
            "match_id": np.array([0, 1]),
            "player_slot": np.array([0, 128]),
            "account_id": np.array([10, 20]),
            "current_mu": np.array([25.0, 26.0]),
            "current_sigma": np.array([5.0, 4.0]),
            "future_mu": np.array([26.0, 24.0]),
            "historical_count": np.array([10, 30]),
            "main_observed_count": np.array([1, 2]),
            "completed_at": np.array([500.0, 700.0]),
            "target_available_at": np.array([1000.0, 1100.0]),
        }
        self.matches = {
            "match_id": np.arange(3),
            "heroes": np.tile(np.arange(1, 11).reshape(1, 2, 5), (3, 1, 1)),
            "duration": np.array([500, 700, 400]),
        }
        self.save_reference()
        rows = []
        slots = [0, 1, 2, 3, 4, 128, 129, 130, 131, 132]
        for mid in range(3):
            for t in [0, 180, 360, 540]:
                row = {"match_id": float(mid), "times": t}
                row.update(
                    {
                        f"{feature}_{slot}": t + slot
                        for slot in slots
                        for feature in ["gold_t", "lh_t", "xp_t"]
                    }
                )
                rows.append(row)
        self.table = pd.DataFrame(rows)
        self.table.to_csv(self.raw / "player_time.csv", index=False)

    def save_reference(self):
        metadata = {"status": "completed"}
        for name, values in [
            ("skill-examples.npz", self.examples),
            ("main-rosters.npz", self.matches),
        ]:
            np.savez_compressed(self.reference / name, **values)
            metadata[name + "_sha256"] = file_hash(self.reference / name)
        (self.reference / "manifest.json").write_text(json.dumps(metadata))

    def cache_path(self, data):
        return self.reference / data.manifest["snapshots"]["cache_file"]

    def test_snapshot_availability_orientation_and_cache(self):
        data, _ = load_skill(self.root)
        self.assertEqual(data.valid[0].sum(), 2)
        self.assertEqual(data.valid[1].sum(), 3)
        np.testing.assert_array_equal(data.lineups[1], self.matches["heroes"][1, ::-1])
        self.assertTrue(np.all(data.snapshots[~data.valid] == 0))
        again, _ = load_skill(self.root)
        np.testing.assert_array_equal(data.snapshots, again.snapshots)
        original_cache, original_hash = self.cache_path(data), file_hash(self.cache_path(data))
        # A changed post-end value cannot become a feature or overwrite prior evidence.
        self.table.loc[(self.table.match_id == 0) & (self.table.times == 540), "gold_t_0"] = 99999
        self.table.to_csv(self.raw / "player_time.csv", index=False)
        changed, _ = load_skill(self.root)
        np.testing.assert_array_equal(data.snapshots[0], changed.snapshots[0])
        self.assertNotEqual(original_cache, self.cache_path(changed))
        self.assertEqual(file_hash(original_cache), original_hash)
        with self.cache_path(changed).open("ab") as handle:
            handle.write(b"unexpected change")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_skill(self.root)

    def test_roster_identity_invalidates_cache_without_overwriting_prior_arrays(self):
        before, _ = load_skill(self.root)
        original_hash = file_hash(self.cache_path(before))
        self.matches["heroes"][0] += 100
        self.matches["duration"][0] = 200
        self.save_reference()
        after, _ = load_skill(self.root)
        self.assertEqual(after.valid[0].sum(), 1)
        np.testing.assert_array_equal(after.lineups[0], self.matches["heroes"][0])
        self.assertNotEqual(self.cache_path(before), self.cache_path(after))
        self.assertEqual(file_hash(self.cache_path(before)), original_hash)
        self.assertEqual(
            after.manifest["snapshots"]["identity"]["reference_rosters_sha256"],
            file_hash(self.reference / "main-rosters.npz"),
        )

    def test_snapshot_match_ids_cannot_wrap_or_truncate(self):
        for bad in [-1, 0.9, 3, np.nan]:
            with self.subTest(match_id=bad):
                table = self.table.copy()
                table.loc[(table.match_id == 0) & (table.times == 180), "match_id"] = bad
                table.to_csv(self.raw / "player_time.csv", index=False)
                with self.assertRaisesRegex(ValueError, "Snapshot match_id"):
                    load_skill(self.root)

    def test_reference_match_ids_and_player_slots_are_validated(self):
        original = {k: v.copy() for k, v in self.examples.items()}
        for field, bad in [
            ("match_id", -1),
            ("match_id", 0.9),
            ("match_id", 3),
            ("player_slot", 0.9),
            ("player_slot", 5),
            ("player_slot", 133),
        ]:
            with self.subTest(field=field, value=bad):
                self.examples = {k: v.copy() for k, v in original.items()}
                self.examples[field] = self.examples[field].astype(float)
                self.examples[field][0] = bad
                self.save_reference()
                with self.assertRaisesRegex(ValueError, "Reference"):
                    load_skill(self.root)


if __name__ == "__main__":
    unittest.main()
