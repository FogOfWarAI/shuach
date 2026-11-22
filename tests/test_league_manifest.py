# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Malformed shard lists must fail before public candidate data are parsed."""

import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "league_auditor", Path(__file__).resolve().parents[1] / "scripts/audit_league_data.py"
)
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


class LeagueManifestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gamematch-manifest-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        directory = self.root / "datasets/raw/leagueData"
        directory.mkdir(parents=True)
        self.manifest = {
            "status": "downloaded",
            "source": AUDITOR.SOURCE,
            "source_revision": AUDITOR.SOURCE_REVISION,
            "parquet_revision": AUDITOR.PARQUET_REVISION,
            "files": [],
        }
        self.official = []
        for index in range(12):
            name = f"{index:04d}.parquet"
            content = f"Small non-Parquet fixture {index}".encode()
            (directory / name).write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            self.manifest["files"].append(
                {"file": f"datasets/raw/leagueData/{name}", "bytes": len(content), "sha256": digest}
            )
            self.official.append(
                {
                    "path": f"default/train/{name}",
                    "type": "file",
                    "size": len(content),
                    "lfs": {"size": len(content), "oid": digest},
                }
            )

    def validate(self, manifest=None, official=None):
        return AUDITOR.validate_manifest(
            self.manifest if manifest is None else manifest,
            self.official if official is None else official,
            self.root,
        )

    def test_expected_unique_shards_pass_without_parquet_parsing(self):
        paths = self.validate()
        self.assertEqual(len(set(paths)), 12)
        self.assertTrue(all(path.is_file() for path in paths))

    def test_duplicate_and_missing_manifest_shards_rejected(self):
        duplicate = copy.deepcopy(self.manifest)
        duplicate["files"][-1] = duplicate["files"][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.validate(duplicate)
        missing = copy.deepcopy(self.manifest)
        missing["files"].pop()
        with self.assertRaisesRegex(ValueError, "twelve"):
            self.validate(missing)

    def test_escaped_and_symlinked_paths_rejected(self):
        for relative in [
            "../0000.parquet",
            "/tmp/0000.parquet",
            "datasets/raw/leagueData/../0000.parquet",
        ]:
            with self.subTest(relative=relative):
                manifest = copy.deepcopy(self.manifest)
                manifest["files"][0]["file"] = relative
                with self.assertRaisesRegex(ValueError, "path"):
                    self.validate(manifest)
        target = self.root / "datasets/raw/leagueData/0000.parquet"
        outside = self.root / "outside.parquet"
        target.rename(outside)
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "inside"):
            self.validate()

    def test_source_and_revision_changes_rejected(self):
        for field in ["source", "source_revision", "parquet_revision", "status"]:
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                manifest[field] = "changed"
                with self.assertRaisesRegex(ValueError, "fixed-revision"):
                    self.validate(manifest)

    def test_official_list_duplicates_missing_and_metadata_mismatch_rejected(self):
        for official in [self.official[:-1], self.official[:-1] + [self.official[0]]]:
            with self.subTest(entries=len(official)), self.assertRaises(ValueError):
                self.validate(official=official)
        for field, value in [("bytes", 1), ("sha256", "0" * 64)]:
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                manifest["files"][0][field] = value
                with self.assertRaisesRegex(ValueError, "saved official"):
                    self.validate(manifest)

    def test_modified_contents_rejected_even_when_size_is_unchanged(self):
        path = self.root / self.manifest["files"][0]["file"]
        path.write_bytes(b"X" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA256 changed"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
