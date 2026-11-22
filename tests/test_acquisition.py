# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Offline acquisition tests protect integrity, existing files and extraction paths."""

import hashlib
import io
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

from gamematch.cli import main
from gamematch.data.acquisition import REGISTRY, fetch_dataset


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "application/zip"}

    def geturl(self):
        return "https://public.example/test.zip"


def archive_bytes(files):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.body = archive_bytes({"data.csv": "x,y\n1,2\n", "unused.py": "raise RuntimeError()"})
        self.asset = {
            "url": "https://public.example/test.zip",
            "source": "https://public.example/",
            "license": "Test fixture only",
            "filename": "test.zip",
            "bytes": len(self.body),
            "sha256": hashlib.sha256(self.body).hexdigest(),
            "members": {"data.csv": "extracted/data.csv"},
        }

    def tearDown(self):
        self.temp.cleanup()

    def fetch(self, body=None, changes=None):
        asset = self.asset | (changes or {})
        with (
            patch.dict(REGISTRY, {"fixture": asset}),
            patch("gamematch.data.acquisition.urlopen", return_value=Response(body or self.body)),
        ):
            return fetch_dataset("fixture", self.root)

    def test_cli_download_cache_and_selective_extraction(self):
        with (
            patch.dict(REGISTRY, {"fixture": self.asset}),
            patch(
                "gamematch.data.acquisition.urlopen", return_value=Response(self.body)
            ) as network,
            redirect_stdout(io.StringIO()) as output,
        ):
            main(["fetch-data", "fixture", "--data-root", str(self.root)])
            record = json.loads(output.getvalue())
            self.assertEqual(record["status"], "verified")
            self.assertFalse(record["cached_download"])
            self.assertTrue(fetch_dataset("fixture", self.root)["cached_download"])
            network.assert_called_once()
        self.assertEqual((self.root / "extracted/data.csv").read_text(), "x,y\n1,2\n")
        self.assertFalse((self.root / "unused.py").exists())
        self.assertEqual(list((self.root / "raw").iterdir()), [self.root / "raw/test.zip"])
        receipt = json.loads((self.root / "acquisition/fixture.json").read_text())
        self.assertEqual(receipt["raw_sha256"], self.asset["sha256"])
        self.assertTrue(receipt["cached_download"])
        self.assertEqual(
            list((self.root / "acquisition").iterdir()), [self.root / "acquisition/fixture.json"]
        )

    def test_failed_provenance_commit_preserves_existing_receipt(self):
        self.fetch()
        receipt = self.root / "acquisition/fixture.json"
        previous = receipt.read_bytes()
        with (
            patch("gamematch.data.acquisition.Path.replace", side_effect=OSError("commit failed")),
            self.assertRaisesRegex(OSError, "commit failed"),
        ):
            self.fetch()
        self.assertEqual(receipt.read_bytes(), previous)
        self.assertEqual(list(receipt.parent.iterdir()), [receipt])

    def test_provenance_symlink_escape_rejected_before_data_changes(self):
        data_root = self.root / "datasets"
        receipt = data_root / "acquisition/fixture.json"
        receipt.parent.mkdir(parents=True)
        unrelated = self.root / "unrelated.json"
        unrelated.write_text("preserve this file")
        receipt.symlink_to(unrelated)
        raw = data_root / "raw/test.zip"
        for cached in (False, True):
            with self.subTest(cached=cached):
                if cached:
                    raw.parent.mkdir()
                    raw.write_bytes(self.body)
                with (
                    patch.dict(REGISTRY, {"fixture": self.asset}),
                    patch("gamematch.data.acquisition.urlopen") as network,
                    self.assertRaisesRegex(ValueError, "escapes"),
                ):
                    fetch_dataset("fixture", data_root)
                network.assert_not_called()
                self.assertEqual(unrelated.read_text(), "preserve this file")
                self.assertTrue(receipt.is_symlink())
                self.assertFalse((data_root / "extracted").exists())
                self.assertEqual(raw.exists(), cached)

    def test_download_hash_size_and_html_failures_leave_no_partial_asset(self):
        for body, changes in [
            (self.body, {"sha256": "0" * 64}),
            (self.body, {"bytes": len(self.body) - 1}),
            (self.body[:-4], {}),
        ]:
            with self.assertRaises(ValueError):
                self.fetch(body, changes)
            self.assertEqual(list((self.root / "raw").iterdir()), [])
        with patch.object(Response, "headers", {"Content-Type": "text/html"}):
            with self.assertRaises(ValueError):
                self.fetch()
        self.assertFalse((self.root / "raw/test.zip").exists())

    def test_modified_cache_is_not_overwritten(self):
        self.fetch()
        target = self.root / "extracted/data.csv"
        target.write_text("user changed data")
        with self.assertRaisesRegex(ValueError, "Existing extracted"):
            self.fetch()
        self.assertEqual(target.read_text(), "user changed data")
        raw = self.root / "raw/test.zip"
        raw.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.fetch()
        self.assertEqual(raw.read_bytes(), b"corrupt")

    def test_nested_archive_and_path_escape(self):
        nested = archive_bytes({"inside.csv": "safe"})
        body = archive_bytes({"nested.zip": nested})
        changes = {
            "nested_archive": "nested.zip",
            "members": {"inside.csv": "extracted/nested.csv"},
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        self.fetch(body, changes)
        self.assertEqual((self.root / "extracted/nested.csv").read_text(), "safe")
        changes["members"] = {"inside.csv": "../outside.csv"}
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.fetch(body, changes)

    def test_symlink_and_unknown_source_rejected(self):
        info = ZipInfo("data.csv")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr(info, "../../outside")
        body = buffer.getvalue()
        with self.assertRaisesRegex(ValueError, "regular"):
            self.fetch(body, {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
        with self.assertRaises(ValueError):
            fetch_dataset("unknown", self.root)


if __name__ == "__main__":
    unittest.main()
