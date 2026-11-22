# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Diagnostics expose broken optional imports and work in fresh source packages."""

import contextlib
import importlib.metadata
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np

from gamematch.cli import main
from gamematch.models.tabular import train_tabular


class DependencyDiagnosticsTests(unittest.TestCase):
    def doctor(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["doctor"])
        return json.loads(output.getvalue())

    def test_doctor_imports_optional_dependencies_and_preserves_device_fields(self):
        with (
            patch("gamematch.cli.importlib.metadata.version", return_value="1.2.3"),
            patch("gamematch.cli.importlib.import_module") as importer,
        ):
            result = self.doctor()
        self.assertTrue(
            {"python", "torch", "machine", "logical_cpus", "cuda_available", "mps_available"}
            <= result.keys()
        )
        self.assertEqual(
            importer.call_args_list, [call(n) for n in ("lightgbm", "trueskill", "openskill")]
        )
        self.assertEqual(
            result["optional_dependencies"],
            {
                n: {"status": "available", "version": "1.2.3"}
                for n in ("lightgbm", "trueskill", "openskill")
            },
        )

    def test_doctor_reports_absent_dependency_without_importing_it(self):
        def version(name):
            if name == "lightgbm":
                raise importlib.metadata.PackageNotFoundError(name)
            return "1.2.3"

        with (
            patch("gamematch.cli.importlib.metadata.version", side_effect=version),
            patch("gamematch.cli.importlib.import_module") as importer,
        ):
            result = self.doctor()
        self.assertEqual(
            result["optional_dependencies"]["lightgbm"],
            {"status": "not_installed", "version": None},
        )
        self.assertEqual(importer.call_args_list, [call("trueskill"), call("openskill")])

    def test_doctor_keeps_installed_version_when_optional_import_fails(self):
        for error in (
            OSError("libomp.dylib unavailable"),
            ImportError("native dependency unavailable"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch("gamematch.cli.importlib.metadata.version", return_value="4.0.0"),
                patch("gamematch.cli.importlib.import_module", side_effect=error),
            ):
                result = self.doctor()
                for dependency in result["optional_dependencies"].values():
                    self.assertEqual(dependency["status"], "import_error")
                    self.assertEqual(dependency["version"], "4.0.0")
                    self.assertIn(str(error), dependency["error"])

    def test_lightgbm_failure_gives_install_or_runtime_guidance_before_checkpoint(self):
        rosters = np.tile(np.arange(1, 11).reshape(1, 2, 5), (2, 1, 1))
        records = SimpleNamespace(rosters=rosters, labels=np.array([0, 1]))
        vocabulary = SimpleNamespace(size=11, transform=lambda values: values)
        original_import = __import__
        for error, message in (
            (ImportError("lightgbm missing"), r"gamematch-research\[tabular\]"),
            (OSError("libomp.dylib unavailable"), "OpenMP"),
        ):

            def importer(name, *args, **kwargs):
                if name == "lightgbm":
                    raise error
                return original_import(name, *args, **kwargs)

            with (
                self.subTest(error=type(error).__name__),
                tempfile.TemporaryDirectory() as temporary,
                patch("builtins.__import__", side_effect=importer),
                patch("gamematch.models.tabular.joblib.dump") as save,
            ):
                with self.assertRaisesRegex(RuntimeError, message) as caught:
                    train_tabular(
                        "lightgbm",
                        {name: records for name in ("train", "valid", "test")},
                        vocabulary,
                        SimpleNamespace(),
                        0,
                        Path(temporary),
                    )
                self.assertIs(caught.exception.__cause__, error)
                save.assert_not_called()

    def test_distribution_inspector_creates_report_directory_in_fresh_checkout(self):
        script = Path(__file__).resolve().parents[1] / "scripts/inspect_distribution.py"
        spec = importlib.util.spec_from_file_location("distribution_diagnostics_probe", script)
        inspector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inspector)
        initial = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                self.assertFalse(Path("reports").exists())
                with (
                    patch.object(
                        inspector, "current_distributions", return_value=[Path("fixture.whl")]
                    ),
                    patch.object(inspector, "inspect", return_value={"file": "fixture.whl"}),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    inspector.main([])
                report = json.loads(Path("reports/package-inspection.json").read_text())
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["artifacts"], [{"file": "fixture.whl"}])
            finally:
                os.chdir(initial)


if __name__ == "__main__":
    unittest.main()
