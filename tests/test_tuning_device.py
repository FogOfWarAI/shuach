# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Portable explicit HPO devices preserve candidate math and resume locks."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gamematch.tuning import search
from gamematch.tuning.data import Bundle, partition


class TuningDeviceTests(unittest.TestCase):
    def test_cpu_override_covers_neural_tasks_and_leaves_tabular_candidates_unchanged(self):
        historical = {
            dataset: {name: search.candidates(dataset, name) for name in names}
            for dataset, names in search.MATRIX.items()
        }
        self.assertEqual(search.search_spaces(), historical)
        overridden = search.search_spaces("cpu")
        neural, tabular = 0, 0
        for dataset, models in historical.items():
            for name, configs in models.items():
                for before, after in zip(configs, overridden[dataset][name]):
                    if before["tabular"]:
                        self.assertEqual(before, after)
                        self.assertNotIn("device", after)
                        tabular += 1
                    else:
                        self.assertEqual(after, {**before, "device": "cpu"})
                        neural += 1
        self.assertGreater(neural, 0)
        self.assertGreater(tabular, 0)

    def test_invalid_or_unavailable_devices_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i, device in enumerate(["gpu", "CPU", "cuda:0", "", 1, True, ["cpu"]]):
                with self.subTest(device=device):
                    output = root / f"invalid-{i}"
                    with self.assertRaisesRegex(ValueError, "Tuning device"):
                        search.run(root, output, device=device)
                    self.assertFalse(output.exists())
            for device, backend in [
                ("mps", "torch.backends.mps.is_available"),
                ("cuda", "torch.cuda.is_available"),
            ]:
                with self.subTest(device=device), patch(backend, return_value=False):
                    output = root / device
                    with self.assertRaisesRegex(
                        ValueError, f"Requested {device.upper()} device is unavailable"
                    ):
                        search.run(root, output, device=device)
                    self.assertFalse(output.exists())

    def test_cli_passes_explicit_device_and_rejects_invalid_choices(self):
        with (
            patch(
                "sys.argv", ["tuning", "--workspace", "work", "--output", "out", "--device", "cpu"]
            ),
            patch.object(search, "run") as run,
        ):
            search.main()
        run.assert_called_once_with("work", "out", None, None, device="cpu")
        with (
            patch("sys.argv", ["tuning", "--device", "gpu"]),
            patch.object(search, "run") as run,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit):
                search.main()
        run.assert_not_called()

    def test_real_cpu_fit_and_resume_keep_the_explicit_device_locked(self):
        rng = np.random.default_rng(91)
        parts = {}
        for name, n in [("train", 24), ("valid", 8), ("test", 8)]:
            parts[name] = partition(
                [rng.integers(1, 10, (n, 2, 5), dtype=np.int64)],
                np.arange(n, dtype=np.float32) % 2,
                np.array([f"{name}-{i}" for i in range(n)]),
                {},
            )
        bundle = Bundle("uci-dota2", "binary", parts, {"vocab_size": 10}, {"fixture": True})
        # Keep its original MPS request so this fixture proves the public override works.
        config = {
            **search.candidates("uci-dota2", "mlp")[0],
            "epochs": 1,
            "patience": 1,
            "hidden_dim": 8,
            "embedding_dim": 4,
            "batch_size": 16,
            "threads": 1,
        }
        self.assertEqual(config["device"], "mps")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            with (
                patch.object(search, "MATRIX", {"uci-dota2": ["mlp"]}),
                patch.object(search, "candidates", return_value=[config]),
                patch.object(search, "load_bundle", return_value=bundle),
                patch.object(search, "audit_primary_split", return_value={"fixture": True}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                search.run(root, output, device="cpu")
                protocol = json.loads((output / "protocol.json").read_text())
                self.assertEqual(protocol["device_override"], "cpu")
                self.assertEqual(protocol["spaces"]["uci-dota2"]["mlp"][0]["device"], "cpu")
                target = output / "uci-dota2/mlp"
                self.assertEqual(len(list(target.glob("final/*/result.json"))), 5)
                before = {
                    p: (p.read_bytes(), p.stat().st_mtime_ns)
                    for p in output.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                }
                with (
                    patch.object(search, "fit", side_effect=AssertionError("refit")),
                    patch.object(search, "predict_raw", side_effect=AssertionError("forward")),
                ):
                    search.run(root, output, device="cpu")
                    with self.assertRaisesRegex(ValueError, "Immutable protocol"):
                        search.run(root, output)
                    with patch("torch.cuda.is_available", return_value=True):
                        with self.assertRaisesRegex(ValueError, "Immutable protocol"):
                            search.run(root, output, device="cuda")
                self.assertEqual(
                    before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
                )
                self.assertEqual(config["device"], "mps")


if __name__ == "__main__":
    unittest.main()
