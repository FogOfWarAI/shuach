# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Reload binary LoL history predictors against their immutable feature store."""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from .data.heroes import file_hash
from .data.indexed_history import IndexedHistoryStore
from .models.history import HistoryDeepSets
from .models.omninet import OwO


class IndexedHistoryPredictor:
    """Replay a binary league-neural-v1 selection against its original feature store.

    Args:
        run_directory: Trusted seed directory containing selection.json. An
            ancestor must contain the producing protocol.json and source snapshot.
        data_directory: Original indexed store, including manifest.json and
            its checksummed NumPy arrays.
        device: Torch device for inference ("cpu", "mps" or "cuda").
        threads: Positive integer passed to torch.set_num_threads, which sets
            the process-wide intra-op thread count.

    Raises:
        ValueError: The selection, model source, store or checkpoint fails validation.
        RuntimeError: The requested accelerator is unavailable.
    """

    def __init__(self, run_directory, data_directory, device="cpu", threads=2):
        self.run_directory = Path(run_directory)
        data_directory = Path(data_directory)
        if type(threads) is not int or threads < 1:
            raise ValueError("A positive integer thread count is required")
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("Choose cpu, mps or cuda")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.set_num_threads(threads)
        self.device = torch.device(device)
        path = self.run_directory / "selection.json"
        self.selection = json.loads(path.read_text())
        selection = self.selection
        protocol_path = next(
            (
                parent / "protocol.json"
                for parent in self.run_directory.parents
                if (parent / "protocol.json").is_file()
            ),
            None,
        )
        if protocol_path is None or file_hash(protocol_path) != selection["protocol_sha256"]:
            raise ValueError("Selected fit has no matching experiment protocol")
        protocol = json.loads(protocol_path.read_text())
        for name in ["history.py", "omninet.py", "regularization.py"]:
            current = Path(__file__).parent / "models" / name
            archived = protocol_path.parent / "source-snapshot/gamematch/models" / name
            if file_hash(current) != file_hash(archived):
                raise ValueError("Model implementation differs from the producing source: " + name)
        if protocol["data_manifest_sha256"] != file_hash(data_directory / "manifest.json"):
            raise ValueError("Predictor requires its exact original feature store")
        if (
            selection["test_evaluated"] is not False
            or selection["configuration"] != protocol["configuration"]
            or selection["model"] not in protocol["models"]
            or selection["seed"] not in protocol["seeds"]
            or selection["columns"] not in protocol["variants"].values()
            or protocol["classes"] != 2
        ):
            raise ValueError("Invalid binary-history selected fit")
        expected = {"checkpoint.pt", "validation.npz", "history.json"}
        if set(selection["artifact_sha256"]) != expected:
            raise ValueError("Unexpected selected artifact inventory")
        for name, digest in selection["artifact_sha256"].items():
            if file_hash(self.run_directory / name) != digest:
                raise ValueError("Selected artifact changed: " + name)
        self.columns = selection["columns"]
        if self.columns not in [[0, 1, 24], list(range(25))]:
            raise ValueError("Unsupported LoL information contract")
        self.temperature = float(selection["temperature"])
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("A finite positive validation temperature is required")
        self.store = IndexedHistoryStore(data_directory)
        if (
            self.store.manifest["classes"] != 2
            or self.store.history.shape[1:] != (2, 5, 10)
            or self.store.static.shape[-1] != 6
            or self.store.performance.shape[-1] != 24
        ):
            raise ValueError("The stored input dimensions do not match the LoL contract")
        config = selection["configuration"]
        if selection["model"] == "history_deepsets":
            model = HistoryDeepSets(
                len(self.columns), 6, width=config["width"], classes=2, dropout=config["dropout"]
            )
        elif selection["model"] == "owo":
            model = OwO(
                len(self.columns),
                6,
                team_size=5,
                history_length=10,
                width=config["width"],
                heads=2,
                layers=1,
                classes=2,
                domains=0,
                dropout=config["dropout"],
                ordered_slots=False,
                pooling="team",
                norm_first=False,
            )
        else:
            raise ValueError("Unsupported binary history architecture")
        checkpoint = torch.load(
            self.run_directory / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        if checkpoint["selection"] != {
            key: value for key, value in selection.items() if key != "artifact_sha256"
        }:
            raise ValueError("Checkpoint metadata disagrees with selected fit")
        if not all(
            torch.is_tensor(v) and torch.isfinite(v).all()
            for v in checkpoint["state_dict"].values()
        ):
            raise ValueError("Checkpoint contains non-finite or non-tensor weights")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if sum(p.numel() for p in model.parameters()) != selection["parameters"]:
            raise ValueError("Checkpoint parameter count changed")
        self.model = model.to(self.device).eval()

    def predict_logits(self, rows, batch_size=256):
        """Compute uncalibrated team 100 loss/win scores.

        Args:
            rows: One-dimensional integer indices into the feature store.
                Order and repetitions are preserved.
            batch_size: Positive number of matches to expand per model call.

        Returns:
            An array [N, 2] with columns for team 100 loss and win. An empty
            integer vector returns an array of shape [0, 2].

        Raises:
            ValueError: Rows have the wrong type, shape or range, or batch size is invalid.
        """
        rows = np.asarray(rows)
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("Rows must be a vector of integer indices")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        if ((rows < 0) | (rows >= len(self.store.target))).any():
            raise ValueError("Prediction row outside the stored match range")
        if not len(rows):
            return np.empty((0, 2), dtype=np.float32)
        output = []
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                arrays, _ = self.store.batch(rows[start : start + batch_size], self.columns)
                inputs = tuple(torch.from_numpy(value).to(self.device) for value in arrays)
                output.append(self.model(*inputs[:3], domain=inputs[3]).cpu().numpy())
        logits = np.concatenate(output)
        if logits.shape != (len(rows), 2) or not np.isfinite(logits).all():
            raise FloatingPointError("Invalid binary logits")
        return logits

    def predict_proba(self, rows, batch_size=256, calibrated=True):
        """Convert loss/win logits to probabilities in the same row and class order.

        Args:
            rows: One-dimensional integer indices into the original store.
            batch_size: Positive number of matches to expand per model call.
            calibrated: Apply the saved validation temperature before softmax
                when true. False uses unit temperature.

        Returns:
            An array [N, 2]. Column 1 is the probability that team 100 wins.
        """
        raw = self.predict_logits(rows, batch_size).astype(float)
        return softmax(raw / (self.temperature if calibrated else 1.0), axis=1)
