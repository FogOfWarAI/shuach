# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Reload trusted experiment artifacts without refitting transforms or calibration.

Inputs must already obey the documented identity and availability contracts.
These predictors do not reconstruct histories from arbitrary live queue records.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import softmax
from torch import nn

from .data.heroes import HeroVocabulary
from .models import make_model
from .models.history import make_history_model
from .models.ordinal import OrdinalMLP
from .models.skill import make_skill_model
from .models.tabular import hero_counts, predict_kill_counts


def _read(path):
    return json.loads(path.read_text())


def _device(device):
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("Choose cpu, mps or cuda explicitly")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable in this process")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable in this process")
    return device


def _batch_count(batch_size, length):
    if type(batch_size) is not int or batch_size < 1 or length < 1:
        raise ValueError("Positive batch size and nonempty input required")
    return range(0, length, batch_size)


def _temperature(value):
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ValueError("A finite positive validation temperature is required")
    return value


def _tensor(array, device):
    # PyTorch cannot wrap a NumPy view that walks backwards, e.g. rows[::-1].
    if any(stride < 0 for stride in array.strides):
        array = array.copy()
    return torch.from_numpy(array).to(device)


def _vocabulary(values):
    vocabulary = HeroVocabulary()
    vocabulary.values = np.asarray(values, dtype=np.int64)
    return vocabulary


class HeroPredictor:
    """Reload a static hero-outcome fit and its training vocabulary.

    Args:
        directory: Trusted seed/model directory whose parent contains data-manifest.json.
        device: Torch device for neural fits ("cpu", "cuda" or "mps").
        batch_size: Maximum rosters per neural forward pass.
    """

    def __init__(self, directory, device="cpu", batch_size=1024):
        self.directory = Path(directory)
        self.device, self.batch_size = _device(device), batch_size
        metrics = _read(self.directory / "metrics.json")
        self.name = metrics["model"]
        data = _read(self.directory.parent / "data-manifest.json")
        self.vocabulary = _vocabulary(data["vocabulary"])
        if self.name == "prior":
            self.prior = _read(self.directory / "checkpoint.json")["positive_probability"]
        elif self.name in {"logistic", "lightgbm"}:
            self.model = joblib.load(self.directory / "checkpoint.joblib")
        else:
            saved = torch.load(
                self.directory / "checkpoint.pt", map_location="cpu", weights_only=True
            )
            config = saved["training"]
            self.model = make_model(
                self.name, self.vocabulary.size, config["embedding_dim"], config["hidden_dim"]
            )
            self.model.load_state_dict(saved["state_dict"])
            self.model.to(self.device).eval()

    def predict(self, rosters):
        """Score both sides in the supplied order.

        Args:
            rosters: Nonempty integer array [N, 2, 5] of raw hero IDs.
                Unseen IDs map to the training vocabulary's unknown token.

        Returns:
            An array [N] of left-side win probabilities, in roster order.
        """
        rosters = np.asarray(rosters)
        if (
            rosters.ndim != 3
            or rosters.shape[1:] != (2, 5)
            or not np.issubdtype(rosters.dtype, np.integer)
        ):
            raise ValueError("Expected integer [N,2,5] hero rosters")
        batches = _batch_count(self.batch_size, len(rosters))
        if self.name == "prior":
            return np.full(len(rosters), self.prior)
        encoded = self.vocabulary.transform(rosters)
        if self.name in {"logistic", "lightgbm"}:
            return self.model.predict_proba(hero_counts(encoded, self.vocabulary.size))[:, 1]
        values = []
        with torch.no_grad():
            for i in batches:
                x = torch.from_numpy(encoded[i : i + self.batch_size]).to(self.device)
                values.append(self.model(x).sigmoid().cpu().numpy())
        return np.concatenate(values)


class HistoryPredictor:
    """Reload an Overwatch history fit, its normalizer and validation temperature.

    Args:
        directory: Trusted seed/model directory whose parent contains the run manifest.
        device: Torch device for the network ("cpu", "cuda" or "mps").
        batch_size: Maximum matches per forward pass.
    """

    def __init__(self, directory, device="cpu", batch_size=128):
        self.directory = Path(directory)
        self.device, self.batch_size = _device(device), batch_size
        manifest = _read(self.directory.parent / "manifest.json")
        self.normalizer, self.config = manifest["normalizer"], manifest["config"]
        self.temperature = _temperature(_read(self.directory / "metrics.json")["temperature"])
        saved = torch.load(self.directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.history_shape = (
            2,
            saved["team_size"],
            saved["history_length"],
            saved["history_features"],
        )
        self.static_shape = (2, saved["team_size"], saved["static_features"])
        self.model = make_history_model(
            saved["model"],
            saved["history_features"],
            saved["static_features"],
            saved["team_size"],
            saved["history_length"],
            saved["training"],
        )
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(self.device).eval()

    def predict(self, history, static, valid, domain):
        """Predict from completed, unstandardized histories in their original feature order.

        Args:
            history: Finite array [N, 2, P, T, H] of right-aligned histories.
                P, T and H must match the fitted checkpoint.
            static: Finite per-player features [N, 2, P, S] before normalization.
            valid: Boolean array [N, 2, P, T]. True marks an observed step.
            domain: Array [N] with 0 for noncompetitive and 1 for competitive.

        Returns:
            Probabilities [N, 3] for left-side loss, draw and win, in that order.
        """
        h, s = (
            np.array(history, dtype=np.float32, copy=True),
            np.array(static, dtype=np.float32, copy=True),
        )
        valid, domain = np.asarray(valid), np.asarray(domain)
        if (
            h.ndim != 5
            or h.shape[1:] != self.history_shape
            or s.shape != (len(h), *self.static_shape)
            or valid.shape != h.shape[:-1]
            or valid.dtype != bool
            or domain.shape != (len(h),)
            or not np.isin(domain, [0, 1]).all()
        ):
            raise ValueError("Unaligned history, static, validity or target-domain arrays")
        if not np.isfinite(h).all() or not np.isfinite(s).all():
            raise ValueError("History features must be finite")
        if self.config.get("target_domain_features_only", False):
            # Training withholds global and cross-domain statistics before normalization.
            s[..., :4] = 0
            for d in (0, 1):
                s[domain == d, ..., (1 - d) * 4 + 4 : (1 - d) * 4 + 8] = 0
        n = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in self.normalizer.items()
            if key != "fitted_partition"
        }
        h = (h - n["history_mean"]) / n["history_scale"]
        h[~valid] = 0
        s = (s - n["static_mean"]) / n["static_scale"]
        domain = domain.astype(np.int64, copy=False)
        values = []
        with torch.no_grad():
            for i in _batch_count(self.batch_size, len(h)):
                tensors = [
                    _tensor(a[i : i + self.batch_size], self.device) for a in [h, s, valid, domain]
                ]
                values.append(self.model(*tensors[:3], domain=tensors[3]).cpu().numpy())
        return softmax(np.concatenate(values) / self.temperature, axis=1)


class AsymmetricPredictor:
    """Reload a Dead by Daylight predator-outcome fit for CPU inference.

    Args:
        directory: Trusted seed/model directory under the dataset-specific output.
            Learned models also need the parent directory's fitted preprocessor.
        batch_size: Maximum matches per neural forward pass.
    """

    def __init__(self, directory, batch_size=1024):
        self.directory, self.batch_size = Path(directory), batch_size
        result = _read(self.directory / "metrics.json")
        self.name, self.temperature = result["model"], _temperature(result["temperature"])
        if self.name in {"prior", "history_frequency"}:
            self.rule = _read(self.directory / "checkpoint.json")
        else:
            self.preprocessor = joblib.load(self.directory.parent / "preprocessor.joblib")
            if self.name in {"logistic", "lightgbm"}:
                self.model = joblib.load(self.directory / "checkpoint.joblib")
            else:
                saved = torch.load(
                    self.directory / "checkpoint.pt", map_location="cpu", weights_only=True
                )
                width, hidden = saved["input_dim"], saved["hidden_dim"]
                self.model = (
                    OrdinalMLP(width, hidden)
                    if self.name == "ordinal"
                    else nn.Sequential(
                        nn.Linear(width, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, 5),
                    )
                )
                self.model.load_state_dict(saved["state_dict"])
                self.model.eval()

    def predict(self, features):
        """Estimate the distribution of the number of prey eliminated.

        Args:
            features: Nonempty pre-match DataFrame from the asymmetric adapter.
                Each row describes one predator, map and avatar at prediction time.

        Returns:
            Calibrated probabilities [N, 5]. Column k is the probability of k kills.

        Raises:
            ValueError: The history-frequency model received invalid count fractions.
        """
        batches = _batch_count(self.batch_size, len(features))
        if self.name == "prior":
            p = np.tile(self.rule["probabilities"], (len(features), 1))
        elif self.name == "history_frequency":
            p = features[[f"previous_kill_fraction_{i}" for i in range(5)]].to_numpy(dtype=float)
            if not ((p >= 0) & (p <= 1)).all() or not np.allclose(p.sum(1), 1, atol=1e-6):
                raise ValueError("Kill fractions must be probabilities summing to one in each row")
        else:
            x = np.asarray(self.preprocessor.transform(features), dtype=np.float32)
            if self.name in {"logistic", "lightgbm"}:
                p = predict_kill_counts(self.model, x)
            else:
                with torch.no_grad():
                    logits = np.concatenate(
                        [
                            self.model(torch.from_numpy(x[i : i + self.batch_size])).numpy()
                            for i in batches
                        ]
                    )
                p = softmax(logits, axis=1)
        return softmax(np.log(np.clip(p, 1e-12, 1)) / self.temperature, axis=1)


class SkillPredictor:
    """Reload a predictor of a player's future observed OpenSkill mean.

    Args:
        directory: Trusted seed/model directory whose parent contains the run manifest.
        device: Torch device for neural fits ("cpu", "cuda" or "mps").
        batch_size: Maximum player examples per neural forward pass.
    """

    def __init__(self, directory, device="cpu", batch_size=256):
        self.directory = Path(directory)
        self.device, self.batch_size = _device(device), batch_size
        manifest = _read(self.directory.parent / "manifest.json")
        self.normalizer = manifest["normalizer"]
        self.vocabulary = _vocabulary(manifest["hero_vocabulary"])
        self.name = _read(self.directory / "metrics.json")["model"]
        if self.name in {"current", "mean_delta"}:
            self.residual = _read(self.directory / "checkpoint.json")["residual"]
        elif self.name == "ridge":
            self.model = joblib.load(self.directory / "checkpoint.joblib")
        else:
            saved = torch.load(
                self.directory / "checkpoint.pt", map_location="cpu", weights_only=True
            )
            self.model = make_skill_model(
                saved["model"], saved["features"], saved["hero_vocab"], saved["training"]
            )
            self.model.load_state_dict(saved["state_dict"])
            self.model.to(self.device).eval()

    def predict(self, snapshots, valid, lineups, current_mu):
        """Predict the rating in the original rating units after five more observed games.

        Args:
            snapshots: Finite array [N, 12, 6] from the skill adapter, with log
                transforms already applied and training standardization still pending.
            valid: Boolean array [N, 12] with at least one observed step per player.
            lineups: Raw integer hero IDs [N, 2, 5], focal player's team first.
            current_mu: Finite current OpenSkill posterior means [N].

        Returns:
            An array [N] of future rating estimates in input row order.
        """
        x = np.asarray(snapshots, dtype=np.float32)
        valid, lineups, current = (
            np.asarray(valid),
            np.asarray(lineups),
            np.asarray(current_mu, dtype=float),
        )
        if (
            x.ndim != 3
            or x.shape[1:] != (12, 6)
            or valid.shape != x.shape[:2]
            or valid.dtype != bool
            or lineups.shape != (len(x), 2, 5)
            or not np.issubdtype(lineups.dtype, np.integer)
            or current.shape != (len(x),)
            or not valid.any(1).all()
        ):
            raise ValueError(
                "Expected aligned 12x6 snapshots, masks, 5v5 lineups and current means"
            )
        if not np.isfinite(x).all() or not np.isfinite(current).all():
            raise ValueError("Snapshot and rating values must be finite")
        batches = _batch_count(self.batch_size, len(x))
        if self.name in {"current", "mean_delta"}:
            return current + self.residual
        n = self.normalizer
        x = (
            (x - np.asarray(n["feature_mean"], dtype=np.float32))
            / np.asarray(n["feature_scale"], dtype=np.float32)
        ).astype(np.float32)
        x[~valid] = 0
        encoded = self.vocabulary.transform(lineups)
        if self.name == "ridge":
            matrix = np.concatenate(
                [
                    x.reshape(len(x), -1),
                    valid.astype(np.float32),
                    hero_counts(encoded, self.vocabulary.size).toarray(),
                ],
                1,
            )
            prediction = self.model.predict(matrix)
        else:
            with torch.no_grad():
                prediction = np.concatenate(
                    [
                        self.model(
                            _tensor(x[i : i + self.batch_size], self.device),
                            _tensor(encoded[i : i + self.batch_size], self.device),
                            _tensor(valid[i : i + self.batch_size], self.device),
                        )
                        .cpu()
                        .numpy()
                        for i in batches
                    ]
                )
        return current + prediction * n["delta_scale"] + n["delta_mean"]
