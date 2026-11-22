# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Common-information player-history baselines and specialized adaptations."""

import torch
from torch import nn

from .omninet import OwO, masked_mean
from .regularization import ActivationDropoutSequential


class HistoryDeepSets(nn.Module):
    def __init__(
        self, history_features, static_features, width=32, classes=3, use_history=True, dropout=0.0
    ):
        super().__init__()
        self.use_history = use_history
        self.player = ActivationDropoutSequential(
            nn.Linear(history_features + static_features + 1, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            dropout=dropout,
        )
        self.head = ActivationDropoutSequential(
            nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, classes), dropout=dropout
        )

    def forward(self, history, static, history_valid, domain=None):
        sequence = masked_mean(history, history_valid)
        if not self.use_history:
            sequence = torch.zeros_like(sequence)
        coverage = history_valid.float().mean(-1, keepdim=True)
        players = self.player(torch.cat([sequence, static, coverage], -1))
        return self.head(players.mean(2).flatten(1))


def make_history_model(name, history_features, static_features, team_size, steps, config):
    common = dict(width=config["width"], classes=3)
    if name in {"history_deepsets", "static_deepsets"}:
        return HistoryDeepSets(
            history_features,
            static_features,
            **common,
            use_history=name == "history_deepsets",
            dropout=config.get("deepsets_dropout", 0.0),
        )
    if name in {"owo", "dawn", "owo_transformer"}:
        return OwO(
            history_features,
            static_features,
            team_size=team_size,
            history_length=steps,
            **common,
            heads=config["heads"],
            layers=config["layers"],
            dropout=config["dropout"],
            domains=2 if name == "dawn" else 0,
            ordered_slots=False,
            omni=name != "owo_transformer",
            pooling=config.get("pooling", "global"),
            norm_first=config.get("norm_first", False),
        )
    raise ValueError(f"Unknown history model: {name}")
