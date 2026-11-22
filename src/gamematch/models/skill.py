# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Common-information snapshot regression baselines for the MMR-Net adaptation."""

import torch
from torch import nn

from .omninet import MMRNet


class SnapshotBaseline(nn.Module):
    def __init__(self, features, hero_vocab, kind, steps=12, width=32, layers=2, dropout=0.1):
        super().__init__()
        self.kind = kind
        self.hero = nn.Embedding(hero_vocab, width // 2, padding_idx=0)
        self.lineup_projection = nn.Linear(width, width // 2)
        self.feature = nn.Linear(features, width - width // 2)
        if kind == "mlp":
            self.head = nn.Sequential(
                nn.Linear(steps * width, width * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(width * 4, 1),
            )
        elif kind == "gru":
            self.gru = nn.GRU(
                width,
                width,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0,
            )
            self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        else:
            raise ValueError(kind)

    def forward(self, snapshots, lineup, valid):
        heroes = self.hero(lineup).mean(2).flatten(1)
        line = self.lineup_projection(heroes)[:, None].expand(-1, snapshots.shape[1], -1)
        x = torch.cat([line, self.feature(snapshots)], -1).masked_fill(~valid.unsqueeze(-1), 0)
        if self.kind == "mlp":
            return self.head(x.flatten(1)).squeeze(-1)
        values, _ = self.gru(x)
        last = torch.where(valid, torch.arange(x.shape[1], device=x.device)[None], -1).max(1).values
        if (last < 0).any():
            raise ValueError("Snapshot regression requires at least one observed step")
        return self.head(values[torch.arange(len(x), device=x.device), last]).squeeze(-1)


def make_skill_model(name, features, hero_vocab, config):
    common = dict(
        steps=12, width=config["width"], layers=config["layers"], dropout=config["dropout"]
    )
    if name in {"mlp", "gru"}:
        return SnapshotBaseline(features, hero_vocab, name, **common)
    if name in {"transformer", "mmrnet"}:
        return MMRNet(features, hero_vocab, **common, heads=config["heads"], omni=name == "mmrnet")
    raise ValueError(name)
