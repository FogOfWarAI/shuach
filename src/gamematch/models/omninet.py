# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Explicit dense OmniNet adaptation for short matchmaking sequences.

All transformer-layer tokens participate in meta self-attention. Its output is
rearranged by token, concatenated across depth and projected back to model width
before the residual addition. This fixes an implementation choice left
underspecified in CUPID/QuickSkill; it is not the authors' released code.
Dense attention is intended for short sequences, not long-context workloads.
"""

import torch
from torch import nn


def masked_mean(x, valid):
    return (x * valid.unsqueeze(-1)).sum(-2) / valid.sum(-1, keepdim=True).clamp_min(1)


class OmniEncoder(nn.Module):
    def __init__(self, width=32, heads=4, layers=2, dropout=0.1, omni=True, norm_first=False):
        super().__init__()
        if width % heads or layers < 1:
            raise ValueError("Width must be divisible by attention heads and depth positive")
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width,
                    heads,
                    dim_feedforward=width * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=norm_first,
                )
                for _ in range(layers)
            ]
        )
        self.omni = omni
        if omni:
            self.meta_attention = nn.MultiheadAttention(
                width, heads, dropout=dropout, batch_first=True
            )
            self.depth_projection = nn.Sequential(
                nn.Linear(width * layers, width), nn.GELU(), nn.Linear(width, width)
            )
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(width)

    def forward(self, x, valid=None):
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError("OmniEncoder expects batch x nonempty sequence x width")
        if valid is None:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if valid.shape != x.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Validity mask must be boolean and match the sequence")
        # Prevent attention on an entirely masked row. The synthetic sentinel is
        # zero and the complete empty sequence is zeroed again before returning.
        safe = valid.clone()
        safe[:, 0] |= ~valid.any(-1)
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        states = []
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~safe)
            x = x.masked_fill(~valid.unsqueeze(-1), 0)
            states.append(x)
        if self.omni:
            stack = torch.stack(states, dim=1)  # B, depth, time, width
            b, depth, steps, width = stack.shape
            flattened = stack.reshape(b, depth * steps, width)
            meta, _ = self.meta_attention(
                flattened,
                flattened,
                flattened,
                key_padding_mask=~safe.repeat(1, depth),
                need_weights=False,
            )
            depthwise = (
                meta.reshape(b, depth, steps, width)
                .transpose(1, 2)
                .reshape(b, steps, depth * width)
            )
            x = self.norm(x + self.dropout(self.depth_projection(depthwise)))
        return x.masked_fill(~valid.unsqueeze(-1), 0)


class ScalarFeatureTokens(nn.Module):
    """One learned affine embedding per standardized numeric feature."""

    def __init__(self, features, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(features, width))
        self.bias = nn.Parameter(torch.zeros(features, width))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class OwO(nn.Module):
    """TOE + SOE + POE, with optional CHAMP-style DAKE context injection.

    ordered_slots=True preserves CUPID's team-and-role slot sensitivity. When
    roles are unobserved, ordered_slots=False uses side/view embeddings and
    pooled output so within-team order is immaterial. The latter is a declared
    public-data adaptation. All three encoders retain cross-layer attention.
    """

    def __init__(
        self,
        history_features,
        static_features,
        team_size=5,
        history_length=10,
        width=32,
        heads=4,
        layers=2,
        dropout=0.1,
        classes=2,
        domains=0,
        domain_features=0,
        context_tokens=2,
        ordered_slots=True,
        omni=True,
        pooling="global",
        norm_first=False,
    ):
        super().__init__()
        self.team_size, self.history_length = team_size, history_length
        self.ordered_slots, self.domains = ordered_slots, domains
        if pooling not in {"global", "team"} or (ordered_slots and pooling != "global"):
            raise ValueError("Team pooling is an unordered-roster adaptation")
        self.pooling = pooling
        self.history_projection = nn.Linear(history_features, width)
        self.time_position = nn.Embedding(history_length, width)
        self.static_embedding = ScalarFeatureTokens(static_features, width)
        self.toe = OmniEncoder(width, heads, layers, dropout, omni, norm_first)
        self.soe = OmniEncoder(width, heads, layers, dropout, omni, norm_first)
        self.poe = OmniEncoder(width, heads, layers, dropout, omni, norm_first)
        self.side = nn.Embedding(2, width)
        self.view = nn.Embedding(2, width)
        if ordered_slots:
            self.slot = nn.Embedding(team_size, width)
        if domains:
            self.domain_embedding = nn.Embedding(domains, width)
            self.dake = nn.Sequential(
                nn.Linear(width + domain_features, width),
                nn.GELU(),
                nn.Linear(width, context_tokens * width),
            )
            self.context_tokens, self.width = context_tokens, width
        head_width = team_size * 4 * width if ordered_slots else width
        if domains and ordered_slots:
            head_width += width
        if pooling == "team":
            head_width = 4 * width + (width if domains else 0)
        self.head = nn.Sequential(
            nn.Linear(head_width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, classes),
        )

    def forward(self, history, static, history_valid, domain=None, domain_attributes=None):
        b, sides, players, steps, _ = history.shape
        if (sides, players, steps) != (2, self.team_size, self.history_length):
            raise ValueError("History shape does not match configured teams and history length")
        if static.shape[:3] != (b, 2, players) or history_valid.shape != history.shape[:-1]:
            raise ValueError("Unaligned static/history features")
        count = b * 2 * players
        position = self.time_position(torch.arange(steps, device=history.device))
        temporal = self.history_projection(history).reshape(count, steps, -1) + position
        temporal_mask = history_valid.reshape(count, steps)
        spatial = self.static_embedding(static).reshape(count, static.shape[-1], -1)
        spatial_mask = torch.ones(spatial.shape[:2], dtype=torch.bool, device=history.device)
        context = None
        if self.domains:
            if domain is None:
                raise ValueError("DAWN requires the target domain")
            embedding = self.domain_embedding(domain)
            if domain_attributes is not None:
                embedding = torch.cat([embedding, domain_attributes], -1)
            context = self.dake(embedding).reshape(b, self.context_tokens, self.width)
            player_context = (
                context[:, None]
                .expand(-1, 2 * players, -1, -1)
                .reshape(count, self.context_tokens, self.width)
            )
            temporal = torch.cat([temporal, player_context], 1)
            spatial = torch.cat([spatial, player_context], 1)
            extra = torch.ones(
                (count, self.context_tokens), dtype=torch.bool, device=history.device
            )
            temporal_mask = torch.cat([temporal_mask, extra], 1)
            spatial_mask = torch.cat([spatial_mask, extra], 1)
        temporal = masked_mean(self.toe(temporal, temporal_mask), temporal_mask).reshape(
            b, 2, players, -1
        )
        spatial = masked_mean(self.soe(spatial, spatial_mask), spatial_mask).reshape(
            b, 2, players, -1
        )
        tokens = torch.stack([temporal, spatial], dim=3)
        tokens = (
            tokens
            + self.side.weight[None, :, None, None, :]
            + self.view.weight[None, None, None, :, :]
        )
        if self.ordered_slots:
            tokens = tokens + self.slot.weight[None, None, :, None, :]
        tokens = tokens.reshape(b, players * 4, -1)
        if context is not None:
            tokens = torch.cat([tokens, context.mean(1, keepdim=True)], 1)
        encoded = self.poe(tokens)
        if self.pooling == "team":
            # Retain ordered sides and temporal/spatial views, while averaging
            # over interchangeable players only. The domain token stays separate.
            pooled = encoded[:, : players * 4].reshape(b, 2, players, 2, -1).mean(2).flatten(1)
            if self.domains:
                pooled = torch.cat([pooled, encoded[:, -1]], -1)
        else:
            pooled = encoded.flatten(1) if self.ordered_slots else encoded.mean(1)
        return self.head(pooled)


class MMRNet(nn.Module):
    """QuickSkill-style snapshot regression with separate lineup embedding.

    Dense continuous feature projection replaces the industrial discretized
    feature embedding. Caller supplies labels for a declared skill proxy and
    enforces that snapshots precede the prediction's intended use.
    """

    def __init__(
        self, features, hero_vocab, steps=12, width=160, heads=10, layers=6, dropout=0.1, omni=True
    ):
        super().__init__()
        self.steps = steps
        half = width // 2
        self.lineup = nn.Embedding(hero_vocab, half, padding_idx=0)
        self.side_projection = nn.Linear(half * 2, half)
        self.performance = nn.Linear(features, width - half)
        self.position = nn.Embedding(steps, width)
        self.encoder = OmniEncoder(width, heads, layers, dropout, omni)
        self.head = nn.Sequential(nn.Linear(steps * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, snapshots, lineup, valid):
        if snapshots.shape[1] != self.steps or lineup.ndim != 3 or lineup.shape[1] != 2:
            raise ValueError("MMRNet requires fixed snapshot count and two hero lineups")
        rosters = self.lineup(lineup).mean(2).flatten(1)
        roster_embedding = self.side_projection(rosters)[:, None].expand(-1, self.steps, -1)
        embedded = torch.cat([roster_embedding, self.performance(snapshots)], -1)
        embedded = embedded + self.position.weight
        encoded = self.encoder(embedded, valid)
        return self.head(encoded.flatten(1)).squeeze(-1)
