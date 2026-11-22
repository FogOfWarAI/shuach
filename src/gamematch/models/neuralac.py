# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Vectorized implementation of NeuralAC's individual/cooperation/competition equations.

Reference: NeuralAC, AAAI 2021, equations (2)--(11); local pdf/neuralac.pdf.
This implementation uses dense pair tensors rather than the author's pair-index code.
Paper bilinear attention has no bias; the author's nn.Linear attention includes one.
The selected implementation follows the paper on this point and records that difference.
"""

import torch
from torch import nn


class NeuralAC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 20,
        interaction_hidden: int = 50,
        dropout: float = 0.2,
        cooperation: bool = True,
        competition: bool = True,
        attention: bool = True,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.use_cooperation = cooperation
        self.use_competition = competition
        self.use_attention = attention
        self.individual = nn.Embedding(vocab_size, 1)
        self.cooperation = nn.Embedding(vocab_size, embedding_dim)
        self.attack = nn.Embedding(vocab_size, embedding_dim)
        self.defense = nn.Embedding(vocab_size, embedding_dim)
        self.cooperation_attention = nn.Linear(embedding_dim, embedding_dim, bias=attention_bias)
        self.competition_attention = nn.Linear(embedding_dim, embedding_dim, bias=attention_bias)
        self.cooperation_network = nn.Sequential(
            nn.Linear(embedding_dim, interaction_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_hidden, 1),
            nn.ReLU(),
        )
        self.competition_network = nn.Sequential(
            nn.Linear(embedding_dim, interaction_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_hidden, 1),
            nn.ReLU(),
        )
        for enabled, modules in [
            (cooperation, [self.cooperation, self.cooperation_network]),
            (competition, [self.attack, self.defense, self.competition_network]),
            (attention and cooperation, [self.cooperation_attention]),
            (attention and competition, [self.competition_attention]),
        ]:
            if not enabled:
                for module in modules:
                    module.requires_grad_(False)

    def pair_effect(self, left, right, network, projection, exclude_self=False):
        # Pack leading batch/side axes to avoid expensive high-rank MPS kernels.
        # Every pair, parameter and reduction retains the original semantics.
        leading_shape = left.shape[:-2]
        left = left.reshape(-1, *left.shape[-2:])
        right = right.reshape(-1, *right.shape[-2:])
        interactions = left.unsqueeze(-2) * right.unsqueeze(-3)
        effects = network(interactions.reshape(-1, interactions.shape[-1])).reshape(
            interactions.shape[:-1]
        )
        mask = (
            torch.eye(left.shape[-2], dtype=torch.bool, device=left.device)
            if exclude_self
            else None
        )
        if self.use_attention:
            projected = projection(left.reshape(-1, left.shape[-1])).reshape(left.shape)
            attention = projected @ right.transpose(-1, -2)
            if mask is not None:
                attention = attention.masked_fill(mask, -torch.inf)
            weights = attention.softmax(dim=-1)
            effects = effects * weights
        elif mask is not None:
            effects = effects.masked_fill(mask, 0)
        return effects.sum(dim=(-2, -1)).reshape(leading_shape)

    def forward(self, rosters):
        if rosters.ndim != 3 or rosters.shape[1:] != (2, 5):
            raise ValueError("The verified NeuralAC implementation accepts 5v5 rosters only")
        scores = self.individual(rosters).squeeze(-1).sum(-1)
        if self.use_cooperation:
            embeddings = self.cooperation(rosters)
            scores = scores + self.pair_effect(
                embeddings,
                embeddings,
                self.cooperation_network,
                self.cooperation_attention,
                exclude_self=True,
            )
        if self.use_competition:
            attacks = self.attack(rosters)
            defenses = self.defense(rosters.flip(1))
            scores = scores + self.pair_effect(
                attacks, defenses, self.competition_network, self.competition_attention
            )
        return scores[:, 0] - scores[:, 1]
