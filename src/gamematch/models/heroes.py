# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""All models consume [batch, side, member] hero indices and return logits.

Sides remain ordered, while members within each side are exchangeable.
These are baseline implementations, not copies of third-party paper code.
"""

import torch
from torch import nn

from .neuralac import NeuralAC
from .regularization import ActivationDropoutSequential

NEURALAC_VARIANTS = {
    "neuralac": {},
    "neuralac_no_cooperation": {"cooperation": False},
    "neuralac_no_competition": {"competition": False},
    "neuralac_no_attention": {"attention": False},
    "neuralac_author_bias": {"attention_bias": True},
}
MODEL_NAMES = (
    "prior",
    "linear",
    "signed_hero",
    "logistic",
    "fm",
    "lightgbm",
    "mlp",
    "deepsets",
    "set_transformer",
    *NEURALAC_VARIANTS,
)


def side_counts(rosters: torch.Tensor, vocab_size: int) -> torch.Tensor:
    out = torch.zeros((*rosters.shape[:2], vocab_size), device=rosters.device)
    out.scatter_add_(2, rosters, torch.ones_like(rosters, dtype=out.dtype))
    return out.flatten(1)


class Linear(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.head = nn.Linear(2 * vocab_size, 1)

    def forward(self, rosters):
        return self.head(side_counts(rosters, self.vocab_size)).squeeze(-1)


class SignedHero(nn.Module):
    """Add each left hero's log-odds contribution and subtract each right hero's.

    The learned intercept represents the left-side advantage. Both sides share
    the same per-hero weight and the training vocabulary's unknown ID contributes zero.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.contribution = nn.Embedding(vocab_size, 1, padding_idx=0)
        nn.init.zeros_(self.contribution.weight)
        self.side_bias = nn.Parameter(torch.zeros(()))

    def forward(self, rosters):
        totals = self.contribution(rosters).squeeze(-1).sum(dim=-1)
        return self.side_bias + totals[:, 0] - totals[:, 1]


class RosterSetTransformer(nn.Module):
    """Encode each team as a set and pool its five members with a learned query.

    Both sides share the attention weights. The final layer receives the two
    pooled teams in side order so it can learn a left-side advantage.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()
        # Each of the two attention heads has embedding_dim channels.
        width = 2 * embedding_dim
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.encoder = nn.TransformerEncoderLayer(
            width, nhead=2, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.seed = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.seed, std=0.02)
        self.pool = nn.MultiheadAttention(width, 2, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(2 * width, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, rosters):
        # Treat each team as a separate set until the side-ordered readout.
        members = self.encoder(self.embedding(rosters.flatten(0, 1)))
        query = self.seed.expand(members.shape[0], -1, -1)
        pooled, _ = self.pool(query, members, members, need_weights=False)
        teams = self.norm(query + pooled).reshape(rosters.shape[0], -1)
        return self.head(teams).squeeze(-1)


class FactorizationMachine(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(2 * vocab_size, 1)
        self.factors = nn.Parameter(torch.empty(2 * vocab_size, embedding_dim))
        nn.init.normal_(self.factors, std=0.01)

    def forward(self, rosters):
        x = side_counts(rosters, self.vocab_size)
        pairwise = 0.5 * ((x @ self.factors).square() - x.square() @ self.factors.square()).sum(-1)
        return self.linear(x).squeeze(-1) + pairwise


class MLP(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.head = ActivationDropoutSequential(
            nn.Linear(2 * vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            dropout=dropout,
        )

    def forward(self, rosters):
        return self.head(side_counts(rosters, self.vocab_size)).squeeze(-1)


class DeepSets(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.phi = ActivationDropoutSequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            dropout=dropout,
        )
        self.rho = ActivationDropoutSequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            dropout=dropout,
        )

    def forward(self, rosters):
        sides = self.phi(self.embedding(rosters)).mean(dim=2)
        return self.rho(sides.flatten(1)).squeeze(-1)


def make_model(
    name: str,
    vocab_size: int,
    embedding_dim: int = 20,
    hidden_dim: int = 64,
    dropout: float | None = None,
    interaction_hidden: int = 50,
):
    """Construct a binary outcome network for encoded hero rosters.

    Args:
        name: Neural roster model ID such as signed_hero, set_transformer or neuralac.
        vocab_size: Size of the fitted vocabulary, including the unknown row.
        embedding_dim: Hero embedding width where applicable. For set_transformer
            this is the width per attention head, with two heads per team.
        hidden_dim: Hidden width for MLP and Deep Sets. For set_transformer it
            sets the encoder feed-forward width and the team readout's hidden width.
        dropout: Activation dropout where supported. None selects the model default.
        interaction_hidden: Width of the NeuralAC interaction networks.

    Returns:
        A PyTorch module mapping int64 rosters [batch, 2, 5] to left-side
        win logits [batch]. Apply sigmoid to obtain probabilities.

    Raises:
        ValueError: The ID has no neural factory implementation.
    """
    if name == "linear":
        return Linear(vocab_size)
    if name == "signed_hero":
        return SignedHero(vocab_size)
    if name == "set_transformer":
        return RosterSetTransformer(vocab_size, embedding_dim, hidden_dim, dropout or 0.0)
    if name == "fm":
        return FactorizationMachine(vocab_size, embedding_dim)
    if name == "mlp":
        return MLP(vocab_size, hidden_dim, dropout=dropout or 0.0)
    if name == "deepsets":
        return DeepSets(vocab_size, embedding_dim, hidden_dim, dropout=dropout or 0.0)
    if name in NEURALAC_VARIANTS:
        return NeuralAC(
            vocab_size,
            embedding_dim,
            interaction_hidden=interaction_hidden,
            dropout=0.2 if dropout is None else dropout,
            **NEURALAC_VARIANTS[name],
        )
    raise ValueError(f"No neural model named {name}")
