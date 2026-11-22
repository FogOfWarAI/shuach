# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Ordered-logit neural likelihood with learned monotonic thresholds."""

import torch
from torch import nn
from torch.nn import functional as F

from .regularization import ActivationDropoutSequential


class OrdinalMLP(nn.Module):
    def __init__(self, features, hidden=64, classes=5, dropout=0.0):
        super().__init__()
        if classes < 2:
            raise ValueError("At least two ordered classes are required")
        self.score = ActivationDropoutSequential(
            nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, 1), dropout=dropout
        )
        self.origin = nn.Parameter(torch.tensor(-1.5))
        self.increments = nn.Parameter(torch.zeros(classes - 2))

    def thresholds(self):
        return torch.cat(
            [self.origin[None], self.origin + torch.cumsum(F.softplus(self.increments), 0)]
        )

    def forward(self, x):
        cdf = torch.sigmoid(self.thresholds()[None] - self.score(x))
        probability = torch.cat([cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]], 1)
        # Returning normalized log probabilities lets cross_entropy/NLL use the
        # same training interface while preserving ordered cumulative structure.
        probability = probability.clamp_min(1e-7)
        return torch.log(probability / probability.sum(1, keepdim=True))
