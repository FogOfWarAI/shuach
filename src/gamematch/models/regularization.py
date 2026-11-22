# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Activation dropout without changing historical checkpoint parameter names."""

from torch import nn
from torch.nn import functional as F


class ActivationDropoutSequential(nn.Sequential):
    def __init__(self, *layers, dropout=0.0):
        super().__init__(*layers)
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        self.dropout_probability = float(dropout)

    def forward(self, x):
        for layer in self:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = F.dropout(x, p=self.dropout_probability, training=self.training)
        return x
