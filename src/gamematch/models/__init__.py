# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Small, independently implemented PyTorch baselines."""

from .heroes import MODEL_NAMES, make_model

__all__ = ["MODEL_NAMES", "make_model"]
