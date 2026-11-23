# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Small, independently implemented PyTorch baselines.

The re-exports are lazy: importing ``gamematch.models.ratings`` for a report
must not pull in PyTorch just because this package initialiser ran. A base
install without the ``neural`` extra can therefore still inspect catalogs and
aggregate saved artifacts.
"""

__all__ = ["MODEL_NAMES", "make_model"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .heroes import MODEL_NAMES, make_model

    return {"MODEL_NAMES": MODEL_NAMES, "make_model": make_model}[name]


def __dir__():
    return sorted(set(globals()) | set(__all__))
