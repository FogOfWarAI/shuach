# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Guards for fair multi-model comparisons.

A benchmark only compares models fairly if they all see the same input:
one dataset split, one vocabulary (including the unknown bucket), one feature
contract and one set of timing rules. Most of this is already true by
construction in the native runners -- `prepare()` returns a single
``(splits, vocabulary)`` pair that every model reuses. The gap is that it was
never *declared*, so a future change could silently break it.

This module makes the invariant explicit and auditable. It is intentionally
independent of any runner: call it once before a model matrix starts.
"""

from __future__ import annotations

import hashlib
import json

__all__ = ["input_contract", "assert_comparable"]


def _stable_hash(value) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def input_contract(*, dataset, split_fingerprint, vocabulary, schema_hash=None,
                   timing_rule="available_at < prediction_time", entities=()):
    """Describe the input all compared models must share.

    Args:
        dataset: Dataset identifier (e.g. ``uci-dota2``).
        split_fingerprint: Hash of the train/valid/test row and group split.
        vocabulary: Fitted vocabulary with ``size`` and ``values`` attributes.
        schema_hash: Optional ``FeatureSchema.content_hash()``. Native runners
            without a feature contract may omit it.
        timing_rule: The availability predicate used by the whole run.
        entities: Sorted entity IDs whose features are combined (e.g. heroes).
    """
    vocabulary_size = vocabulary.size if hasattr(vocabulary, "size") else int(vocabulary)
    values = getattr(vocabulary, "values", None)
    values_hash = _stable_hash([str(v) for v in values]) if values is not None else None
    return {
        "dataset": str(dataset),
        "split_fingerprint": str(split_fingerprint),
        "vocabulary_size": vocabulary_size,
        "vocabulary_values_sha256": values_hash,
        "schema_sha256": schema_hash,
        "timing_rule": timing_rule,
        "entities": sorted(str(e) for e in entities),
    }


def assert_comparable(contracts, models):
    """Reject a model matrix whose members would see different inputs.

    Args:
        contracts: Mapping ``model -> input_contract()`` result.
        models: The model IDs in the run matrix.

    Returns:
        The single shared contract, with ``models`` attached for artifacts.

    Raises:
        ValueError: If a model is missing a contract or any contract differs
            in dataset, split, vocabulary, schema or timing rule.
    """
    if not models:
        raise ValueError("A comparison needs at least one model")
    keys = ("dataset", "split_fingerprint", "vocabulary_size",
            "vocabulary_values_sha256", "schema_sha256", "timing_rule", "entities")
    first = None
    for model in models:
        if model not in contracts:
            raise ValueError(f"missing input contract for model {model}")
        contract = contracts[model]
        if first is None:
            first = contract
            continue
        differing = [k for k in keys if contract.get(k) != first.get(k)]
        if differing:
            raise ValueError(
                f"models do not share the same input ({', '.join(differing)}); "
                "comparison would be invalid"
            )
    return {**first, "models": sorted(set(models))}
