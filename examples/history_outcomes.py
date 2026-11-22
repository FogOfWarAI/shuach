# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""CPU contract/training/reload smoke example; generated data are not a benchmark."""

import io
import json

import torch

from gamematch.models.outcomes import (
    HistoryOutcomeSpec,
    build_history_outcome_model,
    validate_history_batch,
)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    results = []
    for players, labels, domains in [(5, ("loss", "win"), 1), (6, ("loss", "draw", "win"), 2)]:
        spec = HistoryOutcomeSpec(3, 4, (players, players), 3, labels, domains=domains)
        history = torch.randn(4, 2, players, 3, 3)
        static = torch.randn(4, 2, players, 4)
        mask = torch.rand(4, 2, players, 3) > 0.3
        domain = torch.arange(4) % domains
        target = torch.arange(4) % len(labels)
        audit = validate_history_batch(
            spec,
            history,
            static,
            mask,
            domain=domain,
            target=target,
            prediction_time=torch.full((4,), 100),
            history_available_at=torch.full(mask.shape, 90),
        )
        for name in ("history_deepsets", "owo", "dawn"):
            kwargs = dict(width=8, heads=2, layers=1, dropout=0.1)
            model = build_history_outcome_model(name, spec, **kwargs)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(history, static, mask, domain), target)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                probability = model(history, static, mask, domain).softmax(-1)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = build_history_outcome_model(name, spec, **kwargs).eval()
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            with torch.no_grad():
                reloaded = restored(history, static, mask, domain).softmax(-1)
            torch.testing.assert_close(probability, reloaded, atol=0, rtol=0)
            results.append(
                {
                    "model": name,
                    "players_per_side": players,
                    "outcome_labels": labels,
                    "probability_shape": list(probability.shape),
                    "loss": float(loss.detach()),
                    "reload_max_abs_error": float((probability - reloaded).abs().max()),
                    "temporal_evidence_checked": audit["temporal_evidence_checked"],
                }
            )
    print(
        json.dumps(
            {"scope": "Generated CPU fixture; no real-data performance claim", "cells": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
