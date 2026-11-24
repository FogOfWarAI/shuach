# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""End-to-end smoke: declared feature contract -> trainable model -> prediction.

This shows the shortest path from a schema to a working model. It uses
``sample_inputs`` so it runs with no dataset; swap that call for your own
``FeatureBatch.from_natural(...)`` once an adapter is wired.

Run with the neural extra installed::

    python examples/schema_model.py --profile hero-roster
"""

from __future__ import annotations

import argparse

import torch

from gamematch.catalog import feature_schema_for
from gamematch.features import SCHEMAS, sample_inputs

# The catalog profile name and the schema key differ; allow either.
ALIASES = {
    "hero-roster": "hero_binary",
    "overwatch-history": "overwatch_history",
    "league-history": "league_history",
    "asymmetric-context": "dbd_predator",
    "overwatch-ratings": "overwatch_rating",
    "future-rating": "dota_future_rating",
}


def resolve(name: str):
    if name in ALIASES:
        return feature_schema_for(name)
    return SCHEMAS[name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="hero-roster", help="catalog profile or schema key")
    parser.add_argument("--dim", type=int, default=16, help="embedding width")
    parser.add_argument("--steps", type=int, default=20, help="training steps (demo only)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    schema = resolve(args.profile)
    from gamematch.features import make_model_from_schema

    torch.manual_seed(args.seed)
    model = make_model_from_schema(schema, dim=args.dim, seed=args.seed)
    task = schema.task

    # Contract-derived, reproducible batches -- no hand-written tensor shapes.
    train = sample_inputs(schema, batch=16, seed=args.seed)
    test = sample_inputs(schema, batch=8, seed=args.seed + 1)

    # Targets come from the batch when a POST_MATCH feature provides them,
    # otherwise from the OutputSpec declaration (most schemas have no label
    # feature -- the target is an external column in real benchmarks).
    def labels(batch):
        if batch.targets is not None:
            return batch.targets
        n = batch.batch_size
        if task.kind == "binary":
            return (torch.arange(n) % 2).float().numpy()
        if task.kind in {"multiclass", "ordinal"}:
            return (torch.arange(n) % max(2, task.dim)).numpy()
        return (torch.arange(n) % 5).astype("float32") / 5.0

    if task.kind == "binary":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        y = torch.as_tensor(labels(train), dtype=torch.float32).reshape(-1, 1)
    elif task.kind in {"multiclass", "ordinal"}:
        loss_fn = torch.nn.CrossEntropyLoss()
        y = torch.as_tensor(labels(train), dtype=torch.int64).reshape(-1)
    else:
        loss_fn = torch.nn.MSELoss()
        y = torch.as_tensor(labels(train), dtype=torch.float32).reshape(-1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    print(f"schema={schema.name} task={task.kind}/{task.link} features={len(schema.features)}")
    print(f"trainable params={sum(p.numel() for p in model.parameters()):,}")
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad()
        logits = model(train)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if step == 0 or step == args.steps - 1:
            print(f"step {step + 1:>3}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        prediction = model.predict_proba(test)
    print(f"prediction shape={tuple(prediction.shape)}, finite={bool(torch.isfinite(prediction).all())}")
    print("OK: schema -> model -> prediction is connected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
