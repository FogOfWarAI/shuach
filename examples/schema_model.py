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
    parser.add_argument("--config", type=str, help="JSON SchemaTrainConfig (schema_training block)")
    parser.add_argument("--dim", type=int, help="embedding width (overrides config)")
    parser.add_argument("--steps", type=int, help="training steps (overrides config epochs)")
    parser.add_argument("--seed", type=int, help="random seed (overrides config)")
    args = parser.parse_args(argv)

    from gamematch.schema_config import load_schema_train_config

    overrides = {k: v for k, v in {
        "dim": args.dim, "seed": args.seed, "epochs": args.steps,
    }.items() if v is not None}
    config = load_schema_train_config(
        args.config or "examples/schema-train.json", overrides=overrides
    )

    schema = resolve(args.profile)
    from gamematch.features import make_model_from_schema

    torch.manual_seed(config.seed)
    model = make_model_from_schema(schema, dim=config.dim, seed=config.seed)
    task = schema.task
    print(f"config: epochs={config.epochs} batch={config.batch_size} lr={config.learning_rate} dim={config.dim} seed={config.seed}")

    # Contract-derived, reproducible batches -- no hand-written tensor shapes.
    train = sample_inputs(schema, batch=config.batch_size, seed=config.seed)
    test = sample_inputs(schema, batch=max(2, config.batch_size // 2), seed=config.seed + 1)

    # The loss and target shape are derived from OutputSpec -- no switch here.
    if task.criterion() == "none":
        raise SystemExit(
            f"{schema.name} declares an assignment head with no canonical loss; "
            "supply an objective explicitly (see docs/feature-schema.md)."
        )
    loss_fn = model.criterion()
    y = model.targets(train)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    print(f"schema={schema.name} task={task.kind}/{task.link} "
          f"criterion={task.criterion()} orientation={task.resolved_orientation.value} "
          f"features={len(schema.features)}")
    print(f"trainable params={sum(p.numel() for p in model.parameters()):,}")
    model.train()
    for step in range(config.epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(train), y)
        loss.backward()
        optimizer.step()
        if step == 0 or step == config.epochs - 1:
            print(f"step {step + 1:>3}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        prediction = model.predict_proba(test)
    print(f"prediction shape={tuple(prediction.shape)}, finite={bool(torch.isfinite(prediction).all())}")
    if task.kind == "binary":
        from gamematch.evaluation.binary import binary_metrics

        metrics = binary_metrics(test.targets, prediction[:, 1].cpu().numpy())
        print(f"Generated test batch: n={metrics['n']}, log loss={metrics['log_loss']:.4f}, "
              f"Brier score={metrics['brier']:.4f}")
    print("OK: schema -> model -> prediction is connected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
