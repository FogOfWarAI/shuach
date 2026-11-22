# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Build enriched histories and run a validation-only linear feature ablation."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gamematch.data.events import load_overwatch
from gamematch.data.heroes import file_hash
from gamematch.data.history import HistoryData, history_split_masks
from gamematch.data.rich_history import (
    RICH_HISTORY_FEATURES,
    RICH_STATIC_FEATURES,
    build_rich_histories,
)
from gamematch.evaluation.multiclass import multiclass_metrics
from gamematch.models.tabular import require_logistic_convergence

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/richer-history"


def team_features(data, history_width, static_width):
    count = data.valid.sum(-1, keepdims=True)
    average = (data.history[..., :history_width] * data.valid[..., None]).sum(-2)
    average /= np.maximum(count, 1)
    player = np.concatenate(
        [data.static[..., :static_width], average, count / data.valid.shape[-1]], axis=-1
    )
    # Side-preserving, permutation-invariant, equal aggregation for each ablation.
    return np.concatenate(
        [
            player.mean(2).reshape(len(player), -1),
            player.std(2).reshape(len(player), -1),
            data.domain[:, None],
        ],
        axis=-1,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "datasets/raw/overwatch.jsonl"
    identity = {
        str(p.relative_to(ROOT)): file_hash(p)
        for p in [
            source,
            ROOT / "src/gamematch/data/events.py",
            ROOT / "src/gamematch/data/history.py",
            ROOT / "src/gamematch/data/rich_history.py",
            Path(__file__),
        ]
    }
    cache = ROOT / ".cache/rich-history-v1.npz"
    cache_manifest = cache.with_suffix(".json")
    if cache.exists() and cache_manifest.exists():
        saved = json.loads(cache_manifest.read_text())
        if saved["identity"] != identity or file_hash(cache) != saved["sha256"]:
            raise ValueError("Enrichment cache identity changed; choose a new cache path")
        data = HistoryData.load(cache)
    else:
        events, manifest = load_overwatch(source)
        print("BUILD", len(events), "events", flush=True)
        data = build_rich_histories(events)
        data.save(cache)
        cache_manifest.write_text(
            json.dumps(
                {"identity": identity, "sha256": file_hash(cache), "adapter": manifest}, indent=2
            )
            + "\n"
        )
    masks, bounds = history_split_masks(data)
    coverage = {}
    for split, mask in masks.items():
        count = np.rint(np.expm1(data.static[mask, ..., 0])).astype(int)
        coverage[split] = {
            "matches": int(mask.sum()),
            "player_appearance_zero_history_fraction": float((count == 0).mean()),
            "player_appearance_at_least_5_fraction": float((count >= 5).mean()),
            "player_appearance_at_least_10_fraction": float((count >= 10).mean()),
            "all_players_at_least_1_matches": int((count >= 1).all((1, 2)).sum()),
            "all_players_at_least_5_matches": int((count >= 5).all((1, 2)).sum()),
            "count_quantiles": np.quantile(count, [0, 0.25, 0.5, 0.75, 0.9, 0.99]).tolist(),
        }
    protocol = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "history_features": RICH_HISTORY_FEATURES,
        "static_features": RICH_STATIC_FEATURES,
        "availability": "strict delivered feedback available_at < prediction time",
        "coverage": coverage,
        "split_boundaries": bounds,
        "experiment": "validation-only logistic ablation; not an OwO/DAWN performance claim",
        "C": [0.01, 0.1, 1.0],
        "selection_metric": "validation log_loss",
        "test_predictions_evaluated": False,
        "state": "fitting",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = []
    for name, hwidth, swidth in [
        ("original", 5, 13),
        ("richer_sequence", 8, 13),
        ("richer_static", 5, 46),
        ("richer_both", 8, 46),
    ]:
        features = team_features(data, hwidth, swidth)
        if not np.isfinite(features).all():
            raise ValueError("Non-finite enriched features")
        candidates = []
        for c in protocol["C"]:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
            model.fit(features[masks["train"]], data.outcome[masks["train"]])
            iterations = require_logistic_convergence(model[-1])
            p = model.predict_proba(features[masks["valid"]])
            y = data.outcome[masks["valid"]]
            predicted = p.argmax(1)
            metrics = multiclass_metrics(y, p)
            metrics.update(
                {
                    "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
                    "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
                }
            )
            candidates.append(
                {
                    "C": c,
                    "iterations": iterations,
                    "metrics": metrics,
                    "predicted_class_counts": np.bincount(predicted, minlength=3).tolist(),
                }
            )
            print(name, c, metrics["log_loss"], metrics["accuracy"], flush=True)
        selected = min(candidates, key=lambda r: (r["metrics"]["log_loss"], r["C"]))
        results.append(
            {
                "features": name,
                "width": int(features.shape[-1]),
                "candidates": candidates,
                "selected": selected,
            }
        )
    protocol.update(state="completed", ablations=results)
    (OUT / "validation-ablation.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("DONE: validation only; test metrics were not computed", flush=True)


if __name__ == "__main__":
    main()
