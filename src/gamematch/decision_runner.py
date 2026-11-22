# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Known-mechanism experiments separating prediction from assignment evaluation.

The learned scorer receives rosters only. The hidden generator is used to
sample historical labels and, separately, to evaluate unplayed assignments.
Synthetic intervention evidence must never be relabeled as real-player impact.
"""

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import expit, logit

from .artifacts import completed_fit, fit_files, seal_artifacts, verify_completions
from .configuration import runtime_identity, validate_matrix
from .data.heroes import HeroRecords, HeroVocabulary, group_ids
from .evaluation.binary import binary_metrics
from .matching.lobby import partitions
from .models import make_model
from .models.tabular import hero_counts
from .runner import TrainConfig, code_hash, fit_one, write_json


class SyntheticMechanism:
    def __init__(self, population, seed, kind):
        if kind not in {"additive", "cooperative"}:
            raise ValueError("Unknown mechanism")
        self.kind = kind
        rng = np.random.default_rng(seed + 83000)
        self.skill = rng.normal(size=population + 1)
        self.interaction = rng.normal(size=(population + 1, 4))

    def probability(self, rosters):
        strength = self.skill[rosters].sum(-1) / np.sqrt(rosters.shape[-1])
        if self.kind == "cooperative":
            vector = self.interaction[rosters]
            pair = (np.square(vector.sum(-2)).sum(-1) - np.square(vector).sum((-2, -1))) / 2
            pairs = rosters.shape[-1] * (rosters.shape[-1] - 1) / 2
            strength = 0.4 * strength + pair / np.sqrt(pairs * vector.shape[-1])
        return expit(strength[:, 0] - strength[:, 1])


def synthetic_splits(population, sizes, seed, mechanism):
    if population < 10 or any(not isinstance(n, int) or n < 1 for n in sizes.values()):
        raise ValueError("At least ten entities and positive integer partition sizes required")
    if sum(sizes.values()) > math.comb(population, 10) * 126:
        raise ValueError("Requested rows exceed the number of disjoint canonical roster groups")
    rng = np.random.default_rng(seed + 42000)
    splits, used = {}, set()
    for name, size in sizes.items():
        rosters = []
        while len(rosters) < size:
            roster = rng.choice(np.arange(1, population + 1), 10, replace=False).reshape(2, 5)
            key = tuple(sorted(tuple(sorted(team)) for team in roster))
            if key not in used:
                used.add(key)
                rosters.append(roster)
        rosters = np.asarray(rosters, dtype=np.int64)
        probability = mechanism.probability(rosters)
        labels = (rng.random(size) < probability).astype(np.float32)
        splits[name] = HeroRecords(
            rosters, labels, np.array([f"{name}:{i}" for i in range(size)]), group_ids(rosters)
        )
    return splits


def checkpoint_predictor(name, directory, vocabulary, config):
    if name == "logistic":
        model = joblib.load(directory / "checkpoint.joblib")
        return lambda roster: model.predict_proba(
            hero_counts(vocabulary.transform(roster), vocabulary.size)
        )[:, 1]
    model = make_model(name, vocabulary.size, config.embedding_dim, config.hidden_dim)
    model.load_state_dict(
        torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)["state_dict"]
    )
    model.eval()

    def predict(roster):
        encoded = vocabulary.transform(roster)
        values = []
        with torch.no_grad():
            for i in range(0, len(encoded), config.batch_size):
                values.append(
                    torch.sigmoid(
                        model(torch.from_numpy(encoded[i : i + config.batch_size]))
                    ).numpy()
                )
        return np.concatenate(values)

    return predict


def transformed_probability(probability, variant):
    if variant == "raw":
        return probability
    score = logit(np.clip(probability, 1e-7, 1 - 1e-7))
    if variant == "temperature2":
        return expit(score / 2)
    if variant == "offset1":
        return expit(score + 1)
    raise ValueError(variant)


def select_candidates(probability, solver, sampled_indices):
    cost = np.abs(probability - 0.5)
    if solver == "exact":
        return cost.argmin(1)
    if solver == "random":
        return sampled_indices[:, 0]
    if solver == "sampled8":
        selected = np.take_along_axis(cost, sampled_indices, axis=1).argmin(1)
        return sampled_indices[np.arange(len(cost)), selected]
    raise ValueError(solver)


def run_decisions(config_path, workspace, output):
    config_path, workspace, output = (Path(p).resolve() for p in (config_path, workspace, output))
    verify_completions(output)
    config = json.loads(config_path.read_text())
    if config.get("protocol") != "synthetic_independent_decisions":
        raise ValueError("Explicit synthetic_independent_decisions protocol required")
    validate_matrix(config, ("mechanisms", "models", "seeds"))
    training = TrainConfig(**config["training"])
    training.validate()
    if training.device != "cpu":
        raise ValueError("This reference decision runner declares CPU inference and training")
    identity = {"config": config, "code": code_hash(), "runtime": runtime_identity()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Incompatible existing experiment")
    else:
        manifest = {
            **identity,
            "fingerprint": fingerprint,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_type": "synthetic_known_mechanism_intervention",
            "scope": "Unconstrained fixed 5v5 lobby, full 252 ordered partitions; no queue or human causal effects",
            "replicates": "Each seed generates an independent hidden world and labels; all models/solvers within seed share data and lobbies",
            "counterfactual_evaluator": "Hidden probability generator; never the learned score or factual reused label",
            "scorer_variants": ["raw", "temperature2", "offset1"],
            "variant_purpose": "Prespecified monotonic transformations isolate probability calibration from ranking; not new trained models",
            "timing_limit": "All candidate scores cached for factorial decision analysis; these are not online solver latency measurements",
        }
        shutil.copytree(
            Path(__file__).parent,
            output / "source-snapshot/gamematch",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        write_json(output / "resolved_config.json", config)
    manifest["status"] = "running"
    write_json(path, manifest)
    try:
        for kind in config["mechanisms"]:
            for seed in config["seeds"]:
                world = output / f"{kind}-seed-{seed}"
                world.mkdir(exist_ok=True)
                mechanism = SyntheticMechanism(config["population"], seed, kind)
                splits = synthetic_splits(
                    config["population"], config["split_rows"], seed, mechanism
                )
                vocabulary = HeroVocabulary().fit(splits["train"].rosters)
                np.savez_compressed(
                    world / "data.npz",
                    **{
                        f"{s}_{f}": getattr(d, f)
                        for s, d in splits.items()
                        for f in ["rosters", "labels", "row_ids", "groups"]
                    },
                )
                write_json(
                    world / "hidden-mechanism.json",
                    {
                        "kind": kind,
                        "skill": mechanism.skill.tolist(),
                        "interaction": mechanism.interaction.tolist(),
                    },
                )
                rng = np.random.default_rng(seed + 73000)
                lobbies = np.asarray(
                    [
                        rng.choice(np.arange(1, config["population"] + 1), 10, replace=False)
                        for _ in range(config["lobbies"])
                    ]
                )
                template = np.asarray(list(partitions(range(10), 5, False)), dtype=int)
                candidates = lobbies[:, template]
                flat = candidates.reshape(-1, 2, 5)
                true_probability = mechanism.probability(flat).reshape(len(lobbies), -1)
                oracle_cost = np.abs(true_probability - 0.5).min(1)
                sampled = np.asarray([rng.choice(len(template), 8, replace=False) for _ in lobbies])
                np.savez_compressed(
                    world / "decision-environment.npz",
                    lobbies=lobbies,
                    candidate_template=template,
                    true_probability=true_probability,
                    oracle_cost=oracle_cost,
                    sampled_indices=sampled,
                )
                for name in config["models"]:
                    directory = world / name
                    required = (
                        *fit_files(name),
                        "decisions.npz",
                        "decision-metrics.json",
                        "fit-completion.json",
                    )
                    if completed_fit(directory, required, output):
                        continue
                    print(f"Training decision world={kind} seed={seed} model={name}", flush=True)
                    if not completed_fit(directory, fit_files(name), output, "fit-completion.json"):
                        result = fit_one(name, splits, vocabulary, training, seed, directory)
                        write_json(directory / "metrics.json", result)
                        seal_artifacts(directory, fit_files(name), output, "fit-completion.json")
                    predictor = checkpoint_predictor(name, directory, vocabulary, training)
                    base = predictor(flat).reshape(len(lobbies), -1)
                    test_base = predictor(splits["test"].rosters)
                    rows = []
                    saved = {"base_probability": base}
                    for variant in ["raw", "temperature2", "offset1"]:
                        probability = transformed_probability(base, variant)
                        prediction_metrics = binary_metrics(
                            splits["test"].labels, transformed_probability(test_base, variant)
                        )
                        for solver in ["random", "sampled8", "exact"]:
                            selected = select_candidates(probability, solver, sampled)
                            chosen_true = true_probability[np.arange(len(lobbies)), selected]
                            true_cost = np.abs(chosen_true - 0.5)
                            predicted_cost = np.abs(
                                probability[np.arange(len(lobbies)), selected] - 0.5
                            )
                            regret = true_cost - oracle_cost
                            saved[f"{variant}_{solver}_selected"] = selected
                            saved[f"{variant}_{solver}_regret"] = regret
                            rows.append(
                                {
                                    "mechanism": kind,
                                    "world_seed": seed,
                                    "model": name,
                                    "variant": variant,
                                    "solver": solver,
                                    "lobbies": len(lobbies),
                                    "prediction": prediction_metrics,
                                    "mean_true_imbalance": float(true_cost.mean()),
                                    "mean_decision_regret": float(regret.mean()),
                                    "p95_true_imbalance": float(np.quantile(true_cost, 0.95)),
                                    "true_probability_40_60_fraction": float(
                                        (true_cost <= 0.1).mean()
                                    ),
                                    "mean_predicted_imbalance": float(predicted_cost.mean()),
                                    "mean_solver_gap": float(
                                        (predicted_cost - np.abs(probability - 0.5).min(1)).mean()
                                    ),
                                    "candidate_score_budget": {
                                        "random": 1,
                                        "sampled8": 8,
                                        "exact": len(template),
                                    }[solver],
                                }
                            )
                    np.savez_compressed(directory / "decisions.npz", **saved)
                    write_json(directory / "decision-metrics.json", rows)
                    seal_artifacts(directory, required, output)
                    print(
                        f"  {kind}/{seed}/{name}: raw-exact regret={rows[2]['mean_decision_regret']:.6f}",
                        flush=True,
                    )
        manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, manifest)
        results = []
        for p in sorted(output.glob("*/*/decision-metrics.json")):
            results.extend(json.loads(p.read_text()))
        write_json(output / "results.json", results)
    return output / "results.json"
