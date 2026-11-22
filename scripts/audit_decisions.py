# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Recompute assignment results from saved candidates and an explicit pair-sum oracle."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from gamematch.data.heroes import file_hash


def pool_keys(rosters):
    matrix = np.ascontiguousarray(np.sort(rosters.reshape(len(rosters), -1), axis=1))
    return matrix.view(np.dtype((np.void, matrix.dtype.itemsize * matrix.shape[1]))).ravel()


def explicit_probability(rosters, hidden):
    skill = np.asarray(hidden["skill"])
    interactions = np.asarray(hidden["interaction"])
    strengths = skill[rosters].sum(-1) / np.sqrt(5)
    if hidden["kind"] == "cooperative":
        vectors = interactions[rosters]
        pairs = np.zeros(rosters.shape[:2])
        for i, j in itertools.combinations(range(5), 2):
            pairs += (vectors[:, :, i] * vectors[:, :, j]).sum(-1)
        strengths = 0.4 * strengths + pairs / np.sqrt(10 * vectors.shape[-1])
    return expit(strengths[:, 0] - strengths[:, 1])


def audit(run):
    worlds, models = [], []
    for directory in sorted(run.glob("*-seed-*")):
        with np.load(directory / "data.npz", allow_pickle=False) as data:
            training_pools = {
                part: pool_keys(data[f"{part}_rosters"]) for part in ["train", "valid", "test"]
            }
            groups = {part: set(data[f"{part}_groups"]) for part in training_pools}
            for a, b in itertools.combinations(groups, 2):
                if groups[a] & groups[b]:
                    raise ValueError("Factual split group overlap")
        with np.load(directory / "decision-environment.npz", allow_pickle=False) as saved:
            environment = dict(saved)
        lobbies, template = environment["lobbies"], environment["candidate_template"]
        if template.shape != (252, 2, 5):
            raise ValueError("Missing ordered 5v5 candidates")
        all_left = {tuple(side) for side in template[:, 0]}
        if all_left != set(itertools.combinations(range(10), 5)):
            raise ValueError("Candidate enumeration is incomplete")
        for teams in template:
            np.testing.assert_array_equal(np.sort(teams.ravel()), np.arange(10))
        lobby_pools = pool_keys(lobbies)
        if len(np.unique(lobby_pools)) != len(lobbies):
            raise ValueError("Duplicate evaluation lobby pools")
        overlap = {
            part: len(np.intersect1d(lobby_pools, pools)) for part, pools in training_pools.items()
        }
        if any(overlap.values()):
            raise ValueError("Evaluation lobby pool occurred in a factual partition")
        hidden = json.loads((directory / "hidden-mechanism.json").read_text())
        true_probability = explicit_probability(
            lobbies[:, template].reshape(-1, 2, 5), hidden
        ).reshape(len(lobbies), -1)
        oracle_error = float(np.max(np.abs(true_probability - environment["true_probability"])))
        if oracle_error > 1e-12:
            raise ValueError("Hidden mechanism differs from explicit pair equations")
        oracle_cost = np.abs(true_probability - 0.5).min(1)
        np.testing.assert_allclose(oracle_cost, environment["oracle_cost"], atol=1e-12, rtol=0)
        sampled = environment["sampled_indices"]
        if sampled.shape != (len(lobbies), 8) or (sampled < 0).any() or (sampled >= 252).any():
            raise ValueError("Invalid sampled candidate budget")
        if any(len(np.unique(row)) != 8 for row in sampled):
            raise ValueError("Sampled candidates repeat within a lobby")
        worlds.append(
            {
                "world": directory.name,
                "lobbies": len(lobbies),
                "candidates_per_lobby": 252,
                "pool_overlap_with_factual_partitions": overlap,
                "max_oracle_probability_difference": oracle_error,
                "environment_sha256": file_hash(directory / "decision-environment.npz"),
            }
        )
        for path in sorted(directory.glob("*/decision-metrics.json")):
            with np.load(path.with_name("decisions.npz"), allow_pickle=False) as saved:
                decisions = dict(saved)
            base = decisions["base_probability"]
            if (
                base.shape != true_probability.shape
                or not np.isfinite(base).all()
                or ((base < 0) | (base > 1)).any()
            ):
                raise ValueError("Invalid saved scorer probabilities")
            errors = []
            rows = json.loads(path.read_text())
            expected = set(
                itertools.product(
                    ["raw", "temperature2", "offset1"], ["random", "sampled8", "exact"]
                )
            )
            if len(rows) != 9 or {(row["variant"], row["solver"]) for row in rows} != expected:
                raise ValueError("Incomplete or duplicated scorer/solver factorial rows")
            for row in rows:
                variant, solver = row["variant"], row["solver"]
                probabilities = base
                if variant != "raw":
                    z = logit(np.clip(base, 1e-7, 1 - 1e-7))
                    probabilities = expit(z / 2 if variant == "temperature2" else z + 1)
                costs = np.abs(probabilities - 0.5)
                if solver == "exact":
                    selected = costs.argmin(1)
                elif solver == "random":
                    selected = sampled[:, 0]
                else:
                    selected = sampled[
                        np.arange(len(lobbies)),
                        np.take_along_axis(costs, sampled, axis=1).argmin(1),
                    ]
                np.testing.assert_array_equal(selected, decisions[f"{variant}_{solver}_selected"])
                truth = np.abs(true_probability[np.arange(len(lobbies)), selected] - 0.5)
                regret = truth - oracle_cost
                np.testing.assert_allclose(
                    regret, decisions[f"{variant}_{solver}_regret"], atol=1e-12, rtol=0
                )
                predicted = costs[np.arange(len(lobbies)), selected]
                computed = {
                    "mean_true_imbalance": truth.mean(),
                    "mean_decision_regret": regret.mean(),
                    "p95_true_imbalance": np.quantile(truth, 0.95),
                    "true_probability_40_60_fraction": (truth <= 0.1).mean(),
                    "mean_predicted_imbalance": predicted.mean(),
                    "mean_solver_gap": (predicted - costs.min(1)).mean(),
                }
                errors.extend(abs(float(value) - row[key]) for key, value in computed.items())
            if max(errors) > 1e-7:
                raise ValueError("Stored decision metric mismatch")
            models.append(
                {
                    "artifact": str(path.relative_to(run)),
                    "factorial_rows": 9,
                    "max_absolute_metric_error": max(errors),
                }
            )
    return {
        "run": str(run),
        "status": "passed",
        "worlds": worlds,
        "model_checks": models,
        "audited_factorial_rows": sum(row["factorial_rows"] for row in models),
        "scope": "Explicit pair-sum oracle; full candidate coverage; separate lobby pools; every saved selection, regret and aggregate decision metric recomputed. This validates the synthetic experiment, not real-player causal effects.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/decision-environment-audit.json")
    )
    args = parser.parse_args()
    results = [audit(run) for run in args.runs]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(result["run"], result["status"], result["audited_factorial_rows"], flush=True)


if __name__ == "__main__":
    main()
