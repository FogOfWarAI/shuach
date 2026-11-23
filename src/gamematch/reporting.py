# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Portable summaries of declared benchmark artifacts, without refitting."""

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .models.ratings import DEFAULT_RATING_MODELS


def _read(path):
    return json.loads(path.read_text())


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(config):
    protocol = config["protocol"]
    if protocol in {"pilot", "final", "temporal_predator_outcome"}:
        datasets = config["datasets"]
    elif protocol in {"temporal_delayed_feedback", "temporal_completed_player_history"}:
        datasets = ["overwatch"]
    elif protocol == "temporal_future_observed_rating":
        datasets = ["dota2-future-rating"]
    elif protocol == "synthetic_independent_decisions":
        datasets = config["mechanisms"]
    else:
        raise ValueError("Unsupported report protocol; original artifacts were not modified")
    if protocol == "temporal_delayed_feedback":
        models = ["calibrated_prior-pooled"] + [
            f"{model}-{'separate' if separate else 'pooled'}"
            for model in config.get("models", DEFAULT_RATING_MODELS)
            for separate in config.get("separate_domains", [False, True])
        ]
        seeds = [0]
    else:
        models, seeds = config["models"], config["seeds"]
    expected = {
        (dataset, model, seed) for dataset in datasets for model in models for seed in seeds
    }
    if not expected or len(expected) != len(datasets) * len(models) * len(seeds):
        raise ValueError("Empty or duplicated declared report matrix")
    return expected


def _summaries(grouped, expected_seeds):
    summaries = []
    for key, by_seed in sorted(grouped.items()):
        complete = set(by_seed) == set(expected_seeds)
        metrics = {}
        names = sorted(set.intersection(*(set(values) for values in by_seed.values())))
        for name in names:
            values = [by_seed[seed][name] for seed in sorted(by_seed)]
            metrics[name] = {
                "mean": float(np.mean(values)) if complete else None,
                "seed_sample_sd": float(np.std(values, ddof=1))
                if complete and len(values) > 1
                else None,
                "values": values,
            }
        summaries.append(
            {
                "identity": list(key),
                "seeds": sorted(by_seed),
                "expected_seeds": expected_seeds,
                "complete_seed_set": complete,
                "metrics": metrics,
            }
        )
    return summaries


def _scalars(metrics):
    values = {
        name: float(value)
        for name, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and name not in {"n", "positive_rate"}
    }
    if not values or not np.isfinite(list(values.values())).all():
        raise ValueError("Missing or nonfinite scalar metrics")
    return values


def create_run_report(run_directory, output=None):
    """Write summary.json/report.md in a dedicated directory; preserve run inputs.

    All six benchmark protocols are supported. Means require every declared seed.
    Standard deviations describe training variation, not confidence intervals.
    Synthetic solver rows are summarized separately from trained model jobs.
    """
    run = Path(run_directory).resolve()
    output = Path(output).resolve() if output is not None else run / "report"
    marker = run / "manifest.json"
    manifest = _read(marker)
    config, protocol = manifest["config"], manifest["config"]["protocol"]
    expected = _matrix(config)
    paths = sorted(run.rglob("metrics.json"))
    # A dedicated report directory avoids overwriting legacy results/summary files.
    if output == run or output in {path.parent for path in paths}:
        raise ValueError("Choose a dedicated report output directory, not a run or model directory")
    grouped, decisions = defaultdict(dict), defaultdict(dict)
    seen, artifacts = set(), []
    expected_decisions = (
        {
            (dataset, model, variant, solver, seed)
            for dataset, model, seed in expected
            for variant in ["raw", "temperature2", "offset1"]
            for solver in ["random", "sampled8", "exact"]
        }
        if protocol == "synthetic_independent_decisions"
        else set()
    )
    seen_decisions = set()
    for path in paths:
        result = _read(path)
        model, seed = result["model"], result.get("seed", 0)
        if protocol == "temporal_delayed_feedback":
            model += "-separate" if result.get("separate_domains") else "-pooled"
        dataset = result.get("dataset")
        if dataset is None:
            if protocol == "synthetic_independent_decisions":
                dataset = path.parent.parent.name.rsplit("-seed-", 1)[0]
            else:
                dataset = (
                    "dota2-future-rating"
                    if protocol == "temporal_future_observed_rating"
                    else "overwatch"
                )
        identity = (dataset, model, seed)
        if identity not in expected or identity in seen:
            raise ValueError(f"Unexpected or duplicated model artifact: {path}")
        seen.add(identity)
        metrics = result["metrics"].get("evaluation", result["metrics"])
        grouped[(dataset, model)][seed] = _scalars(metrics)
        artifacts.append({"path": str(path.relative_to(run)), "sha256": _hash(path)})
        decision_path = path.with_name("decision-metrics.json")
        if protocol == "synthetic_independent_decisions" and decision_path.exists():
            for row in _read(decision_path):
                if (row["mechanism"], row["model"], row["world_seed"]) != identity:
                    raise ValueError("Decision row identifies another trained scorer")
                key = (dataset, model, row["variant"], row["solver"])
                identity_decision = (*key, seed)
                if (
                    identity_decision not in expected_decisions
                    or identity_decision in seen_decisions
                ):
                    raise ValueError("Unexpected or duplicate decision factorial row")
                seen_decisions.add(identity_decision)
                decisions[key][seed] = _scalars(
                    {
                        name: row[name]
                        for name in [
                            "mean_true_imbalance",
                            "mean_decision_regret",
                            "mean_solver_gap",
                        ]
                    }
                )
            artifacts.append(
                {"path": str(decision_path.relative_to(run)), "sha256": _hash(decision_path)}
            )
    seeds = sorted({seed for _, _, seed in expected})
    summaries = _summaries(grouped, seeds)
    confidence_path = run / "decision-confidence.json"
    confidence = _read(confidence_path) if confidence_path.exists() else None
    if confidence:
        artifacts.append({"path": confidence_path.name, "sha256": _hash(confidence_path)})
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run),
        "protocol": protocol,
        "run_status": manifest["status"],
        "manifest_sha256": _hash(marker),
        "model_jobs_present": len(seen),
        "model_jobs_declared": len(expected),
        "missing_model_jobs": [list(identity) for identity in sorted(expected - seen)],
        "decision_rows_present": len(seen_decisions),
        "decision_rows_declared": len(expected_decisions),
        "missing_decision_rows": [
            list(identity) for identity in sorted(expected_decisions - seen_decisions)
        ],
        "summaries": summaries,
        "decision_summaries": _summaries(decisions, seeds),
        "decision_confidence": confidence,
        "artifacts": artifacts,
        "scope": "Read-only aggregation of saved metrics, not independent metric recomputation, retraining, checkpoint selection or hypothesis testing. Incomplete declared seed sets have no aggregate mean. Deterministic repeats are not independent evidence. Different tasks and outcome scales must not be pooled.",
    }
    lines = [
        "# GameMatch run summary",
        "",
        f"Protocol: **{protocol}**. Run status: **{manifest['status']}**.",
        f"Model artifacts: **{len(seen)} / {len(expected)}** declared jobs.",
        "",
        result["scope"],
        "",
        "Dataset / condition | Model | Seeds | Metric | Mean | Seed sample SD",
        "--- | --- | --- | --- | ---: | ---:",
    ]
    metric = "mae" if protocol == "temporal_future_observed_rating" else "log_loss"
    for row in summaries:
        value = row["metrics"].get(metric)
        if value is None:
            raise ValueError(f"Missing task metric {metric}")
        mean = "pending" if value["mean"] is None else f"{value['mean']:.6f}"
        sd = "—" if value["seed_sample_sd"] is None else f"{value['seed_sample_sd']:.6f}"
        lines.append(
            f"{' | '.join(row['identity'])} | {len(row['seeds'])}/{len(seeds)} | {metric} | {mean} | {sd}"
        )
    if decisions:
        lines += [
            "",
            "## Synthetic assignment diagnostics",
            "",
            "Condition | Model | Variant | Solver | Seeds | Mean regret | Seed sample SD",
            "--- | --- | --- | --- | --- | ---: | ---:",
        ]
        for row in result["decision_summaries"]:
            value = row["metrics"]["mean_decision_regret"]
            mean = "pending" if value["mean"] is None else f"{value['mean']:.6f}"
            sd = "—" if value["seed_sample_sd"] is None else f"{value['seed_sample_sd']:.6f}"
            lines.append(
                f"{' | '.join(row['identity'])} | {len(row['seeds'])}/{len(seeds)} | {mean} | {sd}"
            )
        lines += [
            "",
            "Solver rows reuse trained scorers and are not additional training jobs. Hidden-world regret is synthetic evidence, not a real-player outcome.",
        ]
    if confidence:
        estimated = [
            row for row in confidence.get("single_group_intervals", [])
            if row.get("status") == "estimated"
        ]
        lines += [
            "",
            "## Decision-quality intervals",
            "",
            f"Paired seed / lobby-cluster percentile bootstrap, {confidence['replicates']} replicates.",
            "",
            "Condition | Model | Variant | Solver | Seeds | Mean regret | 95% CI",
            "--- | --- | --- | --- | --- | ---: | ---:",
        ]
        for row in estimated:
            low, high = row["percentile_interval"]
            lines.append(
                f"{' | '.join(row['identity'])} | {row['n_seed_pairs']} | "
                f"{row['delta_first_minus_second']:.6f} | [{low:.6f}, {high:.6f}]"
            )
        contrasts = [c for c in confidence.get("solver_contrasts", []) if c.get("status") == "estimated"]
        if contrasts:
            lines += [
                "",
                "Exact minus random (negative favors exact):",
                "",
                "Condition | Model | Variant | Seeds | Δ regret | 95% CI",
                "--- | --- | --- | --- | ---: | ---:",
            ]
            for row in contrasts:
                low, high = row["percentile_interval"]
                lines.append(
                    f"{' | '.join(row['identity'])} | {row['n_seed_pairs']} | "
                    f"{row['delta_first_minus_second']:.6f} | [{low:.6f}, {high:.6f}]"
                )
        lines += ["", "Exploratory marginal intervals; no multiple-comparison or causal claim."]
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in [
        ("summary.json", json.dumps(result, indent=2, allow_nan=False) + "\n"),
        ("report.md", "\n".join(lines) + "\n"),
    ]:
        destination = output / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(destination)
    return output / "report.md"
