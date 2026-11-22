# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Export auditable summaries and exploratory paired uncertainty from saved runs.

No checkpoint selection, retraining or hyperparameter changes happen here.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gamematch.evaluation.comparison import paired_cluster_bootstrap, paired_losses

RUNS = {
    "heroes": "local-full-heroes-v1",
    "history": "overwatch-history-v1",
    "ratings": "overwatch-ratings-v1",
    "ratings-delay": "overwatch-ratings-delay24h-v1",
    "asymmetric": "asymmetric-v1",
    "skill": "skill-v1",
    "heroes-ablation": "neuralac-ablations-v1",
    "history-target-only": "overwatch-target-only-v1",
    "heroes-convergence": "neuralac-convergence-v1",
    "history-earlier-origin": "overwatch-history-earlier-origin-v1",
    "ratings-earlier-origin": "overwatch-ratings-earlier-origin-v1",
}


def read(path):
    return json.loads(path.read_text())


def scalar_metrics(result):
    values = result["metrics"].get("evaluation", result["metrics"])
    return {
        k: v
        for k, v in values.items()
        if isinstance(v, (int, float)) and k not in {"n", "positive_rate"}
    }


def aggregate(values):
    return {
        "mean": float(np.mean(values)),
        "seed_sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "values": values,
    }


def collect_decisions(root, run_name, task, status):
    run = root / "runs" / run_name
    if not (run / "manifest.json").exists():
        return []
    manifest = read(run / "manifest.json")
    config = manifest["config"]
    status[task] = {
        "run": str(run.relative_to(root)),
        "status": manifest["status"],
        "jobs": len(list(run.rglob("metrics.json"))),
        "expected_jobs": len(config["models"]) * len(config["seeds"]) * len(config["mechanisms"]),
    }
    grouped = defaultdict(list)
    for path in sorted(run.rglob("decision-metrics.json")):
        for row in read(path):
            grouped[(row["mechanism"], row["model"], row["variant"], row["solver"])].append(row)
    summaries = []
    for (mechanism, model, variant, solver), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["world_seed"])
        metrics = {
            metric: aggregate([r[metric] for r in rows])
            for metric in [
                "mean_true_imbalance",
                "mean_decision_regret",
                "mean_solver_gap",
                "true_probability_40_60_fraction",
            ]
        }
        metrics.update(
            {
                metric: aggregate([r["prediction"][metric] for r in rows])
                for metric in ["auc", "log_loss", "brier"]
            }
        )
        seeds = [row["world_seed"] for row in rows]
        summaries.append(
            {
                "mechanism": mechanism,
                "model": model,
                "variant": variant,
                "solver": solver,
                "world_seeds": seeds,
                "complete_seed_set": seeds == sorted(config["seeds"]),
                "metrics": metrics,
            }
        )
    return summaries


def convergence_curves(root):
    curves = []
    for run_name, primary_cap in [
        ("neuralac-convergence-v1", 50),
        ("decisions-convergence-v1", 25),
    ]:
        run = root / "runs" / run_name
        if (
            not (run / "manifest.json").exists()
            or read(run / "manifest.json")["status"] != "completed"
        ):
            continue
        for path in sorted(run.rglob("metrics.json")):
            result = read(path)
            history = path.with_name("history.json")
            setting = result.get("dataset", path.parent.parent.name.split("-seed-")[0])
            curves.append(
                {
                    "run": run_name,
                    "setting": setting,
                    "seed": result["seed"],
                    "primary_cap": primary_cap,
                    "best_epoch": result["best_epoch"],
                    "history": read(history),
                    "history_path": str(history.relative_to(root)),
                    "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
                }
            )
    return curves


def collect(root):
    groups = defaultdict(dict)
    status = {}
    for task, name in RUNS.items():
        run = root / "runs" / name
        if not (run / "manifest.json").exists():
            continue
        manifest = read(run / "manifest.json")
        files = sorted(run.rglob("metrics.json"))
        status[task] = {
            "run": str(run.relative_to(root)),
            "status": manifest["status"],
            "jobs": len(files),
        }
        cfg = manifest["config"]
        status[task]["declared_seeds"] = [0] if task.startswith("ratings") else sorted(cfg["seeds"])
        status[task]["expected_jobs"] = (
            1 + len(cfg["models"]) * len(cfg["separate_domains"])
            if task.startswith("ratings")
            else len(cfg["models"]) * len(cfg["seeds"]) * len(cfg.get("datasets", [None]))
        )
        for path in files:
            result = read(path)
            dataset = result.get(
                "dataset", "dota2-future-rating" if task == "skill" else "overwatch"
            )
            model = result["model"]
            if task.startswith("ratings"):
                model += "-separate" if result.get("separate_domains") else "-pooled"
            key = (task, dataset, model)
            seed = result.get("seed", 0)
            if seed in groups[key]:
                raise ValueError(f"Duplicate seed artifact: {path}")
            groups[key][seed] = (path, result)
    return groups, status


def compare(first, second, cluster_key, metric, replicates):
    seeds = sorted(set(first) & set(second))
    # Deterministic rating baselines can be paired with every neural seed.
    if len(second) == 1 and len(first) > 1:
        seeds = sorted(first)
        second = {s: next(iter(second.values())) for s in seeds}
    paths_a = [first[s][0].with_name("predictions.npz") for s in seeds]
    paths_b = [second[s][0].with_name("predictions.npz") for s in seeds]
    differences, reference = paired_losses(paths_a, paths_b, metric=metric)
    if cluster_key == "calendar_day":
        clusters = np.floor(reference["time"] / 86400).astype(np.int64)
    else:
        clusters = reference[cluster_key]
    result = paired_cluster_bootstrap(
        differences, clusters, replicates=replicates, seed=20261114, metric=metric
    )
    result["cluster_unit"] = cluster_key
    result["seed_ids"] = seeds
    result["few_cluster_warning"] = result["n_clusters"] < 20
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--replicates", type=int, default=2500)
    args = parser.parse_args()
    root = args.workspace.resolve()
    groups, status = collect(root)
    summaries, comparisons = [], []
    for (task, dataset, model), records in sorted(groups.items()):
        values = defaultdict(list)
        for _, result in records.values():
            for metric, value in scalar_metrics(result).items():
                values[metric].append(value)
        declared_seeds = status[task]["declared_seeds"]
        expected = len(declared_seeds)
        complete = sorted(records) == declared_seeds
        item = {
            "task": task,
            "dataset": dataset,
            "model": model,
            "seeds": sorted(records),
            "expected_seed_count": expected,
            "complete_seed_set": complete,
            "metrics": {k: aggregate(v) for k, v in values.items()},
            "artifacts": [str(p.relative_to(root)) for p, _ in records.values()],
        }
        summaries.append(item)
        if not complete:
            continue
        baseline = "current" if task == "skill" else "logistic"
        units = ["group_id"] if task.startswith("heroes") else ["calendar_day"]
        if task == "asymmetric":
            units.append("player_id")
        if task == "skill":
            units = ["player_id", "match_id"]
        if task in {"history", "history-earlier-origin"}:
            baseline = "history_deepsets"
        if task.startswith("ratings"):
            baseline = "calibrated_prior-pooled"
        baseline_key = (task, dataset, baseline)
        if task in {"heroes-ablation", "heroes-convergence"}:
            baseline = "neuralac"
            baseline_key = ("heroes", dataset, baseline)
        if task == "history-target-only":
            baseline = model
            baseline_key = ("history", dataset, baseline)
        if (
            (task, dataset, model) == baseline_key
            or baseline_key not in groups
            or len(groups[baseline_key]) != expected
        ):
            continue
        for unit in units:
            comparisons.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "first": model,
                    "second": baseline
                    + (" [cross-domain]" if task == "history-target-only" else ""),
                    **compare(
                        records,
                        groups[baseline_key],
                        unit,
                        "absolute_error" if task == "skill" else "log_loss",
                        args.replicates,
                    ),
                }
            )
    # Same Overwatch outcomes and availability contract, unlike a cross-dataset rank.
    for history_task, ratings_task, comparison_task in [
        ("history", "ratings", "history-vs-prior"),
        ("history-earlier-origin", "ratings-earlier-origin", "earlier-origin-vs-prior"),
    ]:
        base = groups.get((ratings_task, "overwatch", "calibrated_prior-pooled"))
        if not base:
            continue
        for model in ["static_deepsets", "history_deepsets", "owo_transformer", "owo", "dawn"]:
            records = groups.get((history_task, "overwatch", model), {})
            if len(records) == 5:
                comparisons.append(
                    {
                        "task": comparison_task,
                        "dataset": "overwatch",
                        "first": model,
                        "second": "calibrated_prior-pooled",
                        **compare(records, base, "calendar_day", "log_loss", args.replicates),
                    }
                )
    for model in ["elo", "trueskill", "openskill"]:
        for mode in ["pooled", "separate"]:
            name = f"{model}-{mode}"
            a = groups.get(("ratings-delay", "overwatch", name))
            b = groups.get(("ratings", "overwatch", name))
            if a and b:
                comparisons.append(
                    {
                        "task": "feedback-delay",
                        "dataset": "overwatch",
                        "first": f"{name}+24h",
                        "second": name,
                        **compare(a, b, "calendar_day", "log_loss", args.replicates),
                    }
                )
    # Exploratory same-cohort comparison with every rating family/domain mode;
    # the simple learned history baseline is not compared only with a weak prior.
    for history_task, ratings_task, comparison_task in [
        ("history", "ratings", "history-vs-ratings"),
        ("history-earlier-origin", "ratings-earlier-origin", "earlier-origin-vs-ratings"),
    ]:
        records = groups.get((history_task, "overwatch", "history_deepsets"), {})
        if not records or sorted(records) != status[history_task]["declared_seeds"]:
            continue
        for rating in ["elo", "trueskill", "openskill"]:
            for mode in ["pooled", "separate"]:
                name = f"{rating}-{mode}"
                baseline = groups.get((ratings_task, "overwatch", name))
                if not baseline:
                    continue
                comparisons.append(
                    {
                        "task": comparison_task,
                        "dataset": "overwatch",
                        "first": "history_deepsets",
                        "second": name,
                        "scope": "Exploratory post-hoc comparison with all six rating configurations; same test rows and availability contract, but different representations and input features",
                        **compare(records, baseline, "calendar_day", "log_loss", args.replicates),
                    }
                )
    decisions = collect_decisions(root, "decisions-v1", "decisions", status)
    decision_convergence = collect_decisions(
        root, "decisions-convergence-v1", "decisions-convergence", status
    )
    calibration = None
    calibration_run = root / "runs/hero-calibration-v1"
    if (calibration_run / "manifest.json").exists():
        manifest = read(calibration_run / "manifest.json")
        config = manifest["config"]
        status["hero-calibration"] = {
            "run": str(calibration_run.relative_to(root)),
            "status": manifest["status"],
            "jobs": len(list(calibration_run.rglob("metrics.json"))),
            "expected_jobs": len(config["datasets"])
            * len(config["models"])
            * len(config["seeds"])
            * len(config["variants"]),
            "unit": "calibration fits",
        }
        if manifest["status"] == "completed":
            calibration = read(calibration_run / "summary.json")
    cross_calibration_path = root / "reports/calibration-cross-model.json"
    cross_calibration = read(cross_calibration_path) if cross_calibration_path.exists() else None
    if cross_calibration and (
        calibration is None
        or cross_calibration["calibration_manifest_sha256"]
        != hashlib.sha256((calibration_run / "manifest.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Cross-model calibration analysis belongs to another calibration run")
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary_scope": "All currently completed jobs; incomplete seed sets explicitly labeled and excluded from paired comparisons",
        "uncertainty": "Mean and sample SD across training seeds. Exploratory marginal paired percentile intervals resample seeds and one cluster dimension, conditional on recorded training/validation samples and fitted predictors; models and calibrators are not refitted inside bootstrap replicates. Repeated players and time drift are not jointly removed. No multiple-comparison correction or population-wide superiority claim.",
        "skill_limitation": "942 test records span less than one day; player and match bootstrap sensitivities cannot establish temporal generalization. Only 21 limited-observed-history test examples; not verified novices.",
        "decision_limitation": "Five independently generated hidden worlds per mechanism; the same seed also controls training and is shared across mechanism families. Mean/SD are across worlds, not 2500 independent world draws.",
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "decision_summaries": decisions,
        "decision_convergence_summaries": decision_convergence,
        "hero_calibration": calibration,
        "calibration_cross_model": cross_calibration,
        "convergence_curves": convergence_curves(root),
        "cohorts": read(root / "reports/cohort-analysis.json")
        if (root / "reports/cohort-analysis.json").exists()
        else None,
        "software_performance": {
            name: read(root / "reports" / filename)
            for name, filename in [
                ("search", "search-performance.json"),
                ("neuralac-cpu", "neuralac-performance-cpu.json"),
                ("neuralac-mps", "neuralac-performance-mps.json"),
            ]
            if (root / "reports" / filename).exists()
        },
    }
    out = root / "reports/analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Local experiment analysis",
        "",
        payload["uncertainty"],
        "",
        payload["skill_limitation"],
        "",
        "Incomplete declared seed sets are marked pending in this report and excluded from paired comparisons and manuscript tables. JSON retains explicitly flagged intermediate summaries for audit only.",
        "",
        "## Prediction metrics",
        "",
        "Task | Dataset | Model | Seeds | Metric | Mean | Seed SD",
        "--- | --- | --- | ---: | --- | ---: | ---:",
    ]
    for row in summaries:
        metric = "mae" if row["task"] == "skill" else "log_loss"
        v = row["metrics"][metric]
        sd = "—" if v["seed_sample_sd"] is None else f"{v['seed_sample_sd']:.6f}"
        mean = f"{v['mean']:.6f}" if row["complete_seed_set"] else "pending"
        if not row["complete_seed_set"]:
            sd = "—"
        n = f"{len(row['seeds'])}/{row['expected_seed_count']}"
        lines.append(
            f"{row['task']} | {row['dataset']} | {row['model']} | {n} | {metric} | {mean} | {sd}"
        )
    lines.extend(
        [
            "",
            "## Paired exploratory intervals",
            "",
            "Negative differences favor the first model. These are not adjusted hypothesis tests; fewer than 20 clusters should be treated cautiously.",
            "",
            "Task/dataset | First − second | Cluster | Clusters | Difference | 95% interval",
            "--- | --- | --- | ---: | ---: | ---",
        ]
    )
    for c in comparisons:
        lo, hi = c["percentile_interval"]
        lines.append(
            f"{c['task']}/{c['dataset']} | {c['first']} − {c['second']} | {c['cluster_unit']} | {c['n_clusters']} | {c['delta_first_minus_second']:.6f} | [{lo:.6f}, {hi:.6f}]"
        )
    lines.extend(
        [
            "",
            "## Independent synthetic decisions",
            "",
            payload["decision_limitation"],
            "",
            "World | Model | Probability transform | Solver | NLL | AUC | Decision regret | World SD",
            "--- | --- | --- | --- | ---: | ---: | ---: | ---:",
        ]
    )
    for d in decisions:
        if not d["complete_seed_set"]:
            continue
        m = d["metrics"]
        lines.append(
            f"{d['mechanism']} | {d['model']} | {d['variant']} | {d['solver']} | {m['log_loss']['mean']:.4f} | {m['auc']['mean']:.4f} | {m['mean_decision_regret']['mean']:.4f} | {m['mean_decision_regret']['seed_sample_sd']:.4f}"
        )
    if decision_convergence:
        lines.extend(
            [
                "",
                "## Exploratory synthetic NeuralAC budget extension",
                "",
                "Primary 25-epoch-cap results remain above. These runs allow up to 200 epochs and are not a matched-budget comparison with other primary models.",
                "",
                "World | Variant | Solver | Seeds | NLL | Decision regret | World SD",
                "--- | --- | --- | ---: | ---: | ---: | ---:",
            ]
        )
        for row in decision_convergence:
            if not row["complete_seed_set"]:
                continue
            m = row["metrics"]
            lines.append(
                f"{row['mechanism']} | {row['variant']} | {row['solver']} | {len(row['world_seeds'])}/5 | {m['log_loss']['mean']:.5f} | {m['mean_decision_regret']['mean']:.5f} | {m['mean_decision_regret']['seed_sample_sd']:.5f}"
            )
    if calibration:
        lines.extend(
            [
                "",
                "## Validation-only hero calibration sensitivity",
                "",
                calibration["config"]["scope"],
                "",
                "Dataset | Model | Calibration | NLL | Seed SD",
                "--- | --- | --- | ---: | ---:",
            ]
        )
        for row in calibration["summaries"]:
            m = row["metrics"]["log_loss"]
            lines.append(
                f"{row['dataset']} | {row['model']} | {row['variant']} | {m['mean']:.5f} | {m['seed_sample_sd']:.5f}"
            )
    if cross_calibration:
        lines.extend(
            [
                "",
                "## Exploratory common-calibration comparisons",
                "",
                cross_calibration["scope"],
                "",
                "Dataset | First model | Calibration (both models) | NLL difference from logistic | 95% interval",
                "--- | --- | --- | ---: | ---",
            ]
        )
        for row in cross_calibration["comparisons"]:
            lo, hi = row["percentile_interval"]
            lines.append(
                f"{row['dataset']} | {row['first']} | {row['variant']} | {row['delta_first_minus_second']:.6f} | [{lo:.6f}, {hi:.6f}]"
            )
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "summaries": len(summaries),
                "comparisons": len(comparisons),
                "decisions": len(decisions),
                "status": status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
