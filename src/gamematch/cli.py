# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Command-line entry points for offline prediction and assignment experiments."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np

from . import __version__
from .data.acquisition import REGISTRY, fetch_dataset
from .data.heroes import SOURCES, prepare
from .matching.lobby import exact_balance, local_swap_balance, partitions
from .reporting import create_run_report


def main(argv=None):
    """Dispatch a console command using the arguments after the program name."""
    parser = argparse.ArgumentParser(prog="gamematch")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Report training devices and optional dependency imports"
    )
    commands.add_parser("datasets", help="List pinned sources, sizes and separate data licenses")
    models = commands.add_parser("models", help="List model IDs, tasks and input contracts")
    models.add_argument("--task", choices=["outcome_prediction", "future_observed_rating"])
    commands.add_parser("input-profiles", help="List input contracts and adapter boundaries")
    schema = commands.add_parser(
        "feature-schema", help="Inspect an executable input/output feature contract"
    )
    schema.add_argument("--profile", required=True)
    check = commands.add_parser("check-models", help="Check input and runner compatibility")
    check.add_argument("--profile", required=True)
    check.add_argument("--models", nargs="+", required=True)
    outcome = commands.add_parser("outcome-benchmark", help="Run a compatible outcome protocol")
    outcome.add_argument("--config", type=Path, required=True)
    outcome.add_argument("--workspace", type=Path, default=Path.cwd())
    outcome.add_argument("--output", type=Path)
    outcome.add_argument("--dry-run", action="store_true", help="Validate without loading data")
    fetch = commands.add_parser("fetch-data", help="Download and verify a public dataset")
    fetch.add_argument("dataset", choices=sorted(REGISTRY))
    fetch.add_argument("--data-root", type=Path, default=Path("datasets"))
    inspect = commands.add_parser("inspect", help="Inspect supported static hero data")
    inspect.add_argument("--dataset", choices=list(SOURCES), required=True)
    inspect.add_argument("--workspace", type=Path, default=Path.cwd())
    benchmark = commands.add_parser(
        "benchmark", help="Train and evaluate the declared experiment matrix"
    )
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--workspace", type=Path, default=Path.cwd())
    benchmark.add_argument("--output", type=Path, required=True)
    rating = commands.add_parser(
        "rating-benchmark", help="Run temporal player-rating experiments with delayed feedback"
    )
    rating.add_argument("--config", type=Path, required=True)
    rating.add_argument("--workspace", type=Path, default=Path.cwd())
    rating.add_argument("--output", type=Path, required=True)
    history = commands.add_parser(
        "history-benchmark", help="Train public-data OwO/DAWN and history baselines"
    )
    history.add_argument("--config", type=Path, required=True)
    history.add_argument("--workspace", type=Path, default=Path.cwd())
    history.add_argument("--output", type=Path, required=True)
    asymmetric = commands.add_parser(
        "asymmetric-benchmark", help="Run temporal five-outcome predator prediction"
    )
    asymmetric.add_argument("--config", type=Path, required=True)
    asymmetric.add_argument("--workspace", type=Path, default=Path.cwd())
    asymmetric.add_argument("--output", type=Path, required=True)
    decision = commands.add_parser(
        "decision-benchmark",
        help="Evaluate learned scorers and solvers with independent synthetic mechanisms",
    )
    decision.add_argument("--config", type=Path, required=True)
    decision.add_argument("--workspace", type=Path, default=Path.cwd())
    decision.add_argument("--output", type=Path, required=True)
    skill = commands.add_parser(
        "skill-benchmark", help="Predict a future observed rating from completed-game snapshots"
    )
    skill.add_argument("--config", type=Path, required=True)
    skill.add_argument("--workspace", type=Path, default=Path.cwd())
    skill.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report", help="Summarize any benchmark track without changing run artifacts")
    report.add_argument("run_directory", type=Path)
    report.add_argument("--output", type=Path, help="Dedicated report directory; default RUN/report")
    schema_train = commands.add_parser(
        "schema-train", help="Validate a SchemaTrainConfig JSON without starting training"
    )
    schema_train.add_argument("--config", type=Path, required=True)
    demo = commands.add_parser(
        "lobby-demo", help="Synthetic exact-search example with an independent evaluator"
    )
    demo.add_argument("--seed", type=int, default=0)
    approx = commands.add_parser(
        "lobby-approx", help="Anytime local-swap assignment for lobbies too large to enumerate"
    )
    approx.add_argument("--players", type=int, default=20)
    approx.add_argument("--seed", type=int, default=0)
    approx.add_argument("--time-budget", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.command in {"models", "input-profiles", "check-models", "outcome-benchmark"}:
        from dataclasses import asdict

        from .catalog import PROFILES, check_models, model_catalog
        from .outcomes import plan_outcomes, run_outcomes

        try:
            if args.command == "models":
                value = model_catalog(args.task)
            elif args.command == "input-profiles":
                value = [asdict(profile) for profile in PROFILES.values()]
            elif args.command == "check-models":
                value = check_models(args.profile, args.models)
                print(json.dumps(value, indent=2))
                if not value["input_compatible"]:
                    parser.exit(2, "Incompatible model/input request; see reasons above.\n")
                return
            else:
                if not args.dry_run and args.output is None:
                    parser.error("--output is required unless --dry-run is used")
                if args.dry_run:
                    value = plan_outcomes(json.loads(args.config.read_text()))
                else:
                    value = {"report": str(run_outcomes(
                        args.config, args.workspace, args.output,
                    ))}
            print(json.dumps(value, indent=2))
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    elif args.command == "feature-schema":
        from .catalog import feature_schema_for

        try:
            print(json.dumps(feature_schema_for(args.profile).to_dict(), indent=2))
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    elif args.command == "schema-train":
        from .schema_config import load_schema_train_config

        try:
            config = load_schema_train_config(args.config)
            print(json.dumps(config.to_dict(), indent=2))
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    elif args.command == "doctor":
        optional_dependencies = {}
        for name in ("lightgbm", "trueskill", "openskill"):
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                optional_dependencies[name] = {"status": "not_installed", "version": None}
                continue
            try:
                importlib.import_module(name)
            except (ImportError, OSError) as exc:
                optional_dependencies[name] = {
                    "status": "import_error",
                    "version": version,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            else:
                optional_dependencies[name] = {"status": "available", "version": version}
        # The base install deliberately has no PyTorch; report that state
        # instead of failing at import time.
        try:
            import torch
        except (ImportError, OSError) as exc:
            torch_version = None
            cuda_available = mps_available = None
            torch_error = f"{type(exc).__name__}: {exc}"
        else:
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            mps_available = torch.backends.mps.is_available()
            torch_error = None
        value = {
            "python": platform.python_version(),
            "torch": torch_version,
            "torch_status": "available" if torch_version else "not_installed",
            "torch_error": torch_error,
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "cuda_available": cuda_available,
            "mps_available": mps_available,
            "optional_dependencies": optional_dependencies,
        }
        print(json.dumps(value, indent=2))
    elif args.command == "datasets":
        print(json.dumps(REGISTRY, indent=2))
    elif args.command == "fetch-data":
        print(json.dumps(fetch_dataset(args.dataset, args.data_root), indent=2))
    elif args.command == "inspect":
        _, _, manifest = prepare(args.dataset, args.workspace / "datasets", "pilot")
        print(json.dumps(manifest, indent=2))
    elif args.command == "benchmark":
        from .runner import run_benchmark

        path = run_benchmark(args.config.resolve(), args.workspace.resolve(), args.output.resolve())
        print(f"Report: {path}")
    elif args.command == "report":
        print(create_run_report(args.run_directory, args.output))
    elif args.command == "rating-benchmark":
        from .rating_runner import run_ratings

        print(run_ratings(args.config.resolve(), args.workspace.resolve(), args.output.resolve()))
    elif args.command == "history-benchmark":
        from .history_runner import run_history

        print(run_history(args.config.resolve(), args.workspace.resolve(), args.output.resolve()))
    elif args.command == "asymmetric-benchmark":
        from .asymmetric_runner import run_asymmetric

        print(
            run_asymmetric(args.config.resolve(), args.workspace.resolve(), args.output.resolve())
        )
    elif args.command == "decision-benchmark":
        from .decision_runner import run_decisions

        print(run_decisions(args.config.resolve(), args.workspace.resolve(), args.output.resolve()))
    elif args.command == "skill-benchmark":
        from .skill_runner import run_skill

        print(run_skill(args.config.resolve(), args.workspace.resolve(), args.output.resolve()))
    elif args.command == "lobby-demo":
        rng = np.random.default_rng(args.seed)
        true_skill = rng.normal(size=10)
        estimated_skill = true_skill + rng.normal(scale=0.5, size=10)

        def probabilities(candidates, skills):
            difference = np.asarray(
                [skills[list(a)].mean() - skills[list(b)].mean() for a, b in candidates]
            )
            return 1 / (1 + np.exp(-difference))

        # The optimizer's closure has estimated skill only. True skill is used after selection.
        assignment = exact_balance(
            range(10), 5, lambda c: probabilities(c, estimated_skill), exchangeable_sides=True
        )
        candidates = list(partitions(range(10), 5, exchangeable_sides=True))
        true_p = probabilities(candidates, true_skill)
        chosen = assignment["candidate_index"]
        oracle_cost = float(np.abs(true_p - 0.5).min())
        assignment.update(
            {
                "evidence_type": "synthetic_known_mechanism",
                "seed": args.seed,
                "true_win_probability": float(true_p[chosen]),
                "oracle_best_imbalance": oracle_cost,
                "decision_regret": float(abs(true_p[chosen] - 0.5) - oracle_cost),
                "limitations": "Illustration only; additive synthetic skills, no real players or outcome intervention",
            }
        )
        print(json.dumps(assignment, indent=2))
    elif args.command == "lobby-approx":
        from math import comb

        if args.players < 4 or args.players % 2 or not args.time_budget > 0:
            parser.error("--players must be even and >= 4; --time-budget must be positive")
        rng = np.random.default_rng(args.seed)
        skills = rng.normal(size=args.players)
        team_size = args.players // 2

        def predict(candidates):
            difference = np.asarray(
                [skills[list(a)].mean() - skills[list(b)].mean() for a, b in candidates]
            )
            return 1 / (1 + np.exp(-difference))

        result = local_swap_balance(
            range(args.players), team_size, predict,
            time_budget=args.time_budget, seed=args.seed,
        )
        result["exact_candidate_count"] = comb(args.players, team_size)
        result["evidence_type"] = "synthetic_additive_skills_approximate_solver"
        print(json.dumps(result, indent=2))
