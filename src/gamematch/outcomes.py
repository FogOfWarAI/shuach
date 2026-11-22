# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""A common outcome entry point preserving each adapter's experimental protocol."""

import json
from pathlib import Path

from .catalog import check_models
from .configuration import validate_matrix, validate_training_settings

PROTOCOL_PROFILES = {
    "pilot": "hero-roster",
    "final": "hero-roster",
    "temporal_completed_player_history": "overwatch-history",
    "temporal_predator_outcome": "asymmetric-context",
    "temporal_delayed_feedback": "overwatch-ratings",
}


def plan_outcomes(config):
    """Check a native outcome request before reading data or creating output.

    Args:
        config: Decoded JSON object with protocol, model IDs and the dataset or
            seed axes used by that protocol.

    Returns:
        A dictionary containing the protocol, input profile and per-model
        compatibility records. Device availability and data are checked at run time.

    Raises:
        ValueError: The requested models, inputs or declared settings conflict
            with the native adapter.
    """
    if not isinstance(config, dict):
        raise ValueError("Expected an outcome configuration object")
    protocol = config.get("protocol")
    if not isinstance(protocol, str) or protocol not in PROTOCOL_PROFILES:
        raise ValueError(
            "Unsupported outcome protocol; do not infer the target from the model name"
        )
    profile = PROTOCOL_PROFILES[protocol]
    axes = ["models"]
    if profile != "overwatch-ratings":
        axes.append("seeds")
    if profile in {"hero-roster", "asymmetric-context"} or "datasets" in config:
        axes.append("datasets")
    validate_matrix(config, axes)
    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("training must be a configuration object")
    validate_training_settings(training)
    if profile == "hero-roster":
        overrides = config.get("model_training", {})
        if not isinstance(overrides, dict):
            raise ValueError("model_training must be an object keyed by model ID")
        for model in config["models"]:
            settings = overrides.get(model, {})
            if not isinstance(settings, dict):
                raise ValueError(f"model_training.{model} must be a configuration object")
            validate_training_settings(settings)
    report = check_models(profile, config["models"])
    if config.get("task", report["profile"]["task"]) != report["profile"]["task"]:
        raise ValueError("Declared task differs from the protocol's outcome target")
    if config.get("input_profile", profile) != profile:
        raise ValueError("Declared input profile differs from the selected protocol")
    if "dataset" in config:
        raise ValueError("Use the native datasets list; a singular dataset field is not consumed")
    if not report["native_runner_ready"]:
        reasons = [
            f"{r['model']}: {r['reason']} ({r['execution']})"
            for r in report["models"]
            if r["execution"] != "native_runner"
        ]
        raise ValueError("Incompatible outcome request: " + "; ".join(reasons))
    if "datasets" in axes:
        unknown = set(config["datasets"]) - set(report["profile"]["datasets"])
        if unknown:
            raise ValueError(
                "Dataset is not served by this input adapter: " + ", ".join(sorted(unknown))
            )
    # These native adapters have fixed semantics. An ignored apparent override
    # would be more dangerous than an explicit unsupported-request error.
    for key in ("classes", "outcome_labels", "team_size", "team_sizes", "domains"):
        if key in config or key in config.get("training", {}):
            raise ValueError(f"Native adapter fixes {key}; use the explicit history-model API")
    return {"protocol": protocol, **report}


def run_outcomes(config_path, workspace, output):
    """Run a compatible native protocol and return the path to its summary.

    Args:
        config_path: Path to a JSON configuration accepted by plan_outcomes.
        workspace: Base directory for the adapter's data and preparation caches.
        output: Run directory. Reuse requires matching experiment provenance.

    Returns:
        An absolute Path to report.md for hero runs or results.json for the
        history, asymmetric and rating protocols.
    """
    config_path = Path(config_path).resolve()
    plan = plan_outcomes(json.loads(config_path.read_text()))
    profile = plan["profile"]["name"]
    if profile == "hero-roster":
        from .runner import run_benchmark

        runner = run_benchmark
    elif profile == "overwatch-history":
        from .history_runner import run_history

        runner = run_history
    elif profile == "asymmetric-context":
        from .asymmetric_runner import run_asymmetric

        runner = run_asymmetric
    else:
        from .rating_runner import run_ratings

        runner = run_ratings
    return runner(config_path, Path(workspace).resolve(), Path(output).resolve())
