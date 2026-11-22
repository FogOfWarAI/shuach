# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Public-data OwO/DAWN adaptations with frozen, temporal evaluation."""

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import completed_fit, seal_artifacts, verify_completions
from .configuration import runtime_identity, validate_matrix, validate_training_settings
from .data.events import load_overwatch
from .data.heroes import file_hash
from .data.history import (
    HISTORY_FEATURES,
    STATIC_FEATURES,
    HistoryData,
    build_histories,
    history_split_masks,
    standardize_histories,
)
from .evaluation.multiclass import multiclass_metrics
from .models.history import make_history_model
from .rating_runner import evaluate_probabilities
from .runner import code_hash, set_seed, write_json


def cached_history(workspace, config):
    if config.get("target_domain_features_only", False) and not config.get(
        "same_domain_only", False
    ):
        raise ValueError("Target-domain feature ablation also requires same-domain sequences")
    source = workspace / config.get("source", "datasets/raw/overwatch.jsonl")
    identity = {
        "sha256": file_hash(source),
        "length": config["history_length"],
        "same_domain_only": config.get("same_domain_only", False),
        "target_domain_features_only": config.get("target_domain_features_only", False),
        "feedback_delay_seconds": config.get("feedback_delay_seconds", 0),
        "builder_sha256": file_hash(Path(__file__).parent / "data/history.py"),
        "adapter_sha256": file_hash(Path(__file__).parent / "data/events.py"),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    directory = workspace / ".cache/history"
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / f"{fingerprint}.npz"
    metadata = directory / f"{fingerprint}.json"
    if cached.exists() and metadata.exists():
        record = json.loads(metadata.read_text())
        if file_hash(cached) == record["cache_sha256"]:
            return HistoryData.load(cached), record
    events, data_manifest = load_overwatch(source, identity["feedback_delay_seconds"])
    data = build_histories(events, identity["length"], identity["same_domain_only"])
    if identity["target_domain_features_only"]:
        # Preserve shape and target-mode indicator while withholding global and
        # other-mode outcomes. Selection happens before fitting normalizers.
        data.static[..., :4] = 0
        for domain in (0, 1):
            data.static[data.domain == domain, ..., (1 - domain) * 4 + 4 : (1 - domain) * 4 + 8] = 0
    data.save(cached)
    record = {
        "identity": identity,
        "source": data_manifest,
        "cache_sha256": file_hash(cached),
        "history_features": HISTORY_FEATURES,
        "static_features": STATIC_FEATURES,
        "history_available_fraction": float(data.valid.mean()),
        "feature_scope": "target-domain sequences and static statistics only"
        if identity["target_domain_features_only"]
        else "cross-domain static statistics; sequence scope recorded in identity",
        "limitations": [
            "WDL outcome-only history; no original KDA, economy, role or party features",
            "Unordered 6v6 rosters replace industrial team-and-position slots",
            "Three-class WDL head replaces original binary win head",
            "Architectural and dataset adaptation, not original industrial result reproduction",
        ],
    }
    write_json(metadata, record)
    return data, record


def tensor_partition(data, mask):
    return TensorDataset(
        *[
            torch.from_numpy(array[mask])
            for array in (data.history, data.static, data.valid, data.domain, data.outcome)
        ]
    )


def predict_logits(model, dataset, batch_size, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for history, static, valid, domain, _ in DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        ):
            predictions.append(
                model(
                    history.to(device),
                    static.to(device),
                    valid.to(device),
                    domain=domain.to(device),
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(predictions)


def temperature_fit(logits, outcomes):
    def objective(log_temperature):
        probability = softmax(logits / np.exp(log_temperature), axis=1)
        return multiclass_metrics(outcomes, probability)["log_loss"]

    optimum = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    if not optimum.success:
        raise RuntimeError("Validation temperature optimization did not converge")
    return float(np.exp(optimum.x))


def run_history(config_path, workspace, output):
    config_path, workspace, output = (Path(p).resolve() for p in (config_path, workspace, output))
    verify_completions(output)
    config = json.loads(config_path.read_text())
    if config.get("protocol") != "temporal_completed_player_history":
        raise ValueError("Explicit temporal_completed_player_history protocol required")
    validate_matrix(config, ("models", "seeds"))
    allowed = {"history_deepsets", "static_deepsets", "owo", "dawn", "owo_transformer"}
    if not config.get("models") or set(config["models"]) - allowed:
        raise ValueError("Invalid history model matrix")
    training = config["training"]
    validate_training_settings(training)
    if training["device"] == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable in this process")
    data, data_manifest = cached_history(workspace, config)
    masks, bounds = history_split_masks(
        data, config.get("train_fraction", 0.7), config.get("valid_fraction", 0.15)
    )
    if min(int(m.sum()) for m in masks.values()) < 1:
        raise ValueError("Empty temporal partition")
    for name, cap in [
        ("train", config.get("max_train_rows")),
        ("valid", config.get("max_eval_rows")),
        ("test", config.get("max_eval_rows")),
    ]:
        if cap and masks[name].sum() > cap:
            indices = np.random.default_rng(0).choice(
                np.flatnonzero(masks[name]), cap, replace=False
            )
            masks[name][:] = False
            masks[name][indices] = True
    normalizer = standardize_histories(data, masks["train"])
    datasets = {name: tensor_partition(data, mask) for name, mask in masks.items()}
    runtime = runtime_identity()
    fingerprint = hashlib.sha256(
        json.dumps(
            {"config": config, "data": data_manifest, "code": code_hash(), "runtime": runtime},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Output belongs to different code, configuration or data")
    else:
        manifest = {
            "fingerprint": fingerprint,
            "code_sha256": code_hash(),
            "config": config,
            "runtime": runtime,
            "data": data_manifest,
            "time_boundaries": bounds,
            "split_counts": {k: int(m.sum()) for k, m in masks.items()},
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "torch": torch.__version__,
            "task": "three_class_observed_prediction",
            "evidence_type": "public_data_architecture_adaptation",
            "normalizer": normalizer,
            "selection": "Validation NLL early stopping; validation-only scalar temperature calibration",
            "test_policy": "Frozen network and calibration; player histories receive strictly earlier completed test outcomes",
        }
        shutil.copytree(
            Path(__file__).parent,
            output / "source-snapshot/gamematch",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        write_json(output / "resolved_config.json", config)
        np.savez_compressed(
            output / "split-manifest.npz",
            row_id=data.row_ids,
            time=data.time,
            available_at=data.available_at,
            **masks,
        )
    manifest["status"] = "running"
    write_json(path, manifest)
    try:
        for seed in config["seeds"]:
            for name in config["models"]:
                directory = output / f"seed-{seed}-{name}"
                required = (
                    "metrics.json",
                    "checkpoint.pt",
                    "predictions.npz",
                    "validation-predictions.npz",
                )
                if completed_fit(directory, required, output):
                    print(f"Skipping {name}/{seed}", flush=True)
                    continue
                print(f"Training history {name}/{seed}", flush=True)
                directory.mkdir(exist_ok=True)
                set_seed(seed, training["threads"])
                device = training["device"]
                model = make_history_model(
                    name,
                    data.history.shape[-1],
                    data.static.shape[-1],
                    data.history.shape[2],
                    data.history.shape[3],
                    training,
                ).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=training["learning_rate"],
                    weight_decay=training["weight_decay"],
                )
                loader = DataLoader(
                    datasets["train"],
                    batch_size=training["batch_size"],
                    shuffle=True,
                    generator=torch.Generator().manual_seed(seed),
                )
                best, stale, log, best_epoch = float("inf"), 0, [], 0
                started = time.perf_counter()
                for epoch in range(1, training["epochs"] + 1):
                    tick, loss_sum = time.perf_counter(), 0.0
                    model.train()
                    for history, static, valid, domain, outcome in loader:
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(
                            history.to(device),
                            static.to(device),
                            valid.to(device),
                            domain=domain.to(device),
                        )
                        loss = torch.nn.functional.cross_entropy(logits, outcome.to(device))
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite history training loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 5.0, error_if_nonfinite=True
                        )
                        optimizer.step()
                        loss_sum += float(loss.detach().cpu()) * len(outcome)
                    validation = predict_logits(
                        model, datasets["valid"], training["batch_size"], device
                    )
                    valid_loss = multiclass_metrics(
                        data.outcome[masks["valid"]], softmax(validation, axis=1)
                    )["log_loss"]
                    log.append(
                        {
                            "epoch": epoch,
                            "train_loss": loss_sum / len(datasets["train"]),
                            "valid_log_loss": valid_loss,
                            "seconds": time.perf_counter() - tick,
                        }
                    )
                    write_json(directory / "history.json", log)
                    print(
                        f"  {name} seed={seed} epoch={epoch} valid_log_loss={valid_loss:.6f}",
                        flush=True,
                    )
                    if valid_loss < best:
                        best, stale, best_epoch = valid_loss, 0, epoch
                        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        temporary = directory / "checkpoint.pt.tmp"
                        torch.save(
                            {
                                "state_dict": state,
                                "model": name,
                                "training": training,
                                "history_features": data.history.shape[-1],
                                "static_features": data.static.shape[-1],
                                "team_size": data.history.shape[2],
                                "history_length": data.history.shape[3],
                            },
                            temporary,
                        )
                        temporary.replace(directory / "checkpoint.pt")
                    else:
                        stale += 1
                        if stale >= training["patience"]:
                            break
                model.load_state_dict(
                    torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)[
                        "state_dict"
                    ]
                )
                valid_logits = predict_logits(
                    model, datasets["valid"], training["batch_size"], device
                )
                temperature = temperature_fit(valid_logits, data.outcome[masks["valid"]])
                fit_seconds = time.perf_counter() - started
                start = time.perf_counter()
                test_logits = predict_logits(
                    model, datasets["test"], training["batch_size"], device
                )
                inference_seconds = time.perf_counter() - start
                probability = softmax(test_logits / temperature, axis=1)
                y, domain = data.outcome[masks["test"]], data.domain[masks["test"]]
                all_rows = np.ones(len(y), dtype=bool)
                result = {
                    "model": name,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "epochs_completed": len(log),
                    "temperature": temperature,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "fit_seconds": fit_seconds,
                    "test_seconds": inference_seconds,
                    "device": device,
                    "metrics": evaluate_probabilities(y, probability, all_rows),
                    "uncalibrated_metrics": multiclass_metrics(y, softmax(test_logits, axis=1)),
                    "subgroups": {
                        str(d): evaluate_probabilities(y, probability, domain == d) for d in (0, 1)
                    },
                }
                np.savez_compressed(
                    directory / "predictions.npz",
                    row_id=data.row_ids[masks["test"]],
                    y_true=y,
                    domain=domain,
                    time=data.time[masks["test"]],
                    probabilities=probability,
                    logits=test_logits,
                )
                np.savez_compressed(
                    directory / "validation-predictions.npz",
                    row_id=data.row_ids[masks["valid"]],
                    y_true=data.outcome[masks["valid"]],
                    logits=valid_logits,
                )
                write_json(directory / "metrics.json", result)
                seal_artifacts(directory, required, output)
                print(
                    f"  {name}/{seed} test WDL NLL={result['metrics']['log_loss']:.6f}", flush=True
                )
        manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, manifest)
        write_json(
            output / "results.json",
            [json.loads(p.read_text()) for p in sorted(output.glob("seed-*/metrics.json"))],
        )
    return output / "results.json"
