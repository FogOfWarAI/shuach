# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Temporal future-rating regression with target-availability purging."""

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import completed_fit, fit_files, seal_artifacts, verify_completions
from .configuration import runtime_identity, validate_matrix, validate_training_settings
from .data.heroes import HeroVocabulary
from .data.skill import load_skill, skill_splits
from .models.skill import make_skill_model
from .models.tabular import hero_counts
from .runner import code_hash, set_seed, write_json


def regression_metrics(target, prediction):
    y, p = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or not len(y)
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Expected finite, aligned, nonempty regression vectors")
    error = p - y
    variance = np.square(y - y.mean()).sum()
    return {
        "n": len(y),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mean_error": float(error.mean()),
        "r2": float(1 - np.square(error).sum() / variance) if variance else None,
        "spearman": float(spearmanr(y, p).statistic) if np.std(y) > 0 and np.std(p) > 0 else None,
    }


def infer_skill(model, dataset, batch_size, device):
    model.eval()
    values = []
    with torch.no_grad():
        for snapshots, lineup, valid, _ in DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        ):
            values.append(
                model(snapshots.to(device), lineup.to(device), valid.to(device)).cpu().numpy()
            )
    return np.concatenate(values)


def run_skill(config_path, workspace, output):
    config_path, workspace, output = (Path(p).resolve() for p in (config_path, workspace, output))
    verify_completions(output)
    config = json.loads(config_path.read_text())
    validate_training_settings(config.get("training", {}))
    if config.get("protocol") != "temporal_future_observed_rating":
        raise ValueError("Explicit future-rating protocol required")
    validate_matrix(config, ("models", "seeds"))
    allowed = {"current", "mean_delta", "ridge", "mlp", "gru", "transformer", "mmrnet"}
    if set(config["models"]) - allowed:
        raise ValueError("Unknown skill model")
    data, match_metadata = load_skill(workspace)
    masks, bounds = skill_splits(data, match_metadata)
    for part, cap in [
        ("train", config.get("max_train_rows")),
        ("valid", config.get("max_eval_rows")),
        ("test", config.get("max_eval_rows")),
    ]:
        if cap and masks[part].sum() > cap:
            selected = np.random.default_rng(0).choice(
                np.flatnonzero(masks[part]), cap, replace=False
            )
            masks[part][:] = False
            masks[part][selected] = True
    values = data.snapshots[masks["train"]][data.valid[masks["train"]]]
    mean, scale = values.mean(0), values.std(0).clip(1e-6)
    x = ((data.snapshots - mean) / scale).astype(np.float32)
    x[~data.valid] = 0
    delta = data.target_mu - data.current_mu
    target_mean, target_scale = (
        float(delta[masks["train"]].mean()),
        float(delta[masks["train"]].std().clip(1e-6)),
    )
    normalized = ((delta - target_mean) / target_scale).astype(np.float32)
    vocabulary = HeroVocabulary().fit(data.lineups[masks["train"]])
    rosters = vocabulary.transform(data.lineups)
    datasets = {
        part: TensorDataset(
            torch.from_numpy(x[m]),
            torch.from_numpy(rosters[m]),
            torch.from_numpy(data.valid[m]),
            torch.from_numpy(normalized[m]),
        )
        for part, m in masks.items()
    }
    identity = {
        "config": config,
        "code": code_hash(),
        "data": data.manifest,
        "runtime": runtime_identity(),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Existing skill run has different code, data or config")
    else:
        manifest = {
            **identity,
            "fingerprint": fingerprint,
            "code_sha256": identity["code"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "split_counts": {k: int(v.sum()) for k, v in masks.items()},
            "time_boundaries": bounds,
            "normalizer": {
                "feature_mean": mean.tolist(),
                "feature_scale": scale.tolist(),
                "delta_mean": target_mean,
                "delta_scale": target_scale,
            },
            "hero_vocabulary": vocabulary.values.tolist(),
            "input_available": "After the current game has completed",
            "target_purge": "Training and validation require five-future-match proxy to be available before the next split boundary",
            "prediction": "Current reference mean plus a learned future-state change",
            "low_history_test_count": int((masks["test"] & data.low_history).sum()),
            "limitation": "Sparse, returning-player-conditioned public task; no verified novice or true-skill accuracy claim",
        }
        shutil.copytree(
            Path(__file__).parent,
            output / "source-snapshot/gamematch",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        write_json(output / "resolved_config.json", config)
        np.savez_compressed(
            output / "split-manifest.npz",
            row_id=data.row_id,
            match_id=data.match_id,
            player_id=data.player_id,
            time=data.time,
            target_available_at=data.target_available_at,
            **masks,
        )
    manifest["status"] = "running"
    write_json(path, manifest)
    training = config["training"]
    try:
        for seed in config["seeds"]:
            for name in config["models"]:
                directory = output / f"seed-{seed}-{name}"
                if completed_fit(directory, fit_files(name), output):
                    continue
                print(f"Training future-rating {name}/{seed}", flush=True)
                directory.mkdir(exist_ok=True)
                set_seed(seed, training["threads"])
                started = time.perf_counter()
                history, best_epoch, parameters = [], 0, 0
                if name in {"current", "mean_delta"}:
                    prediction = data.current_mu[masks["test"]] + (
                        target_mean if name == "mean_delta" else 0
                    )
                    write_json(
                        directory / "checkpoint.json",
                        {"residual": target_mean if name == "mean_delta" else 0},
                    )
                elif name == "ridge":
                    matrix = np.concatenate(
                        [
                            x.reshape(len(x), -1),
                            data.valid.astype(np.float32),
                            hero_counts(rosters, vocabulary.size).toarray(),
                        ],
                        1,
                    )
                    model = Ridge(alpha=1.0)
                    with threadpool_limits(limits=training["threads"]):
                        model.fit(matrix[masks["train"]], normalized[masks["train"]])
                        predicted = model.predict(matrix[masks["test"]])
                    prediction = (
                        data.current_mu[masks["test"]] + predicted * target_scale + target_mean
                    )
                    joblib.dump(model, directory / "checkpoint.joblib")
                    parameters = int(model.coef_.size + 1)
                else:
                    device = config.get("model_devices", {}).get(name, training["device"])
                    if device == "mps" and not torch.backends.mps.is_available():
                        raise ValueError("MPS unavailable")
                    model = make_skill_model(name, x.shape[-1], vocabulary.size, training).to(
                        device
                    )
                    parameters = sum(p.numel() for p in model.parameters())
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
                    best, stale = float("inf"), 0
                    for epoch in range(1, training["epochs"] + 1):
                        model.train()
                        loss_sum = 0.0
                        tick = time.perf_counter()
                        for snapshots, lineup, valid, y in loader:
                            optimizer.zero_grad(set_to_none=True)
                            pred = model(snapshots.to(device), lineup.to(device), valid.to(device))
                            loss = torch.nn.functional.mse_loss(pred, y.to(device))
                            if not torch.isfinite(loss):
                                raise FloatingPointError("Nonfinite skill regression loss")
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 5, error_if_nonfinite=True
                            )
                            optimizer.step()
                            loss_sum += float(loss.detach().cpu()) * len(y)
                        valid_delta = infer_skill(
                            model, datasets["valid"], training["batch_size"], device
                        )
                        valid_mse = float(
                            np.square(valid_delta - normalized[masks["valid"]]).mean()
                        )
                        history.append(
                            {
                                "epoch": epoch,
                                "train_mse_standardized_delta": loss_sum / len(datasets["train"]),
                                "valid_mse_standardized_delta": valid_mse,
                                "seconds": time.perf_counter() - tick,
                            }
                        )
                        write_json(directory / "history.json", history)
                        print(
                            f"  {name}/{seed} epoch={epoch} valid_mse={valid_mse:.6f}", flush=True
                        )
                        if valid_mse < best:
                            best, stale, best_epoch = valid_mse, 0, epoch
                            state = {
                                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                            }
                            torch.save(
                                {
                                    "state_dict": state,
                                    "model": name,
                                    "features": x.shape[-1],
                                    "hero_vocab": vocabulary.size,
                                    "training": training,
                                },
                                directory / "checkpoint.pt.tmp",
                            )
                            (directory / "checkpoint.pt.tmp").replace(directory / "checkpoint.pt")
                        else:
                            stale += 1
                            if stale >= training["patience"]:
                                break
                    model.load_state_dict(
                        torch.load(
                            directory / "checkpoint.pt", map_location="cpu", weights_only=True
                        )["state_dict"]
                    )
                    predicted = infer_skill(model, datasets["test"], training["batch_size"], device)
                    prediction = (
                        data.current_mu[masks["test"]] + predicted * target_scale + target_mean
                    )
                target = data.target_mu[masks["test"]]
                low = data.low_history[masks["test"]]
                result = {
                    "model": name,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "epochs_completed": len(history),
                    "parameter_count": parameters,
                    "fit_and_score_seconds": time.perf_counter() - started,
                    "metrics": regression_metrics(target, prediction),
                    "limited_observed_history_subgroup": regression_metrics(
                        target[low], prediction[low]
                    )
                    if low.any()
                    else None,
                    "subgroup_caution": "Very small subgroup; not a novice effectiveness estimate",
                }
                np.savez_compressed(
                    directory / "predictions.npz",
                    row_id=data.row_id[masks["test"]],
                    match_id=data.match_id[masks["test"]],
                    player_id=data.player_id[masks["test"]],
                    time=data.time[masks["test"]],
                    y_true=target,
                    prediction=prediction,
                    current_mu=data.current_mu[masks["test"]],
                    target_available_at=data.target_available_at[masks["test"]],
                    low_history=low,
                )
                write_json(directory / "metrics.json", result)
                seal_artifacts(directory, fit_files(name), output)
                print(f"  {name}/{seed}: MAE={result['metrics']['mae']:.6f}", flush=True)
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
