# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Temporal five-outcome benchmark; full-roster matchmaking is out of scope."""

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import completed_fit, fit_files, seal_artifacts, verify_completions
from .configuration import runtime_identity, validate_matrix, validate_training_settings
from .data.asymmetric import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_asymmetric
from .data.history import history_split_masks
from .evaluation.multiclass import multiclass_metrics
from .evaluation.ordinal import ordinal_metrics
from .history_runner import temperature_fit
from .models.ordinal import OrdinalMLP
from .models.tabular import predict_kill_counts, require_logistic_convergence
from .runner import code_hash, set_seed, write_json


def run_asymmetric(config_path, workspace, output):
    config_path, workspace, output = (Path(p).resolve() for p in (config_path, workspace, output))
    verify_completions(output)
    config = json.loads(config_path.read_text())
    validate_training_settings(config.get("training", {}))
    if config.get("protocol") != "temporal_predator_outcome":
        raise ValueError("Explicit temporal_predator_outcome protocol required")
    validate_matrix(config, ("datasets", "models", "seeds"))
    if set(config["models"]) - {
        "prior",
        "history_frequency",
        "logistic",
        "lightgbm",
        "mlp",
        "ordinal",
    }:
        raise ValueError("Unknown asymmetric model")
    runtime = runtime_identity()
    fingerprint = hashlib.sha256(
        json.dumps(
            {"config": config, "code": code_hash(), "runtime": runtime}, sort_keys=True
        ).encode()
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Incompatible existing experiment")
    else:
        manifest = {
            "config": config,
            "runtime": runtime,
            "fingerprint": fingerprint,
            "code_sha256": code_hash(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Observed asymmetric predator outcome prediction; no four-prey identities or reassigned-team evaluation",
            "selection": "Validation loss early stopping; scalar temperature fitted on validation only",
            "test_policy": "Frozen model; update player history only after previous outcomes become available",
        }
        shutil.copytree(
            Path(__file__).parent,
            output / "source-snapshot/gamematch",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        write_json(output / "resolved_config.json", config)
    manifest["status"] = "running"
    write_json(path, manifest)
    t = config["training"]
    try:
        for dataset in config["datasets"]:
            data = load_asymmetric(
                dataset, workspace / "datasets", config.get("feedback_delay_seconds", 0)
            )
            masks, bounds = history_split_masks(
                data, config.get("train_fraction", 0.7), config.get("valid_fraction", 0.15)
            )
            directory = output / dataset
            directory.mkdir(exist_ok=True)
            dp = directory / "data-manifest.json"
            if dp.exists() and json.loads(dp.read_text()) != data.manifest:
                raise ValueError("Source data changed")
            write_json(dp, data.manifest)
            np.savez_compressed(
                directory / "split-manifest.npz",
                row_id=data.row_ids,
                player_id=data.player_ids,
                time=data.time,
                available_at=data.available_at,
                **masks,
            )
            preprocessor = ColumnTransformer(
                [
                    ("numeric", StandardScaler(), NUMERIC_FEATURES),
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        CATEGORICAL_FEATURES,
                    ),
                ]
            )
            preprocessor.fit(data.features[masks["train"]])
            x = {
                k: np.asarray(preprocessor.transform(data.features[m]), dtype=np.float32)
                for k, m in masks.items()
            }
            y = {k: data.outcome[m] for k, m in masks.items()}
            if not any(directory.glob("seed-*/completion.json")):
                joblib.dump(preprocessor, directory / "preprocessor.joblib")
            write_json(
                directory / "features.json",
                {
                    "names": preprocessor.get_feature_names_out().tolist(),
                    "fit_partition": "train",
                    "time_boundaries": bounds,
                },
            )
            for seed in config["seeds"]:
                for name in config["models"]:
                    target = directory / f"seed-{seed}-{name}"
                    required = (*fit_files(name), "../preprocessor.joblib")
                    if completed_fit(target, required, output):
                        continue
                    target.mkdir(exist_ok=True)
                    print(f"Training {dataset}/{name}/{seed}", flush=True)
                    set_seed(seed, t["threads"])
                    started = time.perf_counter()
                    history, best_epoch = [], 0
                    if name == "prior":
                        prior = (np.bincount(y["train"], minlength=5) + 1) / (len(y["train"]) + 5)
                        probability = {
                            k: np.tile(prior, (len(v), 1)) for k, v in y.items() if k != "train"
                        }
                        write_json(target / "checkpoint.json", {"probabilities": prior.tolist()})
                    elif name == "history_frequency":
                        columns = [f"previous_kill_fraction_{k}" for k in range(5)]
                        probability = {
                            k: data.features.loc[m, columns].to_numpy(dtype=float)
                            for k, m in masks.items()
                            if k != "train"
                        }
                        write_json(
                            target / "checkpoint.json",
                            {
                                "rule": "Dirichlet(1,1,1,1,1) + strictly completed per-player outcome counts"
                            },
                        )
                    elif name in {"logistic", "lightgbm"}:
                        if name == "logistic":
                            model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
                            model.fit(x["train"], y["train"])
                            best_epoch = require_logistic_convergence(model)
                        else:
                            import lightgbm as lgb

                            model = lgb.LGBMClassifier(
                                n_estimators=t["max_trees"],
                                num_leaves=31,
                                learning_rate=0.05,
                                random_state=seed,
                                n_jobs=t["threads"],
                                verbosity=-1,
                                deterministic=True,
                                force_col_wise=True,
                            )
                            model.fit(
                                x["train"],
                                y["train"],
                                eval_set=[(x["valid"], y["valid"])],
                                eval_metric="multi_logloss",
                                callbacks=[lgb.early_stopping(30, verbose=False)],
                            )
                            best_epoch = model.best_iteration_
                        probability = {
                            k: predict_kill_counts(model, x[k]) for k in ["valid", "test"]
                        }
                        joblib.dump(model, target / "checkpoint.joblib")
                    else:
                        model = (
                            OrdinalMLP(x["train"].shape[1], t["hidden_dim"])
                            if name == "ordinal"
                            else nn.Sequential(
                                nn.Linear(x["train"].shape[1], t["hidden_dim"]),
                                nn.ReLU(),
                                nn.Linear(t["hidden_dim"], t["hidden_dim"]),
                                nn.ReLU(),
                                nn.Linear(t["hidden_dim"], 5),
                            )
                        )
                        optimizer = torch.optim.Adam(
                            model.parameters(),
                            lr=t["learning_rate"],
                            weight_decay=t["weight_decay"],
                        )
                        loader = DataLoader(
                            TensorDataset(
                                torch.from_numpy(x["train"]), torch.from_numpy(y["train"])
                            ),
                            batch_size=t["batch_size"],
                            shuffle=True,
                            generator=torch.Generator().manual_seed(seed),
                        )
                        best, stale = float("inf"), 0
                        for epoch in range(1, t["epochs"] + 1):
                            model.train()
                            for xb, yb in loader:
                                optimizer.zero_grad(set_to_none=True)
                                loss = nn.functional.cross_entropy(model(xb), yb)
                                if not torch.isfinite(loss):
                                    raise FloatingPointError("Nonfinite asymmetric training loss")
                                loss.backward()
                                optimizer.step()
                            model.eval()
                            with torch.no_grad():
                                valid_logits = model(torch.from_numpy(x["valid"])).numpy()
                            valid_loss = multiclass_metrics(
                                y["valid"], softmax(valid_logits, axis=1)
                            )["log_loss"]
                            history.append({"epoch": epoch, "valid_log_loss": valid_loss})
                            write_json(target / "history.json", history)
                            if valid_loss < best:
                                best, stale, best_epoch = valid_loss, 0, epoch
                                torch.save(
                                    {
                                        "state_dict": model.state_dict(),
                                        "model": name,
                                        "input_dim": x["train"].shape[1],
                                        "hidden_dim": t["hidden_dim"],
                                    },
                                    target / "checkpoint.pt.tmp",
                                )
                                (target / "checkpoint.pt.tmp").replace(target / "checkpoint.pt")
                            else:
                                stale += 1
                                if stale >= t["patience"]:
                                    break
                        model.load_state_dict(
                            torch.load(target / "checkpoint.pt", weights_only=True)["state_dict"]
                        )
                        model.eval()
                        with torch.no_grad():
                            probability = {
                                k: softmax(model(torch.from_numpy(x[k])).numpy(), axis=1)
                                for k in ["valid", "test"]
                            }
                    temperature = temperature_fit(
                        np.log(np.clip(probability["valid"], 1e-12, 1)), y["valid"]
                    )
                    calibrated = softmax(
                        np.log(np.clip(probability["test"], 1e-12, 1)) / temperature, axis=1
                    )
                    result = {
                        "dataset": dataset,
                        "model": name,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "epochs_completed": len(history),
                        "temperature": temperature,
                        "fit_and_score_seconds": time.perf_counter() - started,
                        "split_counts": {k: int(m.sum()) for k, m in masks.items()},
                        "metrics": ordinal_metrics(y["test"], calibrated),
                        "uncalibrated_metrics": ordinal_metrics(y["test"], probability["test"]),
                    }
                    np.savez_compressed(
                        target / "predictions.npz",
                        row_id=data.row_ids[masks["test"]],
                        player_id=data.player_ids[masks["test"]],
                        time=data.time[masks["test"]],
                        y_true=y["test"],
                        probabilities=calibrated,
                        raw_probabilities=probability["test"],
                    )
                    write_json(target / "metrics.json", result)
                    seal_artifacts(target, required, output)
                    print(f"  test NLL={result['metrics']['log_loss']:.6f}", flush=True)
        manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, manifest)
        write_json(
            output / "results.json",
            [json.loads(p.read_text()) for p in sorted(output.glob("*/seed-*/metrics.json"))],
        )
    return output / "results.json"
