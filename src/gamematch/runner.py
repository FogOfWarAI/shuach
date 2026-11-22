# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Local CPU/GPU training with validation-only selection and durable artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import __version__
from .artifacts import completed_fit, fit_files, seal_artifacts, verify_completions, write_json
from .configuration import runtime_identity, validate_matrix, validate_training_settings
from .data.heroes import HeroRecords, HeroVocabulary, prepare
from .evaluation.binary import binary_metrics
from .models import MODEL_NAMES, make_model
from .models.tabular import train_tabular


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 3
    embedding_dim: int = 20
    hidden_dim: int = 64
    device: str = "cpu"
    threads: int = 4
    max_trees: int = 500
    tree_learning_rate: float = 0.05
    tree_patience: int = 30
    logistic_c: float = 1.0

    def validate(self):
        validate_training_settings(asdict(self))
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available; choose cpu")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available in this process; choose cpu")


def code_hash():
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def set_seed(seed: int, threads: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def as_tensor_data(records: HeroRecords, vocabulary: HeroVocabulary):
    return TensorDataset(
        torch.from_numpy(vocabulary.transform(records.rosters)), torch.from_numpy(records.labels)
    )


def predict(model, data, config):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in DataLoader(data, batch_size=config.batch_size, shuffle=False, num_workers=0):
            predictions.append(torch.sigmoid(model(x.to(config.device))).cpu().numpy())
    return np.concatenate(predictions)


def fit_one(name, splits, vocabulary, config: TrainConfig, seed: int, directory: Path):
    config.validate()
    set_seed(seed, config.threads)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    valid_y = splits["valid"].labels
    test_y = splits["test"].labels
    if name == "prior":
        prior = float(splits["train"].labels.mean())
        write_json(directory / "checkpoint.json", {"positive_probability": prior})
        valid_p = np.full(len(valid_y), prior)
        fit_seconds = time.perf_counter() - started
        inference_started = time.perf_counter()
        test_p = np.full(len(test_y), prior)
        test_seconds = time.perf_counter() - inference_started
        history, best_epoch, parameter_count = [], 0, 0
    elif name in {"logistic", "lightgbm"}:
        model, data, best_epoch = train_tabular(name, splits, vocabulary, config, seed, directory)
        valid_p = model.predict_proba(data["valid"])[:, 1]
        fit_seconds = time.perf_counter() - started
        inference_started = time.perf_counter()
        test_p = model.predict_proba(data["test"])[:, 1]
        test_seconds = time.perf_counter() - inference_started
        parameter_count = (
            int(model.coef_.size + model.intercept_.size) if name == "logistic" else None
        )
        history = []
    else:
        data = {key: as_tensor_data(value, vocabulary) for key, value in splits.items()}
        model = make_model(name, vocabulary.size, config.embedding_dim, config.hidden_dim).to(
            config.device
        )
        parameter_count = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            data["train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        best_loss, stale, best_epoch, history = float("inf"), 0, 0, []
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            loss_sum = 0.0
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(x.to(config.device))
                loss = criterion(logits, y.to(config.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(y)
            valid_p = predict(model, data["valid"], config)
            valid_loss = binary_metrics(valid_y, valid_p)["log_loss"]
            history.append(
                {
                    "epoch": epoch,
                    "train_bce": loss_sum / len(data["train"]),
                    "valid_log_loss": valid_loss,
                    "epoch_seconds": time.perf_counter() - epoch_start,
                }
            )
            write_json(directory / "history.json", history)
            print(f"  {name} seed={seed} epoch={epoch} valid_log_loss={valid_loss:.5f}", flush=True)
            if valid_loss < best_loss:
                best_loss, stale, best_epoch = valid_loss, 0, epoch
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                temporary = directory / "checkpoint.pt.tmp"
                torch.save(
                    {
                        "state_dict": state,
                        "model": name,
                        "epoch": epoch,
                        "vocab_size": vocabulary.size,
                        "training": asdict(config),
                    },
                    temporary,
                )
                temporary.replace(directory / "checkpoint.pt")
            else:
                stale += 1
                if stale >= config.patience:
                    break
        checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        valid_p = predict(model, data["valid"], config)
        fit_seconds = time.perf_counter() - started
        inference_started = time.perf_counter()
        test_p = predict(model, data["test"], config)
        test_seconds = time.perf_counter() - inference_started
    metrics = {
        "validation": binary_metrics(valid_y, valid_p),
        "evaluation": binary_metrics(test_y, test_p),
    }
    np.savez_compressed(
        directory / "predictions.npz",
        row_id=splits["test"].row_ids,
        group_id=splits["test"].groups,
        y_true=test_y,
        p_positive=test_p,
    )
    return {
        "model": name,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "parameter_count": parameter_count,
        "fit_seconds_including_validation": fit_seconds,
        "test_inference_seconds": test_seconds,
        "metrics": metrics,
        "hyperparameters": asdict(config),
        "device": "cpu" if name in {"prior", "logistic", "lightgbm"} else config.device,
        "backend": "lightgbm"
        if name == "lightgbm"
        else "scikit-learn"
        if name == "logistic"
        else "numpy"
        if name == "prior"
        else "pytorch",
    }


def render_report(run_directory: Path):
    manifest = json.loads((run_directory / "manifest.json").read_text())
    if manifest["config"].get("protocol") not in {"pilot", "final"}:
        raise ValueError("Legacy render_report supports hero runs only; use gamematch report")
    results = []
    for path in sorted(run_directory.glob("*/seed-*/metrics.json")):
        result = json.loads(path.read_text())
        result["artifact_path"] = str(path.relative_to(run_directory))
        results.append(result)
    write_json(run_directory / "results.json", results)
    protocol_note = (
        "Pilot scores are development measurements on inner held-out data, not publication results."
        if manifest["config"]["protocol"] == "pilot"
        else "Reference-configuration benchmark on declared final holdouts. See the frozen configuration for budgets and reproduction scope."
    )
    lines = [
        "# GameMatch experiment report",
        "",
        f"Protocol: **{manifest['config']['protocol']}**. Version: {manifest['version']}.",
        "",
        protocol_note,
        "All rows are hero-roster predictions, not player matchmaking or causal decision results.",
        "",
        "| Dataset | Model | Seed | Train / valid / eval | LogLoss ↓ | AUC ↑ | Accuracy ↑ | Brier ↓ | Best epoch / iteration | Fit seconds |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        m = r["metrics"]["evaluation"]
        counts = r["split_rows"]
        auc = "N/A" if m["auc"] is None else f"{m['auc']:.4f}"
        lines.append(
            f"| {r['dataset']} | {r['model']} | {r['seed']} | "
            f"{counts['train']} / {counts['valid']} / {counts['test']} | "
            f"{m['log_loss']:.4f} | {auc} | {m['accuracy']:.4f} | {m['brier']:.4f} | "
            f"{r['best_epoch']} | {r['fit_seconds_including_validation']:.2f} |"
        )
    grouped = {}
    for r in results:
        grouped.setdefault((r["dataset"], r["model"]), []).append(r)
    lines += [
        "",
        "## Across seeds",
        "",
        "Mean ± sample standard deviation; not a confidence interval.",
        "",
        "| Dataset | Model | Seeds | LogLoss | AUC |",
        "|---|---|---:|---:|---:|",
    ]
    for (dataset, model), rows in sorted(grouped.items()):
        cells = []
        for metric in ["log_loss", "auc"]:
            values = [
                r["metrics"]["evaluation"][metric]
                for r in rows
                if r["metrics"]["evaluation"][metric] is not None
            ]
            cell = "N/A" if not values else f"{np.mean(values):.4f}"
            if len(values) > 1:
                cell += f" ± {np.std(values, ddof=1):.4f}"
            cells.append(cell)
        lines.append(f"| {dataset} | {model} | {len(rows)} | {' | '.join(cells)} |")
    lines += [
        "",
        "Training and validation timing excludes source loading and preprocessing.",
        "Source hashes, exact row/group splits, training vocabulary, checkpoints and per-example predictions are saved alongside the report.",
        "No hyperparameter search, interval estimation or model superiority claim is implied.",
        "",
    ]
    (run_directory / "report.md").write_text("\n".join(lines))
    return results


def run_benchmark(config_path: Path, workspace: Path, output: Path):
    config_path, workspace, output = (Path(p).resolve() for p in (config_path, workspace, output))
    raw = json.loads(config_path.read_text())
    if raw.get("protocol") not in {"pilot", "final"}:
        raise ValueError("Declare protocol pilot or final explicitly")
    validate_matrix(raw, ("datasets", "models", "seeds"))
    if set(raw["models"]) - set(MODEL_NAMES):
        raise ValueError("A requested model has not been implemented")
    training = TrainConfig(**raw.get("training", {}))
    training.validate()
    for model in raw["models"]:
        TrainConfig(**(asdict(training) | raw.get("model_training", {}).get(model, {}))).validate()
    verify_completions(output)
    source_hash = code_hash()
    data_root = workspace / raw.get("data_root", "datasets")
    identity = {
        "config": raw,
        "code_sha256": source_hash,
        "data_root": str(data_root.resolve()),
        "runtime": runtime_identity(),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["fingerprint"] != fingerprint:
            raise ValueError(
                "Existing output belongs to different code/config/runtime; select a new output directory"
            )
        manifest = old
    else:
        manifest = {
            **identity,
            "fingerprint": fingerprint,
            "version": __version__,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "environment": {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": training.device,
                "threads": training.threads,
            },
        }
        packages = ["numpy", "pandas", "scikit-learn", "torch"]
        if "lightgbm" in raw["models"]:
            packages.append("lightgbm")
        manifest["environment"]["dependencies"] = {
            name: importlib.metadata.version(name) for name in packages
        }
        snapshot = output / "source-snapshot/gamematch"
        shutil.copytree(
            Path(__file__).parent, snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    write_json(output / "resolved_config.json", raw)
    try:
        for dataset in raw["datasets"]:
            print(f"Preparing {dataset} ({raw['protocol']})", flush=True)
            splits, vocabulary, data_manifest = prepare(
                dataset,
                data_root,
                raw["protocol"],
                raw.get("split_seed", 0),
                raw.get("max_train_rows"),
                raw.get("max_eval_rows"),
            )
            dataset_dir = output / dataset
            dataset_dir.mkdir(exist_ok=True)
            data_path = dataset_dir / "data-manifest.json"
            if data_path.exists() and json.loads(data_path.read_text()) != data_manifest:
                raise ValueError(
                    "Dataset contents/splits changed; refusing to reuse completed runs"
                )
            write_json(data_path, data_manifest)
            np.savez_compressed(
                dataset_dir / "split-manifest.npz",
                **{
                    f"{name}_{field}": getattr(rows, field)
                    for name, rows in splits.items()
                    for field in ["row_ids", "groups"]
                },
            )
            for seed in raw["seeds"]:
                for model in raw["models"]:
                    directory = dataset_dir / f"seed-{seed}-{model}"
                    if completed_fit(directory, fit_files(model), output):
                        print(f"Skipping completed {dataset}/{model}/{seed}", flush=True)
                        continue
                    print(f"Training {dataset}/{model}/{seed}", flush=True)
                    effective = TrainConfig(
                        **(asdict(training) | raw.get("model_training", {}).get(model, {}))
                    )
                    result = fit_one(model, splits, vocabulary, effective, seed, directory)
                    result.update(
                        {
                            "dataset": dataset,
                            "protocol": raw["protocol"],
                            "split_rows": data_manifest["rows"],
                            "fingerprint": fingerprint,
                            "entity_kind": "hero",
                            "evidence_type": "observed_prediction",
                        }
                    )
                    write_json(directory / "metrics.json", result)
                    seal_artifacts(directory, fit_files(model), output)
                    render_report(output)
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        render_report(output)
    return output / "report.md"
