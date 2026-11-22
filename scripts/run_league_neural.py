# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Fixed-budget binary history experiments with on-demand compact data batches."""

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

import gamematch
from gamematch.configuration import runtime_identity
from gamematch.data.heroes import file_hash
from gamematch.data.indexed_history import IndexedHistoryStore
from gamematch.evaluation.binary import binary_metrics
from gamematch.history_runner import temperature_fit
from gamematch.models.history import HistoryDeepSets
from gamematch.models.omninet import OwO
from gamematch.runner import code_hash, set_seed, write_json

ROOT = Path(os.environ.get("GAMEMATCH_WORKSPACE", Path(__file__).resolve().parents[1])).resolve()
DATA = ROOT / "datasets/derived/leagueData-v1"
OUT = ROOT / "runs/league-neural-v1"
RUN_IDENTITY_VALIDATED = False
RELOAD_TOLERANCE = 1e-6
SEEDS = [11, 22, 33]
CONFIG = {
    "width": 16,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 256,
    "epochs": 12,
    "patience": 4,
    "threads": 2,
    "device": "cpu",
}


def read(path):
    return json.loads(path.read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def validate_rows(store, rows):
    values = []
    for name, indices in rows.items():
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
            or ((indices < 0) | (indices >= len(store.target))).any()
            or len(np.unique(indices)) != len(indices)
            or not np.isin(store.target[indices], [0, 1]).all()
        ):
            raise ValueError("Invalid binary partition row indices or labels: " + name)
        values.append(indices)
    joined = np.concatenate(values)
    if len(np.unique(joined)) != len(joined):
        raise ValueError("Training, validation and test rows must not overlap")


def validate_config(config):
    if config.get("device") != "cpu":
        raise ValueError("This fixed-budget history driver declares CPU execution")
    for key in ["width", "batch_size", "epochs", "patience", "threads"]:
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("Positive integer configuration required: " + key)
    if (
        config["width"] % 2
        or not 0 <= config["dropout"] < 1
        or not np.isfinite([config["learning_rate"], config["weight_decay"]]).all()
        or config["learning_rate"] <= 0
        or config["weight_decay"] < 0
    ):
        raise ValueError("Invalid network or optimizer configuration")


def check_raw(raw, count):
    if raw.shape != (count, 2) or not np.isfinite(raw).all():
        raise ValueError("Expected finite binary logits")


def audit_validation(directory, model, selection, store, rows, columns, config):
    with np.load(directory / "validation.npz", allow_pickle=False) as saved:
        if (
            set(saved.files) != {"row_id", "y_true", "raw"}
            or not np.array_equal(saved["row_id"], store.row_id[rows])
            or not np.array_equal(saved["y_true"], store.target[rows])
        ):
            raise ValueError("Saved validation row IDs or labels changed")
        raw = saved["raw"]
    check_raw(raw, len(rows))
    score = binary_metrics(store.target[rows], softmax(raw.astype(float), axis=1)[:, 1])["log_loss"]
    history = read(directory / "history.json")
    if (
        len(history) != selection["epochs_completed"]
        or not 1 <= len(history) <= config["epochs"]
        or [row["epoch"] for row in history] != list(range(1, len(history) + 1))
    ):
        raise ValueError("Saved epoch history does not match the selected training budget")
    scores = np.array([row["validation_log_loss"] for row in history], dtype=float)
    if (
        not np.isfinite(scores).all()
        or not np.isclose(score, selection["validation_log_loss"], atol=1e-10, rtol=0)
        or not np.isclose(score, scores.min(), atol=1e-10, rtol=0)
        or selection["best_epoch"] != int(scores.argmin()) + 1
    ):
        raise ValueError("Selected checkpoint does not minimize recorded validation loss")
    temperature = selection["temperature"]
    if (
        not np.isfinite(temperature)
        or temperature <= 0
        or not np.isclose(
            temperature, temperature_fit(raw, store.target[rows]), atol=1e-8, rtol=1e-6
        )
    ):
        raise ValueError("Stored temperature is not the declared validation-only fit")
    reloaded = predict(model, store, rows, columns, max(1, config["batch_size"] - 1))
    error = float(np.max(np.abs(reloaded - raw)))
    if not np.isfinite(error) or error > RELOAD_TOLERANCE:
        raise ValueError("Reloaded checkpoint changed validation logits")
    return error


def audit_result(directory, selection, store, rows, cell, protocol_sha, record=None):
    fresh = record is not None
    record = record if fresh else read(directory / "result.json")
    expected = {
        "cell": cell,
        "variant": cell.split("/")[0],
        "model": selection["model"],
        "columns": selection["columns"],
        "configuration": selection["configuration"],
        "seed": selection["seed"],
        "protocol_sha256": protocol_sha,
        "selection_sha256": file_hash(directory / "selection.json"),
        "prediction_sha256": file_hash(directory / "predictions.npz"),
    }
    if (
        any(record.get(key) != value for key, value in expected.items())
        or record.get("test_evaluated") is not True
    ):
        raise ValueError("Completed result identity changed")
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        identities = {
            "row_id": store.row_id[rows],
            "y_true": store.target[rows],
            "time": store.time[rows],
            "history_count": store.history[rows].astype(bool).sum(-1),
        }
        if any(
            key not in saved or not np.array_equal(saved[key], value)
            for key, value in identities.items()
        ):
            raise ValueError("Final prediction row IDs, labels, times or history counts changed")
        raw, probability = saved["raw"], saved["p_positive"]
    check_raw(raw, len(rows))
    expected_probability = softmax(raw.astype(float) / selection["temperature"], axis=1)[:, 1]
    if probability.shape != (len(rows),) or not np.allclose(
        probability, expected_probability, atol=1e-12, rtol=0
    ):
        raise ValueError("Final probability does not match the saved logits and temperature")
    if record["metrics"] != binary_metrics(store.target[rows], probability) or record[
        "raw_metrics"
    ] != binary_metrics(store.target[rows], softmax(raw.astype(float), axis=1)[:, 1]):
        raise ValueError("Final metrics do not match the saved predictions")
    error = record.get("reload_max_abs_error")
    if (
        not isinstance(error, (float, int))
        or not np.isfinite(error)
        or not 0 <= error <= RELOAD_TOLERANCE
    ):
        raise ValueError("Invalid saved reload check")
    if (
        not fresh
        and (directory / "selection.json").stat().st_mtime_ns
        > (directory / "result.json").stat().st_mtime_ns
    ):
        raise ValueError("Selection was modified after the final test evaluation")
    return record


def build_model(name, width, cfg):
    if name == "history_deepsets":
        return HistoryDeepSets(width, 6, width=cfg["width"], classes=2, dropout=cfg["dropout"])
    if name == "owo":
        return OwO(
            width,
            6,
            team_size=5,
            history_length=10,
            width=cfg["width"],
            heads=2,
            layers=1,
            classes=2,
            domains=0,
            dropout=cfg["dropout"],
            ordered_slots=False,
            pooling="team",
            norm_first=False,
        )
    raise ValueError("Unknown binary history model")


def tensors(store, rows, columns):
    inputs, target = store.batch(rows, columns)
    return tuple(torch.from_numpy(value) for value in inputs), torch.from_numpy(target).long()


def predict(model, store, rows, columns, batch_size):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            inputs, _ = tensors(store, rows[start : start + batch_size], columns)
            values.append(model(*inputs[:3], domain=inputs[3]).numpy())
    raw = np.concatenate(values)
    check_raw(raw, len(rows))
    return raw


def fit(model_name, store, train, valid, columns, config, seed, directory, protocol_sha):
    """Train/checkpoint using only explicit training and validation row indices."""
    validate_config(config)
    validate_rows(store, {"train": train, "valid": valid})
    if (directory / "selection.json").exists() or (directory / "result.json").exists():
        raise ValueError("A selected fit must be reloaded, not overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(seed, config["threads"])
    model = build_model(model_name, len(columns), config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, min_lr=1e-6
    )
    generator = np.random.default_rng(seed)
    history, best, stale, best_epoch, state = [], float("inf"), 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        tick, total = time.perf_counter(), 0.0
        order = generator.permutation(train)
        model.train()
        for start in range(0, len(order), config["batch_size"]):
            batch = order[start : start + config["batch_size"]]
            inputs, y = tensors(store, batch, columns)
            optimizer.zero_grad(set_to_none=True)
            z = model(*inputs[:3], domain=inputs[3])
            loss = torch.nn.functional.cross_entropy(z, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        raw = predict(model, store, valid, columns, config["batch_size"])
        score = binary_metrics(store.target[valid], softmax(raw.astype(float), axis=1)[:, 1])[
            "log_loss"
        ]
        history.append(
            {
                "epoch": epoch,
                "train_log_loss": total / len(train),
                "validation_log_loss": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - tick,
            }
        )
        write_json(directory / "history.json", history)
        print("EPOCH", model_name, len(columns), seed, epoch, score, flush=True)
        if score < best:
            best, stale, best_epoch = score, 0, epoch
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step(score)
        if stale >= config["patience"]:
            break
    model.load_state_dict(state)
    raw = predict(model, store, valid, columns, config["batch_size"])
    selection = {
        "protocol_sha256": protocol_sha,
        "model": model_name,
        "columns": columns,
        "configuration": config,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "validation_log_loss": best,
        "temperature": temperature_fit(raw, store.target[valid]),
        "test_evaluated": False,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.perf_counter() - started,
    }
    torch.save({"state_dict": state, "selection": selection}, directory / "checkpoint.pt")
    np.savez_compressed(
        directory / "validation.npz",
        row_id=store.row_id[valid],
        y_true=store.target[valid],
        raw=raw,
    )
    selection["artifact_sha256"] = {
        name: file_hash(directory / name)
        for name in ["checkpoint.pt", "validation.npz", "history.json"]
    }
    write_json(directory / "selection.json", selection)
    return model, selection


def reload_selected(directory, model_name, columns, config, seed, protocol_sha):
    selection = read(directory / "selection.json")
    expected = {
        "model": model_name,
        "columns": columns,
        "configuration": config,
        "seed": seed,
        "protocol_sha256": protocol_sha,
        "test_evaluated": False,
    }
    if (
        any(selection.get(key) != value for key, value in expected.items())
        or selection.get("test_evaluated") is not False
    ):
        raise ValueError("Saved selection identity changed")
    if set(selection.get("artifact_sha256", {})) != {
        "checkpoint.pt",
        "validation.npz",
        "history.json",
    }:
        raise ValueError("Unexpected selected-artifact inventory")
    for name, checksum in selection["artifact_sha256"].items():
        if file_hash(directory / name) != checksum:
            raise ValueError("Selected artifact changed: " + name)
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint["selection"] != {k: v for k, v in selection.items() if k != "artifact_sha256"}:
        raise ValueError("Checkpoint does not match selected configuration")
    model = build_model(model_name, len(columns), config)
    model.load_state_dict(checkpoint["state_dict"])
    if sum(p.numel() for p in model.parameters()) != selection["parameters"]:
        raise ValueError("Selected model parameter count changed")
    return model, selection


def verify_snapshot(protocol):
    source = OUT / "source-snapshot/gamematch"
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    if (
        digest.hexdigest() != protocol["source_sha256"]
        or file_hash(OUT / "source-driver.py") != protocol["driver_sha256"]
    ):
        raise ValueError("Frozen source changed")


def main():
    global RUN_IDENTITY_VALIDATED
    RUN_IDENTITY_VALIDATED = False
    OUT.mkdir(parents=True, exist_ok=True)
    store = IndexedHistoryStore(DATA)
    rows = {
        part: np.load(DATA / f"{part}-rows.npy", allow_pickle=False)
        for part in ["train", "valid", "test"]
    }
    validate_rows(store, rows)
    if (
        store.history.shape[1:] != (2, 5, 10)
        or store.static.shape[1:] != (2, 5, 6)
        or store.performance.shape[1] != 24
    ):
        raise ValueError(
            "This driver requires the declared 5v5, ten-step, 24-performance/6-static schema"
        )
    variants = {"outcome_history": [0, 1, 24], "performance_history": list(range(25))}
    config = dict(CONFIG)
    validate_config(config)
    protocol = {
        "source_sha256": code_hash(),
        "driver_sha256": file_hash(Path(__file__)),
        "data_manifest_sha256": file_hash(DATA / "manifest.json"),
        "runtime": runtime_identity(),
        "models": ["history_deepsets", "owo"],
        "variants": variants,
        "seeds": SEEDS,
        "configuration": config,
        "counts": {p: len(r) for p, r in rows.items()},
        "classes": 2,
        "target": "team100 win=1; team200 win=0",
        "selection": "Fixed configuration. Validation raw LogLoss selects epoch; positive temperature fitted on validation only. No outcome-based fit pruning.",
        "scope": "Exploratory richer-information comparison. Same 12-epoch cap, seeds and dropout; unequal parameter counts. Prior outcomes/time versus prior individual performance+role, with identical static outcome summaries.",
        "architecture": "HistoryDeepSets; OwO adaptation with team pooling, one layer, two heads, no domain tokens or current role/draft. Binary output heads are explicit.",
        "data_limitations": "Active-player sample; reconstructed match contexts, missing native match IDs. Single ranked mode does not test CHAMP cross-domain transfer.",
        "execution_order": "Seed-major; all four model/information cells before next seed.",
    }
    marker = OUT / "protocol.json"
    if marker.exists():
        if read(marker) != protocol:
            raise ValueError(
                "Frozen neural protocol changed; resume with archived driver and library"
            )
    else:
        if any(OUT.iterdir()):
            raise ValueError("Nonempty output without a sealed protocol; use a new output")
        shutil.copytree(
            Path(gamematch.__file__).parent,
            OUT / "source-snapshot/gamematch",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copyfile(__file__, OUT / "source-driver.py")
        np.savez_compressed(
            OUT / "split-identity.npz",
            **{part: store.row_id[index] for part, index in rows.items()},
        )
        verify_snapshot(protocol)
        write_json(marker, protocol)
    verify_snapshot(protocol)
    with np.load(OUT / "split-identity.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(rows) or any(
            not np.array_equal(saved[part], store.row_id[index]) for part, index in rows.items()
        ):
            raise ValueError("Frozen split identities changed")
    RUN_IDENTITY_VALIDATED = True
    protocol_sha, completed = file_hash(marker), []
    torch.set_num_threads(config["threads"])
    for seed in protocol["seeds"]:
        for name in protocol["models"]:
            for variant, columns in variants.items():
                directory = OUT / variant / name / f"seed-{seed}"
                cell = f"{variant}/{name}/seed-{seed}"
                write_json(
                    OUT / "status.json",
                    {
                        "status": "running",
                        "cell": cell,
                        "completed": completed,
                        "updated_utc": stamp(),
                    },
                )
                if not (directory / "selection.json").exists():
                    if (directory / "result.json").exists():
                        raise ValueError("Completed result has lost its locked selection")
                    print("FIT", cell, flush=True)
                    fit(
                        name,
                        store,
                        rows["train"],
                        rows["valid"],
                        columns,
                        config,
                        seed,
                        directory,
                        protocol_sha,
                    )
                model, selection = reload_selected(
                    directory, name, columns, config, seed, protocol_sha
                )
                error = audit_validation(
                    directory, model, selection, store, rows["valid"], columns, config
                )
                result_path = directory / "result.json"
                if result_path.exists():
                    audit_result(directory, selection, store, rows["test"], cell, protocol_sha)
                else:
                    raw = predict(model, store, rows["test"], columns, config["batch_size"])
                    p = softmax(raw.astype(float) / selection["temperature"], axis=1)[:, 1]
                    y = store.target[rows["test"]]
                    np.savez_compressed(
                        directory / "predictions.npz",
                        row_id=store.row_id[rows["test"]],
                        y_true=y,
                        raw=raw,
                        p_positive=p,
                        time=store.time[rows["test"]],
                        history_count=store.history[rows["test"]].astype(bool).sum(-1),
                    )
                    result = {
                        "cell": cell,
                        "variant": variant,
                        "model": name,
                        "columns": columns,
                        "configuration": config,
                        "seed": seed,
                        "protocol_sha256": protocol_sha,
                        "selection_sha256": file_hash(directory / "selection.json"),
                        "prediction_sha256": file_hash(directory / "predictions.npz"),
                        "metrics": binary_metrics(y, p),
                        "raw_metrics": binary_metrics(y, softmax(raw.astype(float), axis=1)[:, 1]),
                        "reload_max_abs_error": error,
                        "test_evaluated": True,
                        "completed_utc": stamp(),
                    }
                    result = audit_result(
                        directory, selection, store, rows["test"], cell, protocol_sha, result
                    )
                    write_json(result_path, result)
                    print(
                        "DONE",
                        cell,
                        result["metrics"]["log_loss"],
                        result["metrics"]["accuracy"],
                        flush=True,
                    )
                completed.append(cell)
                del model
    write_json(
        OUT / "status.json", {"status": "completed", "completed": completed, "updated_utc": stamp()}
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if RUN_IDENTITY_VALIDATED:
            write_json(
                OUT / "status.json",
                {
                    "status": "failed",
                    "error": type(error).__name__ + ": " + str(error),
                    "updated_utc": stamp(),
                },
            )
        raise
