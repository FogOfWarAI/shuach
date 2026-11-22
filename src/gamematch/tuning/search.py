# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Bounded, resumable search with immutable spaces and validation-only selection."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy import sparse
from scipy.special import expit, softmax

from ..asymmetric_runner import ordinal_metrics
from ..configuration import runtime_identity
from ..data.heroes import file_hash
from ..evaluation.binary import binary_metrics
from ..evaluation.multiclass import multiclass_metrics
from ..history_runner import temperature_fit
from ..models.tabular import hero_counts, require_logistic_convergence
from ..runner import code_hash, write_json
from ..skill_runner import regression_metrics
from .data import load_bundle, partition
from .engine import evaluate, fit, predict_raw, primary_loss, reload_model

MATRIX = {
    "dbd-2022": ["logistic", "lightgbm", "mlp", "ordinal"],
    "dbd-2023": ["logistic", "lightgbm", "mlp", "ordinal"],
    "future-rating": ["ridge", "mlp", "gru", "transformer", "mmrnet"],
    "uci-dota2": ["logistic", "lightgbm", "fm", "mlp", "deepsets", "neuralac"],
    "neuralac-dota2018": ["logistic", "lightgbm", "fm", "mlp", "deepsets", "neuralac"],
    "neuralac-lol": ["logistic", "lightgbm", "fm", "mlp", "deepsets", "neuralac"],
    "overwatch": ["static_deepsets", "history_deepsets", "owo_transformer", "owo", "dawn"],
    "synthetic-additive": ["logistic", "fm", "mlp", "deepsets", "neuralac"],
    "synthetic-cooperative": ["logistic", "fm", "mlp", "deepsets", "neuralac"],
}


def candidates(dataset, name):
    tabular = name in {"logistic", "lightgbm", "ridge"}
    common = {"threads": 4, "tabular": tabular}
    if name == "logistic":
        return [{**common, "C": x} for x in [1.0, 0.001, 0.01, 0.1, 0.3, 3.0, 10.0, 100.0]]
    if name == "ridge":
        return [
            {**common, "alpha": x} for x in [1.0, 0.001, 0.01, 0.1, 10.0, 100.0, 1000.0, 10000.0]
        ]
    if name == "lightgbm":
        settings = [
            (31, 0.05, 0.0, 20),
            (7, 0.05, 1.0, 100),
            (15, 0.03, 1.0, 50),
            (31, 0.03, 10.0, 100),
            (63, 0.03, 10.0, 100),
            (15, 0.1, 0.0, 50),
            (31, 0.1, 1.0, 100),
            (63, 0.05, 1.0, 50),
        ]
        return [
            {
                **common,
                "num_leaves": n,
                "learning_rate": lr,
                "reg_lambda": penalty,
                "min_child_samples": m,
            }
            for n, lr, penalty, m in settings
        ]
    history, skill, asymmetric = (
        dataset == "overwatch",
        dataset == "future-rating",
        dataset.startswith("dbd"),
    )
    cap = 30 if history else 60 if skill or asymmetric else 150 if name == "neuralac" else 80
    if dataset.startswith("synthetic") and name == "neuralac":
        cap = 200
    batch = 128 if history else 256 if skill else 1024 if asymmetric or name == "neuralac" else 2048
    device = "cpu" if asymmetric or name == "gru" else "mps"
    base = {
        **common,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "optimizer": "adam",
        "scheduler": False,
        "dropout": 0.2
        if name == "neuralac"
        else 0.1
        if skill or name in {"owo", "dawn", "owo_transformer"}
        else 0.0,
        "hidden_dim": 64,
        "embedding_dim": 20,
        "interaction_hidden": 50,
        "width": 32,
        "heads": 4,
        "layers": 2,
        "epochs": cap,
        "patience": 8,
        "batch_size": batch,
        "device": device,
        "loss": "mse",
    }
    # Seven fixed configurations span zero/weak/strong regularization and capacity.
    # Published before any new trial score; no changes based on test results.
    settings = [
        (0.003, 0.0, 0.0, 32, 16, 1, "adam", False, "mse"),
        (0.001, 0.001, 0.1, 64, 32, 2, "adamw", True, "huber"),
        (0.0003, 0.01, 0.2, 128, 64, 2, "adamw", True, "mae"),
        (0.003, 0.001, 0.35, 128, 32, 1, "adamw", True, "huber"),
        (0.001, 0.01, 0.5, 256, 64, 2, "adamw", True, "mae"),
        (0.0003, 0.0001, 0.1, 32, 16, 1, "adam", True, "mae"),
        (0.003, 0.0001, 0.2, 64, 32, 2, "adam", True, "mse"),
    ]
    out = [base]
    for i, (lr, wd, drop, hidden, width, layers, opt, schedule, loss) in enumerate(settings):
        cfg = {
            **base,
            "learning_rate": lr,
            "weight_decay": wd,
            "dropout": drop,
            "hidden_dim": hidden,
            "width": width,
            "layers": layers,
            "optimizer": opt,
            "scheduler": schedule,
            "loss": loss,
            "embedding_dim": [8, 20, 32, 32, 20, 8, 32][i],
            "interaction_hidden": [32, 50, 96, 50, 96, 32, 50][i],
        }
        if history:
            cfg["batch_size"] = 256 if i % 2 == 0 else 512
        elif skill:
            cfg["batch_size"] = 128 if i % 2 == 0 else 512
        elif name != "neuralac":
            cfg["batch_size"] = 512 if i % 2 == 0 else 2048
        if name == "fm":
            cfg["dropout"] = 0.0  # FM has no hidden activation dropout.
        out.append(cfg)
    return out


def now():
    return datetime.now(timezone.utc).isoformat()


def search_spaces(device=None):
    """Preserve the historical plan unless a usable neural device is explicit."""
    if device is not None:
        if not isinstance(device, str) or device not in {"cpu", "mps", "cuda"}:
            raise ValueError("Tuning device must be cpu, mps, cuda or None")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Requested MPS device is unavailable; use --device cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested CUDA device is unavailable; use --device cpu")
    return {
        dataset: {
            name: [
                {**cfg, "device": device} if device is not None and not cfg["tabular"] else cfg
                for cfg in candidates(dataset, name)
            ]
            for name in names
        }
        for dataset, names in MATRIX.items()
    }


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def select_trial(records, expected_seeds):
    """Consume only explicit validation fields, including when extra fields exist."""
    eligible = []
    for trial, items in records.items():
        if sorted(r["seed"] for r in items) != sorted(expected_seeds):
            continue
        if any(r.get("test_evaluated") is not False for r in items):
            raise ValueError("Search records must not have evaluated test data")
        scores = [r["validation_primary"] for r in items]
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite validation score")
        eligible.append((float(np.mean(scores)), trial))
    if not eligible:
        raise ValueError("No complete validation-only trial")
    return min(eligible)[1]


def audit_primary_split(root, bundle, world):
    """Require exact original test IDs, labels and grouping, before fitting."""
    name = bundle.name
    if name.startswith("synthetic-"):
        source = root / f"runs/decisions-v1/{name[10:]}-seed-{world}/logistic/predictions.npz"
    elif name in {"uci-dota2", "neuralac-dota2018", "neuralac-lol"}:
        source = root / f"runs/local-full-heroes-v1/{name}/seed-0-logistic/predictions.npz"
    elif name.startswith("dbd-"):
        source = root / f"runs/asymmetric-v1/{name}/seed-0-logistic/predictions.npz"
    elif name == "overwatch":
        source = root / "runs/overwatch-history-v1/seed-0-history_deepsets/predictions.npz"
    else:
        source = root / "runs/skill-v1/seed-0-gru/predictions.npz"
    with np.load(source, allow_pickle=False) as old:
        test = bundle.parts["test"]
        target = test.metadata["y_true"] if bundle.task == "regression" else test.target
        if not np.array_equal(test.row_id, old["row_id"]) or not np.array_equal(
            target, old["y_true"]
        ):
            raise ValueError(f"Primary test identity changed: {name}")
    return {"primary_prediction_sha256": file_hash(source), "test_identity_equal": True}


def save_data_identity(directory, bundle, audit):
    directory.mkdir(exist_ok=True, parents=True)
    provenance = dict(bundle.provenance)
    preprocessor = provenance.pop("preprocessor", None)
    value = {
        "name": bundle.name,
        "task": bundle.task,
        "specification": bundle.specification,
        "provenance": provenance,
        "split_counts": {p: len(d.target) for p, d in bundle.parts.items()},
        "input_identity": {p: partition_identity(d) for p, d in bundle.parts.items()},
        "preprocessor_identity": joblib.hash(preprocessor) if preprocessor is not None else None,
        **audit,
    }
    path = directory / "data-manifest.json"
    arrays = {}
    for part, data in bundle.parts.items():
        arrays.update({f"{part}_row_id": data.row_id, f"{part}_target": data.target})
        arrays.update({f"{part}_metadata_{key}": value for key, value in data.metadata.items()})
    expected_files = {"split-manifest.npz"}
    if preprocessor is not None:
        expected_files.add("preprocessor.joblib")
    if path.exists():
        saved = json.loads(path.read_text())
        if {k: v for k, v in saved.items() if k != "artifact_sha256"} != value:
            raise ValueError("Data manifest changed")
        check_artifacts(directory, saved.get("artifact_sha256"), expected_files)
        with np.load(directory / "split-manifest.npz", allow_pickle=False) as split:
            if set(split.files) != set(arrays) or any(
                not np.array_equal(split[key], expected) for key, expected in arrays.items()
            ):
                raise ValueError("Saved data split identities changed")
        if (
            preprocessor is not None
            and joblib.hash(joblib.load(directory / "preprocessor.joblib"))
            != value["preprocessor_identity"]
        ):
            raise ValueError("Saved preprocessor identity changed")
        return
    if any(directory.iterdir()):
        raise ValueError("Incomplete data identity; use a new output")
    if preprocessor is not None:
        joblib.dump(preprocessor, directory / "preprocessor.joblib")
    np.savez_compressed(directory / "split-manifest.npz", **arrays)
    value["artifact_sha256"] = {
        name: file_hash(directory / name) for name in sorted(expected_files)
    }
    write_json(path, value)


def array_identity(value):
    array = np.ascontiguousarray(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
    if array.dtype.hasobject:
        raise ValueError("Object arrays cannot identify immutable tuning data")
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
    }


def partition_identity(part):
    tabular = part.tabular
    if sparse.issparse(tabular):
        matrix = tabular.tocsr()
        tabular = {
            "shape": list(matrix.shape),
            **{key: array_identity(getattr(matrix, key)) for key in ["data", "indices", "indptr"]},
        }
    elif tabular is not None:
        tabular = array_identity(tabular)
    return {"inputs": [array_identity(value) for value in part.inputs], "tabular": tabular}


def require_identity(record, expected, context):
    if any(
        key not in record
        or (record[key] is not value if isinstance(value, bool) else record[key] != value)
        for key, value in expected.items()
    ):
        raise ValueError("Invalid " + context + " identity")


def check_artifacts(directory, hashes, expected):
    if not isinstance(hashes, dict) or set(hashes) != set(expected):
        raise ValueError("Missing or unexpected fit/data artifact inventory")
    for name, digest in hashes.items():
        if not (directory / name).is_file() or file_hash(directory / name) != digest:
            raise ValueError("Missing or changed artifact: " + str(directory / name))


def audit_fit(directory, bundle, name, cfg, seed):
    """Validate a completed fit without repeating neural forward inference."""
    record = json.loads((directory / "metrics.json").read_text())
    require_identity(
        record,
        {
            "model": name,
            "seed": seed,
            "configuration": cfg,
            "test_evaluated": False,
            "selection_metric": "mae" if bundle.task == "regression" else "raw_log_loss",
        },
        "fit",
    )
    files = (
        {"checkpoint.joblib", "validation.npz"}
        if cfg["tabular"]
        else {"checkpoint.pt", "validation.npz", "history.json"}
    )
    check_artifacts(directory, record.get("artifact_sha256"), files)
    valid = bundle.parts["valid"]
    with np.load(directory / "validation.npz", allow_pickle=False) as saved:
        if (
            set(saved.files) != {"row_id", "y_true", "raw"}
            or not np.array_equal(saved["row_id"], valid.row_id)
            or not np.array_equal(saved["y_true"], valid.target)
        ):
            raise ValueError("Validation prediction identities changed")
        raw = saved["raw"]
    score = primary_loss(bundle.task, valid.target, raw, bundle.specification)
    if not np.isclose(score, record["validation_primary"], rtol=0, atol=1e-10):
        raise ValueError("Validation score does not match saved raw predictions")
    if cfg["tabular"]:
        model = joblib.load(directory / "checkpoint.joblib")
        parameters = {
            key: cfg[key]
            for key in [
                "C",
                "alpha",
                "num_leaves",
                "learning_rate",
                "reg_lambda",
                "min_child_samples",
            ]
            if key in cfg
        }
        if name != "ridge":
            parameters["random_state"] = seed
            expected_classes = (
                [0, 1] if bundle.task == "binary" else list(range(bundle.specification["classes"]))
            )
            if not np.array_equal(model.classes_, expected_classes):
                raise ValueError("Tabular checkpoint classes changed")
        require_identity(model.get_params(), parameters, "tabular checkpoint")
        epoch = (
            require_logistic_convergence(model)
            if name == "logistic"
            else int(model.best_iteration_)
            if name == "lightgbm"
            else 0
        )
        if (
            model.n_features_in_ != valid.tabular.shape[1]
            or record["best_epoch"] != epoch
            or record["epochs_completed"] != 0
            or record["parameter_count"] is not None
        ):
            raise ValueError("Tabular checkpoint training metadata changed")
    else:
        checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        require_identity(
            checkpoint,
            {
                "model": name,
                "task": bundle.task,
                "specification": bundle.specification,
                "configuration": cfg,
                "seed": seed,
                "best_epoch": record["best_epoch"],
            },
            "checkpoint",
        )
        history = json.loads((directory / "history.json").read_text())
        scores = np.asarray([row["validation_primary"] for row in history])
        if (
            not 1 <= len(history) <= cfg["epochs"]
            or len(history) != record["epochs_completed"]
            or [row["epoch"] for row in history] != list(range(1, len(history) + 1))
            or not np.isfinite(scores).all()
            or record["best_epoch"] != int(scores.argmin()) + 1
            or not np.isclose(score, scores.min(), rtol=0, atol=1e-6)
        ):
            raise ValueError("Checkpoint selection does not match validation history")
    return record


def audit_decisions(directory, bundle, record):
    with (
        np.load(bundle.provenance["decision_environment"], allow_pickle=False) as environment,
        np.load(directory / "decisions.npz", allow_pickle=False) as saved,
    ):
        selected = saved["selected"]
        truth, oracle, lobbies = (
            environment[key] for key in ["true_probability", "oracle_cost", "lobbies"]
        )
        if (
            selected.shape != (len(truth),)
            or not np.issubdtype(selected.dtype, np.integer)
            or np.any((selected < 0) | (selected >= truth.shape[1]))
        ):
            raise ValueError("Invalid saved decision indices")
        imbalance = np.abs(truth[np.arange(len(truth)), selected] - 0.5)
        expected = {
            "selected": selected,
            "true_imbalance": imbalance,
            "regret": imbalance - oracle,
            "oracle_cost": oracle,
            "lobbies": lobbies,
        }
        if set(saved.files) != set(expected) or any(
            not np.array_equal(saved[key], value) for key, value in expected.items()
        ):
            raise ValueError("Decision evidence changed")
    metrics = {
        "mean_decision_regret": float((imbalance - oracle).mean()),
        "mean_true_imbalance": float(imbalance.mean()),
        "fraction_true_40_60": float((imbalance <= 0.1).mean()),
        "solver": "exact_252",
        "score": "raw",
    }
    if record.get("decision_metrics") != metrics:
        raise ValueError("Decision metrics do not match saved decisions")


def audit_final(directory, bundle, fit_record, identity, synthetic, record=None):
    fresh = record is not None
    record = record if fresh else json.loads((directory / "result.json").read_text())
    require_identity(
        record,
        {
            **{key: value for key, value in fit_record.items() if key != "test_evaluated"},
            **identity,
            "test_evaluated": True,
        },
        "final result",
    )
    files = {"predictions.npz", "decisions.npz"} if synthetic else {"predictions.npz"}
    check_artifacts(directory, record.get("evaluation_artifact_sha256"), files)
    test = bundle.parts["test"]
    target = test.metadata["y_true"] if bundle.task == "regression" else test.target
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        expected = {"row_id": test.row_id, **test.metadata, "y_true": target}
        output = (
            "prediction"
            if bundle.task == "regression"
            else "p_positive"
            if bundle.task == "binary"
            else "probabilities"
        )
        if set(saved.files) != set(expected) | {"raw", output} or any(
            not np.array_equal(saved[key], value) for key, value in expected.items()
        ):
            raise ValueError("Final prediction row, label or metadata identities changed")
        raw, probability = saved["raw"], saved[output]
    primary_loss(bundle.task, test.target, raw, bundle.specification)
    temperature = record.get("temperature")
    if bundle.task == "regression":
        expected_probability = (
            test.metadata["current_mu"]
            + raw * bundle.specification["delta_scale"]
            + bundle.specification["delta_mean"]
        )
        metrics = regression_metrics(target, probability)
        uncalibrated = metrics
    elif bundle.task == "binary":
        expected_probability = expit(raw)
        metrics = binary_metrics(target, probability)
        uncalibrated = metrics
    else:
        with np.load(directory / "validation.npz", allow_pickle=False) as valid:
            fitted_temperature = temperature_fit(valid["raw"], bundle.parts["valid"].target)
        if (
            not isinstance(temperature, (int, float))
            or not np.isfinite(temperature)
            or temperature <= 0
            or not np.isclose(temperature, fitted_temperature, rtol=1e-6, atol=1e-8)
        ):
            raise ValueError("Final temperature is not validation-only calibration")
        expected_probability = softmax(raw / temperature, axis=1)
        function = ordinal_metrics if bundle.task == "multiclass" else multiclass_metrics
        metrics, uncalibrated = (
            function(target, probability),
            function(target, softmax(raw, axis=1)),
        )
    if bundle.task in {"binary", "regression"} and temperature is not None:
        raise ValueError("Unexpected calibration temperature")
    if (
        probability.shape != expected_probability.shape
        or not np.array_equal(probability, expected_probability)
        or record.get("metrics") != metrics
        or record.get("uncalibrated_metrics") != uncalibrated
    ):
        raise ValueError("Final probabilities or complete metrics do not match predictions")
    error = record.get("reload_max_abs_error")
    with np.load(directory / "validation.npz", allow_pickle=False) as valid:
        tolerance = 1e-6 + 1e-5 * float(np.max(np.abs(valid["raw"])))
    if not isinstance(error, (float, int)) or not np.isfinite(error) or not 0 <= error <= tolerance:
        raise ValueError("Invalid checkpoint reload check")
    if synthetic:
        audit_decisions(directory, bundle, record)
    if (
        not fresh
        and (directory.parent.parent / "selection.json").stat().st_mtime_ns
        > (directory / "result.json").stat().st_mtime_ns
    ):
        raise ValueError("Selection was modified after final evaluation")
    return record


def decision_evaluation(model, bundle, cfg, directory):
    with np.load(bundle.provenance["decision_environment"], allow_pickle=False) as f:
        lobbies, template, truth, oracle = (
            f["lobbies"],
            f["candidate_template"],
            f["true_probability"],
            f["oracle_cost"],
        )
    roster = lobbies[:, template].reshape(-1, 2, 5)
    values = np.asarray(bundle.specification["vocabulary"])
    # HeroVocabulary reserves zero, and stores observed tokens only.
    from ..data.heroes import HeroVocabulary

    vocab = HeroVocabulary()
    vocab.values = values
    encoded = vocab.transform(roster)
    part = partition(
        [encoded],
        np.zeros(len(roster), dtype=np.float32),
        np.arange(len(roster)),
        {},
        hero_counts(encoded, vocab.size),
    )
    from scipy.special import expit

    p = expit(predict_raw(model, part, "binary", cfg)).reshape(len(lobbies), -1)
    selected = np.abs(p - 0.5).argmin(1)
    imbalance = np.abs(truth[np.arange(len(lobbies)), selected] - 0.5)
    regret = imbalance - oracle
    np.savez_compressed(
        Path(directory) / "decisions.npz",
        selected=selected,
        regret=regret,
        true_imbalance=imbalance,
        oracle_cost=oracle,
        lobbies=lobbies,
    )
    return {
        "mean_decision_regret": float(regret.mean()),
        "mean_true_imbalance": float(imbalance.mean()),
        "fraction_true_40_60": float((imbalance <= 0.1).mean()),
        "solver": "exact_252",
        "score": "raw",
    }


def run(root, output, dataset_filter=None, model_filter=None, device=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if dataset_filter and set(dataset_filter) - set(MATRIX):
        raise ValueError("Unknown tuning dataset")
    if model_filter and set(model_filter) - {m for models in MATRIX.values() for m in models}:
        raise ValueError("Unknown tuning model")
    spaces = search_spaces(device)
    output.mkdir(exist_ok=True, parents=True)
    protocol = {
        "schema_version": 2,
        "resume_integrity": "Required fit/evaluation hashes; validation-score and final-metric recomputation; locked complete candidate and seed inventories.",
        "spaces": spaces,
        "device_override": device,
        "stage1_seed": 100,
        "confirmation_seed": 101,
        "confirmation_top_k": 3,
        "final_seeds": [0, 1, 2, 3, 4],
        "selection": "Raw validation NLL; rating MAE in original units. Top 3 rechecked on second seed, select by two-seed mean.",
        "synthetic_selection": "Stage 1 world 0, confirmation world 1; final worlds 0..4. Validation NLL only; no oracle regret selection.",
        "baseline_repeats": "Deterministic real-data tabular final fit once; synthetic repeats vary world, all 5 retained.",
        "prior_test_access": True,
        "evidence": "Exploratory follow-up; existing test sets already observed before search design.",
        "calibration": "After model/configuration selection, fit temperature on original validation for Overwatch/DBD, consistent with primary evaluation.",
        "test_policy": "No trial receives test data; selection is written and hashed before any final test evaluation. No adaptive space edits.",
        "code_sha256": code_hash(),
        "runtime": runtime_identity(),
    }
    mark = output / "protocol.json"
    if mark.exists():
        if json.loads(mark.read_text()) != protocol:
            raise ValueError("Immutable protocol/code/runtime mismatch; use a new output")
    else:
        if any(output.iterdir()):
            raise ValueError("Nonempty tuning output without a sealed protocol")
        shutil.copytree(
            Path(__file__).parents[1],
            output / "source-snapshot/gamematch",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        write_json(mark, protocol)
    digest = hashlib.sha256()
    source = output / "source-snapshot/gamematch"
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != protocol["code_sha256"]:
        raise ValueError("Frozen tuning source changed")
    for dataset, names in MATRIX.items():
        if dataset_filter and dataset not in dataset_filter:
            continue
        if model_filter and not set(names).intersection(model_filter):
            continue
        bundles = {}

        def get_bundle(world=0):
            if world not in bundles:
                b = load_bundle(root, dataset, world)
                audit = audit_primary_split(root, b, world)
                save_data_identity(output / dataset / f"data-world-{world}", b, audit)
                bundles[world] = b
            return bundles[world]

        # Completed jobs must still reject changed input data when resumed.
        for world in range(5) if dataset.startswith("synthetic-") else [0]:
            get_bundle(world)

        for name in names:
            if model_filter and name not in model_filter:
                continue
            target = output / dataset / name
            target.mkdir(parents=True, exist_ok=True)
            status_path = target / "status.json"
            previous = json.loads(status_path.read_text()) if status_path.exists() else None
            if previous is not None:
                require_identity(previous, {"dataset": dataset, "model": name}, "status")
            completed = previous is not None and previous.get("status") == "completed"
            selection_file = target / "selection.json"
            locked = selection_file.exists()
            if (completed or (target / "final").exists()) and not locked:
                raise ValueError("Final or completed run has lost its locked selection")
            status = {
                "dataset": dataset,
                "model": name,
                "status": "searching",
                "started_utc": now(),
            }
            if not completed:
                write_json(status_path, status)
            configs = spaces[dataset][name]
            synthetic = dataset.startswith("synthetic-")
            two_seeds = not configs[0]["tabular"] or synthetic
            final_seeds = [0, 1, 2, 3, 4] if two_seeds else [0]
            final_root = target / "final"
            expected_final = {f"seed-{seed}" for seed in final_seeds}
            actual_final = (
                {p.name for p in final_root.iterdir() if p.is_dir()}
                if final_root.exists()
                else set()
            )
            if actual_final - expected_final or (completed and actual_final != expected_final):
                raise ValueError("Completed final seed inventory is missing or unexpected")
            if completed:
                expected_results = {f"final/seed-{seed}/result.json" for seed in final_seeds}
                check_artifacts(target, previous.get("final_result_sha256"), expected_results)
            records = {}

            def trial(i, seed, world):
                directory = target / "search" / f"trial-{i:02d}" / f"seed-{seed}"
                metrics = directory / "metrics.json"
                if (directory / "predictions.npz").exists() or (directory / "result.json").exists():
                    raise ValueError("Search fit contains test artifacts")
                b = get_bundle(world)
                if metrics.exists():
                    return audit_fit(directory, b, name, configs[i], seed)
                if locked or completed:
                    raise ValueError("Locked selection is missing a completed search fit")
                print(f"SEARCH {dataset}/{name} trial={i} seed={seed}", flush=True)
                model, r = fit(
                    b.task,
                    name,
                    b.specification,
                    b.parts["train"],
                    b.parts["valid"],
                    configs[i],
                    seed,
                    directory,
                )
                del model
                r = audit_fit(directory, b, name, configs[i], seed)
                print(
                    f"VALID {dataset}/{name} trial={i} seed={seed} score={r['validation_primary']:.7f} seconds={r['seconds']:.1f}",
                    flush=True,
                )
                return r

            try:
                for i in range(len(configs)):
                    records[i] = [trial(i, 100, 0)]
                shortlist = sorted(records, key=lambda i: (records[i][0]["validation_primary"], i))[
                    :3
                ]
                if two_seeds:
                    for i in shortlist:
                        records[i].append(trial(i, 101, 1 if synthetic else 0))
                if {p.name for p in (target / "search").iterdir() if p.is_dir()} != {
                    f"trial-{i:02d}" for i in range(len(configs))
                }:
                    raise ValueError("Search trial inventory changed")
                search_hashes = {}
                for i in range(len(configs)):
                    trial_path = target / "search" / f"trial-{i:02d}"
                    expected_seeds = {100, 101} if two_seeds and i in shortlist else {100}
                    if {p.name for p in trial_path.iterdir() if p.is_dir()} != {
                        f"seed-{seed}" for seed in expected_seeds
                    }:
                        raise ValueError("Search seed inventory changed")
                    for seed in sorted(expected_seeds):
                        path = trial_path / f"seed-{seed}" / "metrics.json"
                        search_hashes[str(path.relative_to(target))] = file_hash(path)
                winner = select_trial(records, [100, 101] if two_seeds else [100])
                selection = {
                    "dataset": dataset,
                    "model": name,
                    "task": get_bundle().task,
                    "trial": winner,
                    "configuration": configs[winner],
                    "selection_seeds": [100, 101] if two_seeds else [100],
                    "validation_mean": float(
                        np.mean([r["validation_primary"] for r in records[winner]])
                    ),
                    "shortlist": shortlist,
                    "protocol_sha256": file_hash(mark),
                    "search_metrics_sha256": search_hashes,
                    "data_manifest_sha256": {
                        str(world): file_hash(
                            output / dataset / f"data-world-{world}" / "data-manifest.json"
                        )
                        for world in bundles
                    },
                    "test_used": False,
                }
                if selection_file.exists() and json.loads(selection_file.read_text()) != selection:
                    raise ValueError("Selection changed after lock")
                if not selection_file.exists():
                    write_json(selection_file, selection)
                status.update(
                    status="final_evaluation",
                    selection=winner,
                    selection_sha256=file_hash(selection_file),
                )
                if not completed:
                    write_json(status_path, status)
                elif previous.get("selection") != winner or previous.get(
                    "selection_sha256"
                ) != file_hash(selection_file):
                    raise ValueError("Completed status does not match the locked selection")
                for seed in final_seeds:
                    directory = target / "final" / f"seed-{seed}"
                    result_path = directory / "result.json"
                    world = seed if synthetic else 0
                    b = get_bundle(world)
                    cfg = configs[winner]
                    identity = {
                        "dataset": dataset,
                        "task": b.task,
                        "specification": b.specification,
                        "world": world,
                        "protocol_sha256": file_hash(mark),
                        "data_manifest_sha256": file_hash(
                            output / dataset / f"data-world-{world}" / "data-manifest.json"
                        ),
                        "selection_sha256": file_hash(selection_file),
                    }
                    model = None
                    if (directory / "metrics.json").exists():
                        r = audit_fit(directory, b, name, cfg, seed)
                    else:
                        if completed or result_path.exists():
                            raise ValueError("Completed result is missing its final fit")
                        model, r = fit(
                            b.task,
                            name,
                            b.specification,
                            b.parts["train"],
                            b.parts["valid"],
                            cfg,
                            seed,
                            directory,
                        )
                        r = audit_fit(directory, b, name, cfg, seed)
                    identity["fit_metrics_sha256"] = file_hash(directory / "metrics.json")
                    if result_path.exists():
                        audit_final(directory, b, r, identity, synthetic)
                        continue
                    if completed:
                        raise ValueError("Completed seed is missing its final result")
                    print(f"FINAL {dataset}/{name} trial={winner} seed={seed}", flush=True)
                    if model is None:
                        model = reload_model(directory, b.task, name, b.specification, cfg)
                    result = {
                        **r,
                        **evaluate(model, b, cfg, directory),
                        **identity,
                        "test_evaluated": True,
                    }
                    if synthetic:
                        result["decision_metrics"] = decision_evaluation(model, b, cfg, directory)
                    # Verify reload predictions on validation examples without test-based selection.
                    reloaded = reload_model(directory, b.task, name, b.specification, cfg)
                    a = predict_raw(model, b.parts["valid"], b.task, cfg)
                    other = predict_raw(reloaded, b.parts["valid"], b.task, cfg)
                    if not np.allclose(a, other, atol=1e-6, rtol=1e-5):
                        raise ValueError("Checkpoint reload discrepancy")
                    result["reload_max_abs_error"] = float(np.max(np.abs(a - other)))
                    result["evaluation_artifact_sha256"] = {
                        filename: file_hash(directory / filename)
                        for filename in (
                            ["predictions.npz", "decisions.npz"]
                            if synthetic
                            else ["predictions.npz"]
                        )
                    }
                    audit_final(directory, b, r, identity, synthetic, result)
                    write_json(result_path, result)
                    del model, reloaded
                if not completed:
                    status.update(
                        status="completed",
                        completed_utc=now(),
                        final_result_sha256={
                            f"final/seed-{seed}/result.json": file_hash(
                                target / "final" / f"seed-{seed}" / "result.json"
                            )
                            for seed in final_seeds
                        },
                    )
                    write_json(status_path, status)
            except Exception as exc:
                status.update(
                    status="failed", error=f"{type(exc).__name__}: {exc}", updated_utc=now()
                )
                if not completed:
                    write_json(status_path, status)
                raise
        bundles.clear()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    print("REQUESTED MATRIX COMPLETED", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default=".")
    p.add_argument("--output", default="runs/hpo-v1")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--models", nargs="+")
    p.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        help="Override neural candidates only; omitted preserves the historical per-model device plan",
    )
    args = p.parse_args()
    run(args.workspace, args.output, args.datasets, args.models, device=args.device)


if __name__ == "__main__":
    main()
