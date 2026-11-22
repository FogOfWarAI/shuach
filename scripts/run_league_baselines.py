# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Validation-selected baselines on the new compact, pre-match LoL histories."""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import gamematch
from gamematch.configuration import runtime_identity
from gamematch.data.heroes import file_hash
from gamematch.data.indexed_history import IndexedHistoryStore
from gamematch.evaluation.binary import binary_metrics
from gamematch.models.tabular import require_logistic_convergence
from gamematch.runner import code_hash, write_json

ROOT = Path(os.environ.get("GAMEMATCH_WORKSPACE", Path(__file__).resolve().parents[1])).resolve()
OUT = ROOT / "runs/league-baselines-v1"
DATA = ROOT / "datasets/derived/leagueData-v1"
RUN_IDENTITY_VALIDATED = False
RELOAD_TOLERANCE = 1e-8


def read(path):
    return json.loads(path.read_text())


def source_hash(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require_identity(record, expected, context):
    if any(key not in record or record[key] != value for key, value in expected.items()):
        raise ValueError("Artifact identity changed: " + context)


def positive_probability(model, features):
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Expected ordered binary probability columns [0, 1]")
    with threadpool_limits(limits=2):
        return model.predict_proba(features)[:, 1]


def prediction_archive(path, row_ids, labels, times=None):
    with np.load(path, allow_pickle=False) as saved:
        expected = {"row_id": row_ids, "y_true": labels}
        if times is not None:
            expected["time"] = times
        if any(
            key not in saved or not np.array_equal(saved[key], value)
            for key, value in expected.items()
        ):
            raise ValueError("Prediction row IDs, labels or times changed: " + str(path))
        probability = saved["p_positive"]
    return probability, binary_metrics(labels, probability)


def reload_difference(expected, actual):
    if expected.shape != actual.shape:
        raise ValueError("Reloaded prediction shape changed")
    error = float(np.max(np.abs(expected - actual)))
    if not np.isfinite(error) or error > RELOAD_TOLERANCE:
        raise ValueError("Reloaded probabilities differ")
    return error


def check_recorded_error(record, key):
    error = record.get(key)
    if (
        not isinstance(error, (int, float))
        or not np.isfinite(error)
        or not 0 <= error <= RELOAD_TOLERANCE
    ):
        raise ValueError("Invalid saved reload check: " + key)


def check_logistic_parameters(model, identity):
    if (
        not isinstance(model, Pipeline)
        or len(model.steps) != 2
        or not isinstance(model[0], StandardScaler)
        or not isinstance(model[1], LogisticRegression)
        or model[0].get_params(deep=False) != StandardScaler().get_params(deep=False)
        or model[1].get_params(deep=False)
        != LogisticRegression(
            C=identity["configuration"]["C"], max_iter=2000, random_state=identity["seed"]
        ).get_params(deep=False)
    ):
        raise ValueError("Candidate Logistic parameters differ from the declared fit")


def validate_locked_evidence(output, variants, spaces, protocol_hash, seed, row_ids, labels):
    """Read-only preflight: a locked search cannot restart or recreate its evidence."""
    for variant in variants:
        for name, space in spaces.items():
            folder = output / variant / name
            selected, final = folder / "selection.json", folder / "result.json"
            if not selected.exists() and not final.exists():
                continue
            if not selected.exists():
                raise ValueError("Completed result lost its locked selection: " + str(folder))
            trials = []
            for index, config in enumerate(space):
                trial = folder / f"trial-{index}"
                if any(
                    not (trial / filename).exists()
                    for filename in ["validation.json", "validation.npz", "checkpoint.joblib"]
                ):
                    raise ValueError("Locked candidate evidence is incomplete: " + str(trial))
                record = read(trial / "validation.json")
                require_identity(
                    record,
                    {
                        "variant": variant,
                        "model": name,
                        "trial": index,
                        "configuration": config,
                        "protocol_sha256": protocol_hash,
                        "seed": seed,
                        "test_evaluated": False,
                    },
                    str(trial),
                )
                for filename, field in [
                    ("checkpoint.joblib", "checkpoint_sha256"),
                    ("validation.npz", "validation_sha256"),
                ]:
                    if file_hash(trial / filename) != record.get(field):
                        raise ValueError(
                            "Locked candidate artifact changed: " + str(trial / filename)
                        )
                _, metrics = prediction_archive(trial / "validation.npz", row_ids, labels)
                if record.get("metrics") != metrics:
                    raise ValueError("Locked candidate validation metrics changed: " + str(trial))
                trials.append(record)
            choice = min(trials, key=lambda item: (item["metrics"]["log_loss"], item["trial"]))
            if read(selected) != choice:
                raise ValueError("Locked candidate selection changed: " + str(folder))
            if final.exists() and read(final).get("selection_sha256") != file_hash(selected):
                raise ValueError(
                    "Completed result lost its original selection identity: " + str(folder)
                )


def audit_candidate(trial, identity, features, row_ids, labels, record=None):
    """Check a local trusted checkpoint, saved validation identities and metrics."""
    fresh = record is not None
    record = record if fresh else read(trial / "validation.json")
    require_identity(record, identity, str(trial))
    if record.get("test_evaluated") is not False:
        raise ValueError("A search candidate must not contain test evaluation")
    for filename, field in [
        ("checkpoint.joblib", "checkpoint_sha256"),
        ("validation.npz", "validation_sha256"),
    ]:
        if record.get(field) != file_hash(trial / filename):
            raise ValueError("Candidate artifact checksum changed: " + str(trial / filename))
    probability, metrics = prediction_archive(trial / "validation.npz", row_ids, labels)
    if record["metrics"] != metrics:
        raise ValueError("Candidate validation metrics do not match saved predictions")
    model = joblib.load(trial / "checkpoint.joblib")
    if identity["model"] == "logistic":
        check_logistic_parameters(model, identity)
    iterations = (
        require_logistic_convergence(model[-1])
        if identity["model"] == "logistic"
        else int(model.best_iteration_)
    )
    if iterations != record["iterations"] or model.n_features_in_ != features.shape[1]:
        raise ValueError("Candidate checkpoint training metadata changed")
    error = reload_difference(probability, positive_probability(model, features))
    if fresh:
        record["reload_max_abs_error"] = error
    else:
        check_recorded_error(record, "reload_max_abs_error")
    return record


def audit_final(folder, identity, selection, features, row_ids, labels, times, record=None):
    fresh = record is not None
    record = record if fresh else read(folder / "result.json")
    require_identity(record, identity, str(folder))
    if record.get("test_evaluated") is not True:
        raise ValueError("Final artifact must explicitly identify test evaluation")
    predictions = folder / "predictions.npz"
    if record.get("predictions_sha256") != file_hash(predictions):
        raise ValueError("Final prediction checksum changed")
    probability, metrics = prediction_archive(predictions, row_ids, labels, times)
    if record["metrics"] != metrics or record["predicted_positive_fraction"] != float(
        (probability >= 0.5).mean()
    ):
        raise ValueError("Final metrics do not match saved predictions")
    trial = folder / f"trial-{selection['trial']}"
    if selection["checkpoint_sha256"] != file_hash(trial / "checkpoint.joblib"):
        raise ValueError("Selected checkpoint checksum changed")
    model = joblib.load(trial / "checkpoint.joblib")
    error = reload_difference(probability, positive_probability(model, features))
    check_recorded_error(record, "reload_max_abs_error")
    if fresh:
        record["test_reload_max_abs_error"] = error
    else:
        check_recorded_error(record, "test_reload_max_abs_error")
        if (folder / "selection.json").stat().st_mtime_ns > (
            folder / "result.json"
        ).stat().st_mtime_ns:
            raise ValueError("Selection was modified after the final result")
    return record


def aggregate(store, rows, columns):
    batches = []
    for start in range(0, len(rows), 1024):
        (history, static, valid, _), _ = store.batch(rows[start : start + 1024], columns)
        count = valid.sum(-1, keepdims=True)
        mean = (history * valid[..., None]).sum(-2) / np.maximum(count, 1)
        variance = (history * history * valid[..., None]).sum(-2) / np.maximum(
            count, 1
        ) - mean * mean
        player = np.concatenate(
            [static, mean, np.sqrt(np.maximum(variance, 0)), count / 10], axis=-1
        )
        batches.append(
            np.concatenate(
                [player.mean(2).reshape(len(player), -1), player.std(2).reshape(len(player), -1)],
                axis=-1,
            ).astype(np.float32)
        )
    result = np.concatenate(batches)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite aggregate features")
    return result


def main():
    global RUN_IDENTITY_VALIDATED
    RUN_IDENTITY_VALIDATED = False
    import lightgbm as lgb

    OUT.mkdir(parents=True, exist_ok=True)
    store = IndexedHistoryStore(DATA)
    rows = {
        part: np.load(DATA / f"{part}-rows.npy", allow_pickle=False)
        for part in ["train", "valid", "test"]
    }
    variants = {"outcome_history": [0, 1, 24], "performance_history": list(range(25))}
    spaces = {
        "logistic": [{"C": c} for c in [0.01, 0.1, 1.0]],
        "lightgbm": [
            {"num_leaves": leaves, "reg_lambda": penalty}
            for leaves, penalty in [(7, 1.0), (31, 5.0), (63, 10.0)]
        ],
    }
    protocol = {
        "source_sha256": code_hash(),
        "driver_sha256": file_hash(Path(__file__)),
        "data_manifest_sha256": file_hash(DATA / "manifest.json"),
        "runtime": runtime_identity(),
        "variants": variants,
        "spaces": spaces,
        "seed": 20261114,
        "threads": 2,
        "lightgbm_common": {"n_estimators": 500, "learning_rate": 0.05, "min_child_samples": 100},
        "selection": "Lowest validation raw LogLoss; no test evaluation until each variant/model selection is locked.",
        "scope": "New-data exploratory baseline comparison, conditional on active-player sampling and reconstructed match contexts.",
        "feature_policy": "Only earlier completed performance. No current draft, role, performance, rank snapshot, player/team identity embedding.",
        "counts": {p: len(r) for p, r in rows.items()},
    }
    marker = OUT / "protocol.json"
    if marker.exists():
        if read(marker) != protocol:
            raise ValueError("Frozen LoL baseline protocol changed")
        if (
            source_hash(OUT / "source-snapshot/gamematch") != protocol["source_sha256"]
            or file_hash(OUT / "source-driver.py") != protocol["driver_sha256"]
        ):
            raise ValueError("Frozen LoL baseline source snapshot changed")
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
            OUT / "split-identity.npz", **{p: store.row_id[r] for p, r in rows.items()}
        )
        if source_hash(OUT / "source-snapshot/gamematch") != protocol["source_sha256"]:
            raise ValueError("Package source changed while freezing the experiment")
        write_json(marker, protocol)
    with np.load(OUT / "split-identity.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(rows) or any(
            not np.array_equal(saved[part], store.row_id[index]) for part, index in rows.items()
        ):
            raise ValueError("Saved baseline split identities changed")
    protocol_hash = file_hash(marker)
    validate_locked_evidence(
        OUT,
        variants,
        spaces,
        protocol_hash,
        protocol["seed"],
        store.row_id[rows["valid"]],
        store.target[rows["valid"]],
    )
    RUN_IDENTITY_VALIDATED = True
    completed = []
    for variant, columns in variants.items():
        matrices = {part: aggregate(store, r, columns) for part, r in rows.items()}
        for name, space in spaces.items():
            folder = OUT / variant / name
            folder.mkdir(parents=True, exist_ok=True)
            write_json(
                OUT / "status.json",
                {
                    "status": "running",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "variant": variant,
                    "model": name,
                    "completed": completed,
                },
            )
            trials = []
            for i, config in enumerate(space):
                trial = folder / f"trial-{i}"
                trial.mkdir(exist_ok=True)
                result_path = trial / "validation.json"
                identity = {
                    "variant": variant,
                    "model": name,
                    "trial": i,
                    "configuration": config,
                    "protocol_sha256": protocol_hash,
                    "features": matrices["train"].shape[1],
                    "seed": protocol["seed"],
                }
                if result_path.exists():
                    result = audit_candidate(
                        trial,
                        identity,
                        matrices["valid"],
                        store.row_id[rows["valid"]],
                        store.target[rows["valid"]],
                    )
                    trials.append(result)
                    continue
                print("FIT", variant, name, i, flush=True)
                if name == "logistic":
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=config["C"], max_iter=2000, random_state=20261114),
                    )
                    with threadpool_limits(limits=2):
                        model.fit(matrices["train"], store.target[rows["train"]])
                    iterations = require_logistic_convergence(model[-1])
                else:
                    model = lgb.LGBMClassifier(
                        **config,
                        **protocol["lightgbm_common"],
                        random_state=20261114,
                        n_jobs=2,
                        verbosity=-1,
                        deterministic=True,
                        force_col_wise=True,
                    )
                    model.fit(
                        matrices["train"],
                        store.target[rows["train"]],
                        eval_set=[(matrices["valid"], store.target[rows["valid"]])],
                        eval_metric="binary_logloss",
                        callbacks=[lgb.early_stopping(30, verbose=False)],
                    )
                    iterations = int(model.best_iteration_)
                p = positive_probability(model, matrices["valid"])
                metrics = binary_metrics(store.target[rows["valid"]], p)
                joblib.dump(model, trial / "checkpoint.joblib")
                np.savez_compressed(
                    trial / "validation.npz",
                    row_id=store.row_id[rows["valid"]],
                    y_true=store.target[rows["valid"]],
                    p_positive=p,
                )
                result = {
                    **identity,
                    "test_evaluated": False,
                    "iterations": iterations,
                    "metrics": metrics,
                    "checkpoint_sha256": file_hash(trial / "checkpoint.joblib"),
                    "validation_sha256": file_hash(trial / "validation.npz"),
                }
                result = audit_candidate(
                    trial,
                    identity,
                    matrices["valid"],
                    store.row_id[rows["valid"]],
                    store.target[rows["valid"]],
                    result,
                )
                write_json(result_path, result)
                trials.append(result)
                print(
                    "VALID", variant, name, i, metrics["log_loss"], metrics["accuracy"], flush=True
                )
                del model
            choice = min(trials, key=lambda r: (r["metrics"]["log_loss"], r["trial"]))
            selected = folder / "selection.json"
            if selected.exists() and read(selected) != choice:
                raise ValueError("Locked candidate selection changed")
            if not selected.exists():
                write_json(selected, choice)
            result_path = folder / "result.json"
            final_identity = {
                "variant": variant,
                "model": name,
                "trial": choice["trial"],
                "configuration": choice["configuration"],
                "seed": protocol["seed"],
                "selection_sha256": file_hash(selected),
                "protocol_sha256": protocol_hash,
            }
            if result_path.exists():
                audit_final(
                    folder,
                    final_identity,
                    choice,
                    matrices["test"],
                    store.row_id[rows["test"]],
                    store.target[rows["test"]],
                    store.time[rows["test"]],
                )
            else:
                trial = folder / f"trial-{choice['trial']}"
                model = joblib.load(trial / "checkpoint.joblib")
                with np.load(trial / "validation.npz", allow_pickle=False) as saved:
                    reference = saved["p_positive"]
                error = reload_difference(reference, positive_probability(model, matrices["valid"]))
                p = positive_probability(model, matrices["test"])
                y = store.target[rows["test"]]
                np.savez_compressed(
                    folder / "predictions.npz",
                    row_id=store.row_id[rows["test"]],
                    y_true=y,
                    p_positive=p,
                    time=store.time[rows["test"]],
                )
                result = {
                    **final_identity,
                    "reload_max_abs_error": error,
                    "metrics": binary_metrics(y, p),
                    "predicted_positive_fraction": float((p >= 0.5).mean()),
                    "test_evaluated": True,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "predictions_sha256": file_hash(folder / "predictions.npz"),
                }
                result = audit_final(
                    folder,
                    final_identity,
                    choice,
                    matrices["test"],
                    store.row_id[rows["test"]],
                    y,
                    store.time[rows["test"]],
                    result,
                )
                write_json(result_path, result)
                print(
                    "TEST",
                    variant,
                    name,
                    result["metrics"]["log_loss"],
                    result["metrics"]["accuracy"],
                    flush=True,
                )
            completed.append([variant, name])
        del matrices
    prevalence = float(store.target[rows["train"]].mean())
    prior = binary_metrics(store.target[rows["test"]], np.full(len(rows["test"]), prevalence))
    prior_record = {
        "protocol_sha256": protocol_hash,
        "train_positive_fraction": prevalence,
        "test_metrics": prior,
        "test_evaluated": True,
    }
    prior_path = OUT / "prior.json"
    if prior_path.exists():
        if read(prior_path) != prior_record:
            raise ValueError("Saved prior differs from the declared data and protocol")
    else:
        write_json(prior_path, prior_record)
    write_json(
        OUT / "status.json",
        {
            "status": "completed",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed": completed,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if RUN_IDENTITY_VALIDATED:
            write_json(
                OUT / "status.json",
                {"status": "failed", "error": type(error).__name__ + ": " + str(error)},
            )
        raise
