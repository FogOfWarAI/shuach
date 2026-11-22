# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Train from explicit train/validation arguments; test data is never accepted."""

import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import expit, logsumexp, softmax
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits
from torch import nn

from ..asymmetric_runner import ordinal_metrics
from ..data.heroes import file_hash
from ..evaluation.binary import binary_metrics
from ..evaluation.multiclass import multiclass_metrics
from ..history_runner import temperature_fit
from ..models.heroes import make_model
from ..models.history import make_history_model
from ..models.ordinal import OrdinalMLP
from ..models.regularization import ActivationDropoutSequential
from ..models.skill import make_skill_model
from ..models.tabular import require_logistic_convergence
from ..runner import set_seed, write_json
from ..skill_runner import regression_metrics


def _class_count(task, spec, fallback=None):
    classes = spec.get("classes", 3 if task == "history" else fallback)
    if (
        isinstance(classes, (bool, np.bool_))
        or not isinstance(classes, (int, np.integer))
        or classes < 2
    ):
        raise ValueError("A categorical task requires an integer class count of at least two")
    if task == "history" and classes != 3:
        raise ValueError("History tuning supports three-class loss/draw/win targets only")
    return int(classes)


def _validate_target(task, target, spec, classes=None):
    y = np.asarray(target)
    if y.ndim != 1 or not len(y):
        raise ValueError("Targets must be a nonempty one-dimensional vector")
    if task in {"multiclass", "history"}:
        classes = _class_count(task, spec, classes)
        if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= classes)):
            raise ValueError(
                "Categorical targets must be integer labels within the declared classes"
            )
    elif task == "binary":
        if (
            not (np.issubdtype(y.dtype, np.number) or np.issubdtype(y.dtype, np.bool_))
            or np.iscomplexobj(y)
            or not np.isin(y, [0, 1]).all()
        ):
            raise ValueError("Binary targets must contain only zero or one")
    elif task == "regression":
        if not np.issubdtype(y.dtype, np.number) or np.iscomplexobj(y) or not np.isfinite(y).all():
            raise ValueError("Regression targets must be finite real numbers")
        scale = spec.get("delta_scale")
        if (
            isinstance(scale, (bool, np.bool_))
            or not isinstance(scale, (int, float, np.integer, np.floating))
            or not np.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("Regression delta_scale must be finite and positive")
    else:
        raise ValueError(f"Unknown tuning task: {task}")
    return y


def _validate_partition(task, part, spec):
    target = _validate_target(task, part.target, spec)
    if np.asarray(part.row_id).shape != target.shape:
        raise ValueError("Target and row ID vectors must be aligned")
    if any(x.ndim < 1 or len(x) != len(target) for x in part.inputs):
        raise ValueError("Every input array must align with the target vector")
    if part.tabular is not None and part.tabular.shape[0] != len(target):
        raise ValueError("Tabular inputs must align with the target vector")


def model_factory(task, name, spec, cfg):
    if task == "binary":
        return make_model(
            name,
            spec["vocab_size"],
            cfg["embedding_dim"],
            cfg["hidden_dim"],
            dropout=cfg["dropout"],
            interaction_hidden=cfg["interaction_hidden"],
        )
    if task == "multiclass":
        if name == "ordinal":
            return OrdinalMLP(
                spec["features"], cfg["hidden_dim"], spec["classes"], dropout=cfg["dropout"]
            )
        return ActivationDropoutSequential(
            nn.Linear(spec["features"], cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["hidden_dim"], cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["hidden_dim"], spec["classes"]),
            dropout=cfg["dropout"],
        )
    if task == "history":
        _class_count(task, spec)
        return make_history_model(
            name,
            spec["history_features"],
            spec["static_features"],
            spec["team_size"],
            spec["history_length"],
            {**cfg, "deepsets_dropout": cfg["dropout"]},
        )
    if task == "regression":
        return make_skill_model(name, spec["features"], spec["hero_vocab"], cfg)
    raise ValueError(task)


def primary_loss(task, target, prediction, spec):
    prediction = np.asarray(prediction)
    classes = prediction.shape[1] if prediction.ndim == 2 else None
    target = _validate_target(task, target, spec, classes)
    expected_shape = (
        (len(target), _class_count(task, spec, classes))
        if task in {"multiclass", "history"}
        else target.shape
    )
    if prediction.shape != expected_shape:
        raise ValueError("Predictions must align with targets and the declared output shape")
    if not np.issubdtype(prediction.dtype, np.number) or np.iscomplexobj(prediction):
        raise ValueError("Predictions must contain real numeric values")
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite validation predictions")
    if task == "binary":
        return float((np.logaddexp(0, prediction.astype(float)) - target * prediction).mean())
    if task in {"multiclass", "history"}:
        return float(
            (
                logsumexp(prediction.astype(float), axis=1)
                - prediction[np.arange(len(target)), target.astype(int)]
            ).mean()
        )
    return float(np.abs(prediction - target).mean() * spec["delta_scale"])


def forward(model, inputs, task):
    if task == "history":
        return model(*inputs[:3], domain=inputs[3])
    return model(*inputs)


def predict_raw(model, part, task, cfg):
    if cfg.get("tabular"):
        if task == "regression":
            return np.asarray(model.predict(part.tabular))
        p = np.clip(model.predict_proba(part.tabular), 1e-12, 1)
        return np.log(p[:, 1] / p[:, 0]) if task == "binary" else np.log(p)
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(part.target), cfg["batch_size"]):
            inputs = tuple(
                x[start : start + cfg["batch_size"]].to(cfg["device"]) for x in part.inputs
            )
            values.append(forward(model, inputs, task).cpu().numpy())
    return np.concatenate(values)


def fit(task, name, spec, train, valid, cfg, seed, directory):
    """No test partition or test metric enters this function or its return value."""
    _validate_partition(task, train, spec)
    _validate_partition(task, valid, spec)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(seed, cfg.get("threads", 4))
    started = time.perf_counter()
    history, best_epoch = [], 0
    if cfg.get("tabular"):
        if name == "logistic":
            model = LogisticRegression(C=cfg["C"], max_iter=2000, random_state=seed)
            with threadpool_limits(limits=cfg["threads"]):
                model.fit(train.tabular, train.target)
            best_epoch = require_logistic_convergence(model)
        elif name == "ridge":
            model = Ridge(alpha=cfg["alpha"])
            with threadpool_limits(limits=cfg["threads"]):
                model.fit(train.tabular, train.target)
        elif name == "lightgbm":
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=1000,
                num_leaves=cfg["num_leaves"],
                learning_rate=cfg["learning_rate"],
                reg_lambda=cfg["reg_lambda"],
                min_child_samples=cfg["min_child_samples"],
                random_state=seed,
                n_jobs=cfg["threads"],
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            )
            model.fit(
                train.tabular,
                train.target,
                eval_set=[(valid.tabular, valid.target)],
                eval_metric="binary_logloss" if task == "binary" else "multi_logloss",
                callbacks=[lgb.early_stopping(40, verbose=False)],
            )
            best_epoch = int(model.best_iteration_)
        else:
            raise ValueError(name)
        joblib.dump(model, directory / "checkpoint.joblib")
        params = None
    else:
        model = model_factory(task, name, spec, cfg).to(cfg["device"])
        params = sum(p.numel() for p in model.parameters())
        optimizer_type = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
        optimizer = optimizer_type(
            model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=2, min_lr=1e-6
            )
            if cfg["scheduler"]
            else None
        )
        inputs = tuple(x.to(cfg["device"]) for x in train.inputs)
        target = torch.from_numpy(train.target).to(cfg["device"])
        generator = torch.Generator().manual_seed(seed)
        best, stale, state = float("inf"), 0, None
        for epoch in range(1, cfg["epochs"] + 1):
            tick = time.perf_counter()
            model.train()
            order = torch.randperm(len(target), generator=generator).to(cfg["device"])
            loss_sum = torch.zeros((), device=cfg["device"])
            for start in range(0, len(target), cfg["batch_size"]):
                ix = order[start : start + cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                pred = forward(model, tuple(x[ix] for x in inputs), task)
                if task == "binary":
                    loss = nn.functional.binary_cross_entropy_with_logits(pred, target[ix].float())
                elif task in {"multiclass", "history"}:
                    loss = nn.functional.cross_entropy(pred, target[ix].long())
                elif cfg["loss"] == "mae":
                    loss = nn.functional.l1_loss(pred, target[ix])
                elif cfg["loss"] == "huber":
                    loss = nn.functional.smooth_l1_loss(pred, target[ix])
                else:
                    loss = nn.functional.mse_loss(pred, target[ix])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += loss.detach() * len(ix)
            raw = predict_raw(model, valid, task, cfg)
            score = primary_loss(task, valid.target, raw, spec)
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation objective")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss_sum.cpu()) / len(target),
                    "validation_primary": score,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "seconds": time.perf_counter() - tick,
                }
            )
            write_json(directory / "history.json", history)
            if score < best:
                best, stale, best_epoch = score, 0, epoch
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if scheduler:
                scheduler.step(score)
            if stale >= cfg["patience"]:
                break
        model.load_state_dict(state)
        torch.save(
            {
                "state_dict": state,
                "model": name,
                "task": task,
                "specification": spec,
                "configuration": cfg,
                "seed": seed,
                "best_epoch": best_epoch,
            },
            directory / "checkpoint.pt",
        )
    raw = predict_raw(model, valid, task, cfg)
    score = primary_loss(task, valid.target, raw, spec)
    np.savez_compressed(
        directory / "validation.npz", row_id=valid.row_id, y_true=valid.target, raw=raw
    )
    result = {
        "model": name,
        "seed": seed,
        "configuration": cfg,
        "validation_primary": score,
        "selection_metric": "mae" if task == "regression" else "raw_log_loss",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "parameter_count": params,
        "seconds": time.perf_counter() - started,
        "test_evaluated": False,
        "artifact_sha256": {
            name: file_hash(directory / name)
            for name in (
                ["checkpoint.joblib", "validation.npz"]
                if cfg.get("tabular")
                else ["checkpoint.pt", "validation.npz", "history.json"]
            )
        },
    }
    write_json(directory / "metrics.json", result)
    return model, result


def reload_model(directory, task, name, spec, cfg):
    if cfg.get("tabular"):
        return joblib.load(Path(directory) / "checkpoint.joblib")
    saved = torch.load(Path(directory) / "checkpoint.pt", map_location="cpu", weights_only=True)
    if saved["configuration"] != cfg or saved["model"] != name or saved["specification"] != spec:
        raise ValueError("Checkpoint identity mismatch")
    model = model_factory(task, name, spec, cfg)
    model.load_state_dict(saved["state_dict"])
    return model.to(cfg["device"]).eval()


def evaluate(model, bundle, cfg, directory):
    valid, test = bundle.parts["valid"], bundle.parts["test"]
    raw = predict_raw(model, test, bundle.task, cfg)
    fields = {"row_id": test.row_id, **test.metadata, "raw": raw}
    temperature = None
    if bundle.task == "regression":
        prediction = (
            test.metadata["current_mu"]
            + raw * bundle.specification["delta_scale"]
            + bundle.specification["delta_mean"]
        )
        fields.update(y_true=test.metadata["y_true"], prediction=prediction)
        metrics = regression_metrics(fields["y_true"], prediction)
        raw_metrics = metrics
    elif bundle.task == "binary":
        p = expit(raw)
        fields.update(y_true=test.target, p_positive=p)
        metrics = binary_metrics(test.target, p)
        raw_metrics = metrics
    else:
        validation = predict_raw(model, valid, bundle.task, cfg)
        temperature = temperature_fit(validation, valid.target)
        p = softmax(raw / temperature, axis=1)
        fn = ordinal_metrics if bundle.task == "multiclass" else multiclass_metrics
        metrics = fn(test.target, p)
        raw_metrics = fn(test.target, softmax(raw, axis=1))
        fields.update(y_true=test.target, probabilities=p)
    np.savez_compressed(Path(directory) / "predictions.npz", **fields)
    return {"metrics": metrics, "uncalibrated_metrics": raw_metrics, "temperature": temperature}
