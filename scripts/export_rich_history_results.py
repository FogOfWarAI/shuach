# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Audited three-seed feature/readout contrasts; incomplete cells stay pending."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from extended_metrics import extended_metrics
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from gamematch.data.heroes import file_hash
from gamematch.data.history import HistoryData, history_split_masks
from gamematch.evaluation.comparison import paired_cluster_bootstrap, per_example_loss
from gamematch.evaluation.multiclass import multiclass_metrics
from gamematch.history_runner import temperature_fit
from gamematch.runner import write_json
from gamematch.tuning.engine import model_factory, primary_loss

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/rich-history-v1"
OUT = ROOT / "reports/richer-history/neural-comparison"
SEEDS = [11, 22, 33]
CELLS = [
    {
        "features": features,
        "history_width": hw,
        "static_width": sw,
        "model": model,
        "pooling": pooling,
    }
    for model in ["history_deepsets", "owo", "dawn"]
    for features, hw, sw in [("original", 5, 13), ("rich", 8, 46)]
    for pooling in (["global"] if model == "history_deepsets" else ["global", "team"])
]
METRIC_KEYS = {
    "n",
    "log_loss",
    "accuracy",
    "brier_multiclass_sum",
    "probability_mae",
    "probability_rmse",
    "class_counts",
}
DIAGNOSTIC_KEYS = {"predicted_counts", "confusion_matrix", "balanced_accuracy", "macro_f1"}


def read(path):
    return json.loads(path.read_text())


def cell_name(cell):
    return f"{cell['model']}--{cell['features']}--{cell['pooling']}"


def summary(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "values": values}


def metrics(data, mask=None, raw=False):
    y = data["y_true"]
    p = softmax(data["raw"].astype(float), axis=1) if raw else data["probabilities"]
    if mask is not None:
        y, p = y[mask], p[mask]
    predicted = p.argmax(1)
    cm = confusion_matrix(y, predicted, labels=[0, 1, 2])
    recalls = np.divide(cm.diagonal(), cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    return {
        **multiclass_metrics(y, p),
        **extended_metrics({"y_true": y, "probabilities": p}),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(
            f1_score(y, predicted, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "loss_recall": float(recalls[0]),
        "draw_recall": float(recalls[1]),
        "win_recall": float(recalls[2]),
        "predicted_draw_fraction": float((predicted == 1).mean()),
        "predicted_win_fraction": float((predicted == 2).mean()),
        "win_probability_std": float(p[:, 2].std()),
        "confusion_matrix": cm.tolist(),
        "predicted_counts": np.bincount(predicted, minlength=3).tolist(),
    }


def aggregate(records):
    return {
        k: summary([r[k] for r in records])
        for k in records[0]
        if all(isinstance(r[k], (int, float)) for r in records)
    }


def require_metrics(actual, expected, required):
    if set(expected) != required:
        raise ValueError("Saved metric schema is incomplete or unexpected")
    for key, value in expected.items():
        if key not in actual or not np.allclose(actual[key], value, atol=2e-7, rtol=2e-7):
            raise ValueError("Saved metric cannot be reproduced: " + key)


def check_predictions(data, reference):
    for key, value in reference.items():
        np.testing.assert_array_equal(data[key], value, err_msg="Wrong evaluation rows: " + key)
    raw = data["raw"]
    if raw.shape != (len(reference["y_true"]), 3) or not np.isfinite(raw).all():
        raise ValueError("Prediction logits must be finite, aligned three-class rows")


def audit_fit(folder, cell, seed, protocol, protocol_sha, reference, validation_reference):
    result = read(folder / "result.json")
    configuration = {**protocol["configuration"], "pooling": cell["pooling"]}
    if (
        result["protocol_sha256"] != protocol_sha
        or result["configuration"] != configuration
        or result["cell"] != cell
        or result["seed"] != seed
        or result["model"] != cell["model"]
        or result["test_evaluated"] is not True
    ):
        raise ValueError("Completed feature/readout fit identity changed")
    training = read(folder / "metrics.json")
    for key in [
        "configuration",
        "seed",
        "model",
        "best_epoch",
        "validation_primary",
        "selection_metric",
        "epochs_completed",
        "parameter_count",
        "seconds",
    ]:
        if result[key] != training[key]:
            raise ValueError("Test result disagrees with training: " + key)
    if training["test_evaluated"] is not False:
        raise ValueError("Training selection record contains a test evaluation")
    if training["selection_metric"] != "raw_log_loss":
        raise ValueError("Checkpoint selection must use raw validation log loss")
    required = ["checkpoint.pt", "validation.npz", "predictions.npz", "history.json"]
    identities = {name: file_hash(folder / name) for name in required}
    identities.update({name: file_hash(folder / name) for name in ["result.json", "metrics.json"]})
    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
        data = dict(saved)
    check_predictions(data, reference)
    with np.load(folder / "validation.npz", allow_pickle=False) as saved:
        validation = dict(saved)
    check_predictions(validation, validation_reference)
    score = primary_loss("history", validation["y_true"], validation["raw"], {})
    if not np.isclose(score, training["validation_primary"], atol=1e-10, rtol=1e-10):
        raise ValueError("Validation selection score cannot be reproduced")
    history = read(folder / "history.json")
    epochs = training["epochs_completed"]
    best_epoch = training["best_epoch"]
    if (
        type(epochs) is not int
        or not 1 <= epochs <= configuration["epochs"]
        or len(history) != epochs
        or [record["epoch"] for record in history] != list(range(1, epochs + 1))
        or type(best_epoch) is not int
        or not 1 <= best_epoch <= epochs
    ):
        raise ValueError("Training epoch history is incomplete or inconsistent")
    scores = np.asarray([record["validation_primary"] for record in history], dtype=float)
    if (
        not np.isfinite(scores).all()
        or best_epoch != int(scores.argmin()) + 1
        or not np.isclose(score, scores[best_epoch - 1], atol=1e-6, rtol=0)
    ):
        raise ValueError("Checkpoint is not the validation-selected epoch")
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    specification = {
        "history_features": cell["history_width"],
        "static_features": cell["static_width"],
        "team_size": 6,
        "history_length": 10,
        "classes": 3,
    }
    for key, value in {
        "model": cell["model"],
        "task": "history",
        "configuration": configuration,
        "seed": seed,
        "best_epoch": best_epoch,
        "specification": specification,
    }.items():
        if checkpoint[key] != value:
            raise ValueError("Checkpoint identity changed: " + key)
    if not all(torch.isfinite(value).all() for value in checkpoint["state_dict"].values()):
        raise ValueError("Checkpoint contains nonfinite weights")
    with torch.random.fork_rng(devices=[]):
        model = model_factory("history", cell["model"], specification, configuration)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if sum(p.numel() for p in model.parameters()) != training["parameter_count"]:
        raise ValueError("Checkpoint parameter count changed")
    reload_error = result["reload_max_abs_error"]
    if not np.isfinite(reload_error) or not 0 <= reload_error <= 1e-5:
        raise ValueError("Saved checkpoint reload did not reproduce training predictions")
    temperature = float(result["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid validation-fitted temperature")
    recovered = temperature_fit(validation["raw"], validation["y_true"])
    if not np.isclose(temperature, recovered, atol=1e-8, rtol=1e-8):
        raise ValueError("Temperature cannot be recovered from validation predictions")
    if data["probabilities"].shape != data["raw"].shape:
        raise ValueError("Calibrated probability shape changed")
    np.testing.assert_allclose(
        data["probabilities"], softmax(data["raw"] / temperature, axis=1), atol=2e-7, rtol=2e-7
    )
    measured, uncalibrated = metrics(data), metrics(data, raw=True)
    require_metrics(measured, result["metrics"], METRIC_KEYS)
    require_metrics(uncalibrated, result["uncalibrated_metrics"], METRIC_KEYS)
    require_metrics(measured, result["classification_diagnostics"], DIAGNOSTIC_KEYS)
    return data, result, identities


def verify_archive(protocol, base, root):
    if protocol["seeds"] != SEEDS or protocol["cells"] != CELLS or protocol["fits"] != 30:
        raise ValueError("Expected the ten declared cells and three distinct paired seeds")
    digest = hashlib.sha256()
    snapshot = base / "source-snapshot/gamematch"
    sources = sorted(snapshot.rglob("*.py"))
    if not sources:
        raise ValueError("Frozen source snapshot is missing")
    for path in sources:
        digest.update(str(path.relative_to(snapshot)).encode())
        digest.update(path.read_bytes())
    if (
        digest.hexdigest() != protocol["source_sha256"]
        or file_hash(base / "source-driver.py") != protocol["driver_sha256"]
        or file_hash(root / ".cache/rich-history-v1.npz") != protocol["cache_sha256"]
        or file_hash(root / ".cache/rich-history-v1.json") != protocol["cache_manifest_sha256"]
    ):
        raise ValueError("Frozen source or feature cache identity changed")
    manifest = read(root / ".cache/rich-history-v1.json")
    if (
        manifest["sha256"] != protocol["cache_sha256"]
        or manifest["identity"] != protocol["cache_generation_identity"]
    ):
        raise ValueError("Cache manifest disagrees with the frozen protocol")


def cell_completion(folder):
    done = [seed for seed in SEEDS if (folder / f"seed-{seed}/result.json").is_file()]
    status_path = folder / "status.json"
    if done != SEEDS or not status_path.is_file():
        return done, False
    status = read(status_path)
    if status["status"] != "completed":
        return done, False
    if status["completed_seeds"] != SEEDS or status["expected_seeds"] != SEEDS:
        raise ValueError("Completed cell status has inconsistent seed identities")
    return done, True


def loss_contrast(arrays, terms, raw=False):
    differences = []
    for index in range(len(SEEDS)):
        total = None
        for name, weight in terms:
            data = arrays[name][index]
            p = softmax(data["raw"].astype(float), axis=1) if raw else data["probabilities"]
            loss = weight * per_example_loss({"y_true": data["y_true"], "probabilities": p})
            total = loss if total is None else total + loss
        differences.append(total)
    return np.stack(differences)


def main():
    protocol = read(BASE / "protocol.json")
    protocol_sha = file_hash(BASE / "protocol.json")
    verify_archive(protocol, BASE, ROOT)
    cache = HistoryData.load(ROOT / ".cache/rich-history-v1.npz")
    masks, _ = history_split_masks(cache)
    test = masks["test"]
    counts = cache.valid[test].sum(-1)
    reference = {
        "row_id": cache.row_ids[test],
        "y_true": cache.outcome[test],
        "time": cache.time[test],
        "domain": cache.domain[test],
        "history_count": counts,
    }
    validation_reference = {
        "row_id": cache.row_ids[masks["valid"]],
        "y_true": cache.outcome[masks["valid"]],
    }
    prevalence = np.bincount(cache.outcome[masks["train"]], minlength=3).astype(float)
    prevalence /= prevalence.sum()
    del cache
    all_seen = (counts > 0).all(axis=(1, 2))
    cohorts = {
        "all": np.ones(len(counts), dtype=bool),
        "all_players_seen": all_seen,
        "some_players_unseen": ~all_seen,
        "domain_0": reference["domain"] == 0,
        "domain_1": reference["domain"] == 1,
    }
    entries, pending, arrays, artifacts = [], [], {}, {}
    for cell in protocol["cells"]:
        name = cell_name(cell)
        folder = BASE / name
        done, complete = cell_completion(folder)
        if not complete:
            pending.append(
                {"cell": name, "completed_seeds": done, "expected_seeds": protocol["seeds"]}
            )
            continue
        data, fitted = [], []
        for seed in protocol["seeds"]:
            values, fit, hashes = audit_fit(
                folder / f"seed-{seed}",
                cell,
                seed,
                protocol,
                protocol_sha,
                reference,
                validation_reference,
            )
            data.append(values)
            fitted.append(fit)
            artifacts[f"{name}/seed-{seed}"] = hashes
        arrays[name] = data
        entries.append(
            {
                "cell": cell,
                "name": name,
                "seeds": protocol["seeds"],
                "metrics": aggregate([metrics(d) for d in data]),
                "raw_metrics": aggregate([metrics(d, raw=True) for d in data]),
                "cohorts": {
                    key: aggregate([metrics(d, mask) for d in data])
                    for key, mask in cohorts.items()
                    if mask.any()
                },
                "per_seed_diagnostics": [metrics(d) for d in data],
                "training": {
                    k: [r[k] for r in fitted]
                    for k in ["best_epoch", "epochs_completed", "parameter_count", "seconds"]
                },
            }
        )
    comparisons = []

    def contrast(label, terms):
        if any(name not in arrays for name, _ in terms):
            return
        for raw in [False, True]:
            differences = loss_contrast(arrays, terms, raw=raw)
            for cohort, mask in cohorts.items():
                if not mask.any() or len(np.unique(reference["time"][mask] // 86400)) < 2:
                    continue
                ci = paired_cluster_bootstrap(
                    differences[:, mask],
                    reference["time"][mask] // 86400,
                    replicates=2000,
                    seed=20261114,
                )
                comparisons.append(
                    {
                        "label": label,
                        "terms": terms,
                        "cohort": cohort,
                        "probabilities": "raw" if raw else "calibrated",
                        **ci,
                    }
                )

    for model in ["history_deepsets", "owo", "dawn"]:
        for pooling in ["global"] if model == "history_deepsets" else ["global", "team"]:
            contrast(
                f"{model}/{pooling}: rich minus original",
                [(f"{model}--rich--{pooling}", 1), (f"{model}--original--{pooling}", -1)],
            )
        if model == "history_deepsets":
            continue
        for features in ["original", "rich"]:
            contrast(
                f"{model}/{features}: team minus global",
                [(f"{model}--{features}--team", 1), (f"{model}--{features}--global", -1)],
            )
        contrast(
            f"{model}: input-by-readout interaction",
            [
                (f"{model}--rich--team", 1),
                (f"{model}--original--team", -1),
                (f"{model}--rich--global", -1),
                (f"{model}--original--global", 1),
            ],
        )
    prior = multiclass_metrics(reference["y_true"], np.tile(prevalence, (len(counts), 1)))
    payload = {
        "complete": not pending,
        "protocol_sha256": protocol_sha,
        "exporter_sha256": file_hash(Path(__file__)),
        "entries": entries,
        "pending": pending,
        "contrasts": comparisons,
        "artifact_sha256": artifacts,
        "train_prevalence_baseline": {"probabilities": prevalence.tolist(), "metrics": prior},
        "scope": "Exploratory fixed-budget public-data adaptations, after prior test access. Three paired training seeds; complete cells only. Intervals assume independent UTC days, with no multiplicity correction; repeated players can cross days. Not a causal or original-paper superiority claim.",
        "dropout_note": "The tuning engine explicitly passes dropout 0.1 to all three architectures, including deepsets_dropout. Input widths/readouts change parameter counts; no claim of equal parameter budget.",
        "cohort_counts": {key: int(mask.sum()) for key, mask in cohorts.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "results.json", payload)
    lines = [
        "# 历史信息 × 队伍读出对照",
        "",
        f"完成组合：{len(entries)}/10；每组必须完成三个种子才进入本表。",
        "",
        "固定预算、同划分、配对种子 11/22/33。公开数据适配，探索性实验；保留平局。",
        "",
        "模型 | 输入 | 读出 | LogLoss ↓ | 原始 LogLoss ↓ | ACC ↑ | Balanced ACC ↑ | Macro-F1 ↑ | 概率 RMSE ↓ | 概率 MAE ↓",
        "--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for entry in entries:
        c, m, raw = entry["cell"], entry["metrics"], entry["raw_metrics"]
        lines.append(
            f"{c['model']} | {c['features']} | {c['pooling']} | {m['log_loss']['mean']:.6f} ± {m['log_loss']['sd']:.6f} | {raw['log_loss']['mean']:.6f} | {m['accuracy']['mean']:.2%} | {m['balanced_accuracy']['mean']:.2%} | {m['macro_f1']['mean']:.4f} | {m['probability_rmse']['mean']:.6f} | {m['probability_mae']['mean']:.6f}"
        )
    lines += [
        "",
        "± 为三个训练种子的样本标准差。JSON 同时提供混淆矩阵、逐类召回、按历史覆盖／模式分层及配对日期自举区间。",
        "",
        "训练引擎对三种模型均传入 dropout=0.1（包括独立的 deepsets_dropout）。输入与读出改变参数量，不能将本实验视为等参数量比较。",
        "",
        "交互项为 (rich/team − original/team) − (rich/global − original/global) 的逐例损失差；并非模型排名或因果效应。",
        "",
        "[完整可重算结果](results.json)",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "complete_cells": len(entries),
                "pending_cells": len(pending),
                "contrasts": len(comparisons),
            }
        )
    )


if __name__ == "__main__":
    main()
