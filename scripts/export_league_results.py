# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Compare matched LoL information sets from validated saved predictions."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from gamematch.data.heroes import file_hash
from gamematch.data.indexed_history import IndexedHistoryStore
from gamematch.evaluation.binary import binary_metrics
from gamematch.evaluation.comparison import paired_cluster_bootstrap, per_example_loss
from gamematch.history_runner import temperature_fit
from gamematch.models.history import HistoryDeepSets
from gamematch.models.omninet import OwO
from gamematch.models.tabular import require_logistic_convergence
from gamematch.runner import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/richer-history/league-comparison"
VARIANTS = ["outcome_history", "performance_history"]
COLUMNS = {"outcome_history": [0, 1, 24], "performance_history": list(range(25))}
SEEDS = [11, 22, 33]
SPACES = {
    "logistic": [{"C": c} for c in [0.01, 0.1, 1.0]],
    "lightgbm": [{"num_leaves": n, "reg_lambda": r} for n, r in [(7, 1.0), (31, 5.0), (63, 10.0)]],
}
METRIC_KEYS = {
    "n",
    "positive_rate",
    "log_loss",
    "brier",
    "probability_mae",
    "probability_rmse",
    "accuracy",
    "auc",
    "ece_15_equal_width",
}
NAMES = {
    "logistic": "Logistic",
    "lightgbm": "LightGBM",
    "history_deepsets": "History Deep Sets",
    "owo": "OwO (team readout)",
}


def read(path):
    return json.loads(path.read_text())


def metrics(data, mask=None):
    y, p = data["y_true"], data["p_positive"]
    if mask is not None:
        y, p = y[mask], p[mask]
    pred = p >= 0.5
    return {
        **binary_metrics(y, p),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "predicted_positive_fraction": float(pred.mean()),
        "probability_std": float(p.std()),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def aggregate(records):
    return {
        key: {
            "mean": float(np.mean([r[key] for r in records])),
            "sd": float(np.std([r[key] for r in records], ddof=1)) if len(records) > 1 else None,
            "values": [r[key] for r in records],
        }
        for key, value in records[0].items()
        if isinstance(value, (int, float))
    }


def check_metrics(actual, expected):
    if set(expected) != METRIC_KEYS:
        raise ValueError("Incomplete or unexpected saved metric keys")
    for key, value in expected.items():
        matches = (
            actual[key] is None
            if value is None
            else (
                actual[key] is not None
                and np.shape(actual[key]) == np.shape(value)
                and np.allclose(actual[key], value, atol=1e-8, rtol=1e-8)
            )
        )
        if not matches:
            raise ValueError("Cannot reproduce saved metric: " + key)


def require_identity(record, expected, context):
    if any(
        key not in record
        or (record[key] is not value if isinstance(value, bool) else record[key] != value)
        for key, value in expected.items()
    ):
        raise ValueError("Artifact identity changed: " + context)


def check_reload_error(record, key, tolerance):
    error = record.get(key)
    if not isinstance(error, (float, int)) or not np.isfinite(error) or not 0 <= error <= tolerance:
        raise ValueError("Invalid recorded reload error: " + key)


def verify_protocol(base, protocol, neural, manifest_path, split_reference):
    require_identity(protocol, {"variants": COLUMNS}, "information sets")
    if neural:
        require_identity(
            protocol,
            {"models": ["history_deepsets", "owo"], "seeds": SEEDS, "classes": 2},
            "neural budget",
        )
    else:
        require_identity(protocol, {"spaces": SPACES, "seed": 20261114}, "three-trial budget")
    source = base / "source-snapshot/gamematch"
    paths = sorted(source.rglob("*.py"))
    if not paths:
        raise ValueError("Missing frozen source snapshot")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    if (
        digest.hexdigest() != protocol["source_sha256"]
        or file_hash(base / "source-driver.py") != protocol["driver_sha256"]
        or file_hash(manifest_path) != protocol["data_manifest_sha256"]
    ):
        raise ValueError("Frozen source, driver or dataset identity changed")
    with np.load(base / "split-identity.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(split_reference):
            raise ValueError("Split inventory changed")
        for part, row_ids in split_reference.items():
            np.testing.assert_array_equal(saved[part], row_ids, err_msg="Split identity: " + part)
    require_identity(
        protocol, {"counts": {p: len(v) for p, v in split_reference.items()}}, "split counts"
    )


def load_predictions(path, reference, neural=False):
    with np.load(path, allow_pickle=False) as saved:
        data = dict(saved)
    for name, value in reference.items():
        np.testing.assert_array_equal(data[name], value, err_msg="Evaluation identity: " + name)
    if neural and (
        data["raw"].shape != (len(reference["y_true"]), 2) or not np.isfinite(data["raw"]).all()
    ):
        raise ValueError("Expected aligned finite binary logits")
    return data


def audit_tabular_selection(folder, selection, protocol, protocol_sha, reference, variant, model):
    expected_trials = {f"trial-{i}" for i in range(3)}
    if {path.name for path in folder.glob("trial-*") if path.is_dir()} != expected_trials:
        raise ValueError("Expected all three declared validation trials")
    candidates, hashes = [], {}
    for index, config in enumerate(protocol["spaces"][model]):
        trial = folder / f"trial-{index}"
        record = read(trial / "validation.json")
        require_identity(
            record,
            {
                "variant": variant,
                "model": model,
                "trial": index,
                "configuration": config,
                "protocol_sha256": protocol_sha,
                "seed": protocol["seed"],
                "test_evaluated": False,
                "features": 4 * (6 + 2 * len(COLUMNS[variant]) + 1),
            },
            "validation candidate",
        )
        for name, field in [
            ("checkpoint.joblib", "checkpoint_sha256"),
            ("validation.npz", "validation_sha256"),
        ]:
            checksum = file_hash(trial / name)
            if checksum != record[field]:
                raise ValueError("Candidate checkpoint or validation checksum changed")
            hashes[f"trial-{index}/{name}"] = checksum
        validation = load_predictions(trial / "validation.npz", reference)
        check_metrics(
            binary_metrics(validation["y_true"], validation["p_positive"]), record["metrics"]
        )
        check_reload_error(record, "reload_max_abs_error", 1e-8)
        # These are local trained artifacts, checked against the frozen selection chain.
        # Loading metadata is not an independent forward-prediction audit.
        estimator = joblib.load(trial / "checkpoint.joblib")
        if (
            not np.array_equal(estimator.classes_, [0, 1])
            or estimator.n_features_in_ != record["features"]
        ):
            raise ValueError("Candidate checkpoint class order or feature width changed")
        if model == "logistic":
            fitted = estimator[-1]
            iterations = require_logistic_convergence(fitted)
            expected_parameters = {**config, "random_state": protocol["seed"], "max_iter": 2000}
            if not np.isfinite(np.r_[fitted.coef_.ravel(), fitted.intercept_]).all():
                raise ValueError("Nonfinite logistic checkpoint coefficients")
        else:
            fitted, iterations = estimator, int(estimator.best_iteration_)
            expected_parameters = {
                **config,
                **protocol["lightgbm_common"],
                "random_state": protocol["seed"],
            }
        require_identity(fitted.get_params(), expected_parameters, "checkpoint hyperparameters")
        if record["iterations"] != iterations or iterations < 1:
            raise ValueError("Checkpoint training iterations changed")
        hashes[f"trial-{index}/validation.json"] = file_hash(trial / "validation.json")
        candidates.append(record)
    best = min(
        candidates, key=lambda candidate: (candidate["metrics"]["log_loss"], candidate["trial"])
    )
    if selection != best:
        raise ValueError("Selection is not the lowest validation LogLoss among all three trials")
    return hashes


def build_neural_model(name, columns, cfg):
    if name == "history_deepsets":
        return HistoryDeepSets(
            len(columns), 6, width=cfg["width"], classes=2, dropout=cfg["dropout"]
        )
    if name == "owo":
        return OwO(
            len(columns),
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
    raise ValueError("Unexpected neural architecture")


def audit_neural_selection(
    folder, selection, protocol, protocol_sha, reference, variant, model, seed
):
    require_identity(
        selection,
        {
            "model": model,
            "columns": COLUMNS[variant],
            "configuration": protocol["configuration"],
            "seed": seed,
            "protocol_sha256": protocol_sha,
            "test_evaluated": False,
        },
        "neural selection",
    )
    if set(selection["artifact_sha256"]) != {"checkpoint.pt", "validation.npz", "history.json"}:
        raise ValueError("Incomplete neural selection artifact inventory")
    hashes = {}
    for name, checksum in selection["artifact_sha256"].items():
        hashes[name] = file_hash(folder / name)
        if hashes[name] != checksum:
            raise ValueError("Selected neural artifact changed")
    validation = load_predictions(folder / "validation.npz", reference, neural=True)
    score = binary_metrics(
        validation["y_true"], softmax(validation["raw"].astype(float), axis=1)[:, 1]
    )["log_loss"]
    if not np.isclose(score, selection["validation_log_loss"], atol=1e-10, rtol=1e-10):
        raise ValueError("Neural validation LogLoss cannot be reproduced")
    history = read(folder / "history.json")
    epochs, best_epoch = selection["epochs_completed"], selection["best_epoch"]
    if (
        type(epochs) is not int
        or not 1 <= epochs <= protocol["configuration"]["epochs"]
        or len(history) != epochs
        or type(best_epoch) is not int
        or not 1 <= best_epoch <= epochs
        or [record["epoch"] for record in history] != list(range(1, epochs + 1))
    ):
        raise ValueError("Neural epoch history is incomplete")
    scores = np.asarray([record["validation_log_loss"] for record in history])
    if (
        not np.isfinite(scores).all()
        or best_epoch != int(scores.argmin()) + 1
        or not np.isclose(score, scores[best_epoch - 1], atol=1e-10, rtol=1e-10)
    ):
        raise ValueError("Neural checkpoint is not the validation-selected epoch")
    temperature = float(selection["temperature"])
    recovered = temperature_fit(validation["raw"], validation["y_true"])
    if not np.isfinite(temperature) or not np.isclose(temperature, recovered, atol=1e-8, rtol=1e-8):
        raise ValueError("Temperature cannot be recovered from validation predictions")
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint["selection"] != {k: v for k, v in selection.items() if k != "artifact_sha256"}:
        raise ValueError("Checkpoint does not match neural selection")
    if not all(torch.isfinite(value).all() for value in checkpoint["state_dict"].values()):
        raise ValueError("Checkpoint contains nonfinite weights")
    with torch.random.fork_rng(devices=[]):
        estimator = build_neural_model(model, COLUMNS[variant], protocol["configuration"])
    estimator.load_state_dict(checkpoint["state_dict"], strict=True)
    if sum(p.numel() for p in estimator.parameters()) != selection["parameters"]:
        raise ValueError("Neural checkpoint parameter count changed")
    return hashes


def load_fit(
    folder, protocol_sha, reference, neural, variant, model, seed, protocol, validation_reference
):
    result = read(folder / "result.json")
    selected = folder / "selection.json"
    if (
        result["protocol_sha256"] != protocol_sha
        or result["selection_sha256"] != file_hash(selected)
        or result["test_evaluated"] is not True
    ):
        raise ValueError("Result selection/protocol identity mismatch")
    selection = read(selected)
    if selection["protocol_sha256"] != protocol_sha or selection["test_evaluated"] is not False:
        raise ValueError("Invalid validation-only selection")
    if neural:
        require_identity(
            result,
            {
                "cell": f"{variant}/{model}/seed-{seed}",
                "variant": variant,
                "model": model,
                "seed": seed,
                "columns": COLUMNS[variant],
                "configuration": protocol["configuration"],
            },
            "neural result",
        )
        hashes = audit_neural_selection(
            folder, selection, protocol, protocol_sha, validation_reference, variant, model, seed
        )
        check_reload_error(result, "reload_max_abs_error", 1e-6)
        prediction_sha = result["prediction_sha256"]
    else:
        hashes = audit_tabular_selection(
            folder, selection, protocol, protocol_sha, validation_reference, variant, model
        )
        require_identity(
            result,
            {
                "variant": variant,
                "model": model,
                "seed": seed,
                "trial": selection["trial"],
                "configuration": selection["configuration"],
            },
            "tabular result",
        )
        for key in ["reload_max_abs_error", "test_reload_max_abs_error"]:
            check_reload_error(result, key, 1e-8)
        prediction_sha = result["predictions_sha256"]
    if file_hash(folder / "predictions.npz") != prediction_sha:
        raise ValueError("Saved predictions changed")
    data = load_predictions(
        folder / "predictions.npz",
        {name: value for name, value in reference.items() if neural or name != "history_count"},
        neural=neural,
    )
    observed = metrics(data)
    check_metrics(observed, result["metrics"])
    if (
        not neural
        and result["predicted_positive_fraction"] != observed["predicted_positive_fraction"]
    ):
        raise ValueError("Saved prediction diagnostic changed")
    raw = data
    if neural:
        temperature = float(selection["temperature"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid temperature")
        np.testing.assert_allclose(
            data["p_positive"],
            softmax(data["raw"].astype(float) / temperature, axis=1)[:, 1],
            atol=1e-10,
            rtol=1e-10,
        )
        raw = {**data, "p_positive": softmax(data["raw"].astype(float), axis=1)[:, 1]}
        check_metrics(metrics(raw), result["raw_metrics"])
    return (
        data,
        raw,
        {
            **hashes,
            **{
                name: file_hash(folder / name)
                for name in ["result.json", "selection.json", "predictions.npz"]
            },
        },
    )


def cell_folders(base, variant, model, neural, seed):
    folders = (
        [base / variant / model / f"seed-{value}" for value in SEEDS]
        if neural
        else [base / variant / model]
    )
    seeds = SEEDS if neural else [seed]
    completed = sum((folder / "result.json").is_file() for folder in folders)
    return folders, seeds, completed


def main():
    data_directory = ROOT / "datasets/derived/leagueData-v1"
    store = IndexedHistoryStore(data_directory)
    split_rows = {
        part: np.load(data_directory / f"{part}-rows.npy", allow_pickle=False)
        for part in ["train", "valid", "test"]
    }
    rows = split_rows["test"]
    split_reference = {part: store.row_id[index] for part, index in split_rows.items()}
    validation_reference = {
        "row_id": store.row_id[split_rows["valid"]],
        "y_true": store.target[split_rows["valid"]],
    }
    counts = store.history[rows].astype(bool).sum(-1)
    reference = {
        "row_id": store.row_id[rows],
        "y_true": store.target[rows],
        "time": store.time[rows],
        "history_count": counts,
    }
    all_seen = (counts > 0).all(axis=(1, 2))
    cohorts = {
        "all": np.ones(len(rows), dtype=bool),
        "all_players_seen": all_seen,
        "some_players_unseen": ~all_seen,
        "all_players_at_least_five": (counts >= 5).all(axis=(1, 2)),
    }
    entries, pending, arrays, raw_arrays, artifacts, protocol_hashes = [], [], {}, {}, {}, {}
    for neural, run_name, models in [
        (False, "league-baselines-v1", ["logistic", "lightgbm"]),
        (True, "league-neural-v1", ["history_deepsets", "owo"]),
    ]:
        base = ROOT / "runs" / run_name
        if not (base / "protocol.json").exists():
            pending += [
                {"model": model, "variant": variant} for model in models for variant in VARIANTS
            ]
            continue
        protocol = read(base / "protocol.json")
        protocol_hashes[run_name] = file_hash(base / "protocol.json")
        verify_protocol(base, protocol, neural, data_directory / "manifest.json", split_reference)
        for model in models:
            for variant in VARIANTS:
                folders, seeds, completed = cell_folders(
                    base, variant, model, neural, protocol.get("seed")
                )
                if completed != len(seeds):
                    pending.append(
                        {
                            "model": model,
                            "variant": variant,
                            "completed_fits": completed,
                            "expected_fits": len(seeds),
                        }
                    )
                    continue
                saved, raw_saved = [], []
                for folder, seed in zip(folders, seeds):
                    data, raw, hashes = load_fit(
                        folder,
                        protocol_hashes[run_name],
                        reference,
                        neural,
                        variant,
                        model,
                        seed,
                        protocol,
                        validation_reference,
                    )
                    saved.append(data)
                    raw_saved.append(raw)
                    artifacts[str(folder.relative_to(ROOT))] = hashes
                arrays[model, variant], raw_arrays[model, variant] = saved, raw_saved
                entries.append(
                    {
                        "model": model,
                        "variant": variant,
                        "seeds": seeds,
                        "metrics": aggregate([metrics(data) for data in saved]),
                        "raw_metrics": aggregate([metrics(data) for data in raw_saved]),
                        "cohorts": {
                            key: aggregate([metrics(data, mask) for data in saved])
                            for key, mask in cohorts.items()
                            if mask.any()
                        },
                        "per_seed": [metrics(data) for data in saved],
                    }
                )
    contrasts = []
    for model in NAMES:
        if any((model, variant) not in arrays for variant in VARIANTS):
            continue
        for treatment, collection in [("reported", arrays), ("raw", raw_arrays)]:
            differences = np.stack(
                [
                    per_example_loss(a) - per_example_loss(b)
                    for a, b in zip(
                        collection[model, "performance_history"],
                        collection[model, "outcome_history"],
                    )
                ]
            )
            for cohort, mask in cohorts.items():
                if mask.any() and len(np.unique(reference["time"][mask] // 86400)) >= 2:
                    ci = paired_cluster_bootstrap(
                        differences[:, mask],
                        reference["time"][mask] // 86400,
                        replicates=2000,
                        seed=20261114,
                    )
                    contrasts.append(
                        {
                            "model": model,
                            "probabilities": treatment,
                            "cohort": cohort,
                            "contrast": "performance_history minus outcome_history",
                            **ci,
                        }
                    )
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "passed",
        "completed_cells": len(entries),
        "pending_cells": len(pending),
        "audited_fits": len(artifacts),
        "audited_tabular_candidates": 3 * sum(entry["model"] in SPACES for entry in entries),
        "audited_neural_fits": 3 * sum(entry["model"] not in SPACES for entry in entries),
        "protocol_sha256": protocol_hashes,
        "exporter_sha256": file_hash(Path(__file__)),
        "artifact_sha256": artifacts,
        "enforced_checks": [
            "Frozen source/driver and data-manifest identities; split row identities and counts",
            "Every tabular trial: identity, checkpoint metadata/hash, validation rows/hash and metrics",
            "Tabular selection: minimum validation LogLoss over exactly three candidates, trial-index tie break",
            "Neural cells: all three declared seeds required before reporting",
            "Neural selection: validation rows, selected epoch, NLL, temperature recovery and checkpoint identity",
            "Test prediction identities/checksum and complete metric-key recomputation",
        ],
        "independent_forward_inference_audit": False,
        "inference_scope": "Checkpoint metadata/state structure and recorded reload errors are checked. This exporter does not independently rerun model predictions.",
    }
    write_json(OUT / "export-audit.json", audit)
    write_json(
        OUT / "results.json",
        {
            "complete": not pending,
            "entries": entries,
            "pending": pending,
            "contrasts": contrasts,
            "protocol_sha256": protocol_hashes,
            "artifact_sha256": artifacts,
            "exporter_sha256": file_hash(Path(__file__)),
            "export_audit_sha256": file_hash(OUT / "export-audit.json"),
            "cohort_counts": {key: int(mask.sum()) for key, mask in cohorts.items()},
            "scope": "Exploratory same-game paired information comparison. Baselines select one of three configurations on validation; neural models use a fixed configuration and three seeds. Not equal search budgets. Neural probabilities use validation-only temperature; raw scores also reported. Day intervals have few independent time blocks and do not remove cross-day player dependence. No online matchmaking or CHAMP cross-mode claim.",
        },
    )
    lines = [
        "# LoL 历史信息对照",
        "",
        f"已完成 {len(entries)}/8 个模型／信息组合；神经模型须三个种子全部完成。",
        "",
        "模型 | 信息 | LogLoss ↓ | ACC ↑ | AUC ↑ | 概率 RMSE ↓ | 概率 MAE ↓ | Balanced ACC ↑",
        "--- | --- | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for entry in entries:
        m = entry["metrics"]
        lines.append(
            f"{NAMES[entry['model']]} | {entry['variant']} | {m['log_loss']['mean']:.6f} | {m['accuracy']['mean']:.2%} | {m['auc']['mean']:.6f} | {m['probability_rmse']['mean']:.6f} | {m['probability_mae']['mean']:.6f} | {m['balanced_accuracy']['mean']:.2%}"
        )
    lines += [
        "",
        "同一时间划分和 43,291 场测试对局。Outcome history 仅有过往赛果／时长／间隔，Performance history 增加过往个人表现和位置；两者共用六项静态历史汇总。",
        "",
        "Logistic/LightGBM 各以验证集选取三个候选之一，神经模型为固定预算三种子，搜索预算并不相同。JSON 提供原始／校准分数、种子波动、冷启动分层和按日期配对区间。",
        "",
        "当前局表现、角色、英雄、段位快照与玩家 ID 不进入预测。该样本限于单一排位模式，不能验证 CHAMP 的跨模式收益。",
        "",
        "[可重算结果](results.json)",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    completed_models = [
        model for model in NAMES if all((model, variant) in arrays for variant in VARIANTS)
    ]
    if completed_models:
        import os

        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from plotting_fonts import configure_plot_font

        configure_plot_font(chinese_labels=[])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        for axis, metric, title in zip(
            axes, ["log_loss", "accuracy"], ["Test LogLoss (lower is better)", "Test accuracy"]
        ):
            for offset, variant in [(-0.16, VARIANTS[0]), (0.16, VARIANTS[1])]:
                values = [
                    next(e for e in entries if e["model"] == model and e["variant"] == variant)[
                        "metrics"
                    ][metric]
                    for model in completed_models
                ]
                axis.errorbar(
                    np.arange(len(values)) + offset,
                    [v["mean"] for v in values],
                    yerr=[v["sd"] or 0 for v in values],
                    fmt="o",
                    capsize=3,
                    label=variant.replace("_", " "),
                )
            axis.set_xticks(
                np.arange(len(completed_models)),
                [NAMES[model] for model in completed_models],
                rotation=15,
            )
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.suptitle("Same LoL test games; completed model/information cells only", fontsize=12)
        fig.savefig(OUT / "comparison.png", dpi=180)
        fig.savefig(OUT / "comparison.pdf")
        plt.close(fig)
    print(json.dumps({"completed_cells": len(entries), "pending_cells": len(pending)}))


if __name__ == "__main__":
    main()
