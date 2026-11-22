# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Audit and compare locked HPO results with the original fixed configurations."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from audit_tuning_integrity import broad_selection, source_identity
from extended_metrics import extended_metrics
from plotting_fonts import configure_plot_font

from gamematch.asymmetric_runner import ordinal_metrics
from gamematch.evaluation.binary import binary_metrics
from gamematch.evaluation.comparison import (
    load_predictions,
    paired_cluster_bootstrap,
    per_example_loss,
)
from gamematch.evaluation.multiclass import multiclass_metrics
from gamematch.skill_runner import regression_metrics
from gamematch.tuning.search import MATRIX, select_trial

ROOT = Path(__file__).resolve().parents[1]
RUNS = [ROOT / "runs/hpo-v1", ROOT / "runs/hpo-v1-baselines"]
OUT = ROOT / "reports/hyperparameter-search"
LABELS = {
    "mlp": "MLP",
    "fm": "FM",
    "deepsets": "Deep Sets",
    "neuralac": "NeuralAC",
    "logistic": "Logistic",
    "lightgbm": "LightGBM",
    "ordinal": "Ordinal MLP",
    "gru": "GRU",
    "mmrnet": "MMR-Net adaptation",
    "transformer": "Transformer",
    "ridge": "Ridge",
    "history_deepsets": "History Deep Sets",
    "static_deepsets": "Static Deep Sets",
    "owo": "OwO adaptation",
    "dawn": "DAWN adaptation",
    "owo_transformer": "OwO / Transformer",
}
GROUP_MAPPING = {
    "future-rating": "future-rating",
    "decision-additive": "synthetic-additive",
    "decision-cooperative": "synthetic-cooperative",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_path(dataset, model, seed):
    if dataset.startswith("synthetic-"):
        return ROOT / f"runs/decisions-v1/{dataset[10:]}-seed-{seed}/{model}"
    if dataset.startswith("dbd-"):
        return ROOT / f"runs/asymmetric-v1/{dataset}/seed-{seed}-{model}"
    if dataset == "overwatch":
        return ROOT / f"runs/overwatch-history-v1/seed-{seed}-{model}"
    if dataset == "future-rating":
        return ROOT / f"runs/skill-v1/seed-{seed}-{model}"
    return ROOT / f"runs/local-full-heroes-v1/{dataset}/seed-{seed}-{model}"


def audit_decision_result(folder, result):
    mechanism = result["dataset"].removeprefix("synthetic-")
    environment = (
        ROOT / f"runs/decisions-v1/{mechanism}-seed-{result['seed']}/decision-environment.npz"
    )
    with (
        np.load(environment, allow_pickle=False) as truth,
        np.load(folder / "decisions.npz", allow_pickle=False) as saved,
    ):
        selected = saved["selected"]
        probability = truth["true_probability"]
        if (
            selected.shape != (len(probability),)
            or not np.issubdtype(selected.dtype, np.integer)
            or np.any((selected < 0) | (selected >= probability.shape[1]))
        ):
            raise ValueError(f"Invalid selected candidate indices: {folder}")
        np.testing.assert_array_equal(saved["lobbies"], truth["lobbies"])
        oracle = np.abs(probability - 0.5).min(1)
        imbalance = np.abs(probability[np.arange(len(probability)), selected] - 0.5)
        for field, expected in {
            "oracle_cost": oracle,
            "true_imbalance": imbalance,
            "regret": imbalance - oracle,
        }.items():
            if saved[field].shape != expected.shape or not np.allclose(
                saved[field], expected, atol=1e-12, rtol=1e-12
            ):
                raise ValueError(f"Independent decision outcome differs: {folder}: {field}")
        expected_metrics = {
            "mean_decision_regret": float((imbalance - oracle).mean()),
            "mean_true_imbalance": float(imbalance.mean()),
            "fraction_true_40_60": float((imbalance <= 0.1).mean()),
        }
        for metric, value in expected_metrics.items():
            if not np.isclose(result["decision_metrics"][metric], value, atol=1e-12, rtol=1e-12):
                raise ValueError(f"Independent decision metric differs: {folder}: {metric}")


def audit_result(folder, selection_hash):
    result = read(folder / "result.json")
    if result["selection_sha256"] != selection_hash:
        raise ValueError(f"Selection hash mismatch: {folder}")
    with np.load(folder / "predictions.npz", allow_pickle=False) as f:
        data = dict(f)
    if "prediction" in data:
        recomputed = regression_metrics(data["y_true"], data["prediction"])
    elif "p_positive" in data:
        recomputed = binary_metrics(data["y_true"], data["p_positive"])
    else:
        fn = ordinal_metrics if result["dataset"].startswith("dbd") else multiclass_metrics
        recomputed = fn(data["y_true"], data["probabilities"])
    for metric, value in result["metrics"].items():
        if isinstance(value, (int, float)) and metric in recomputed:
            if not np.isclose(value, recomputed[metric], atol=1e-10, rtol=1e-10):
                raise ValueError(f"Recomputed metric differs: {folder}: {metric}")
    if "decision_metrics" in result:
        audit_decision_result(folder, result)
    if not 0 <= result["reload_max_abs_error"] <= 1e-5:
        raise ValueError(f"Invalid reload error: {folder}")
    result["metrics"].update(extended_metrics(data))
    return result


def old_new_intervals(dataset, model, finals):
    if dataset.startswith("synthetic-"):
        delta = []
        for folder in finals:
            seed = int(folder.name.split("-")[-1])
            with (
                np.load(folder / "decisions.npz") as a,
                np.load(old_path(dataset, model, seed) / "decisions.npz") as b,
            ):
                delta.append(float((a["regret"] - b["raw_exact_regret"]).mean()))
        rng = np.random.default_rng(20261114)
        draws = np.asarray(delta)[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
        return [
            {
                "unit": "world",
                "delta_first_minus_second": float(np.mean(delta)),
                "percentile_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
                "world_differences": delta,
                "caution": "Only five worlds; this interval does not account for HPO selection uncertainty.",
            }
        ]
    differences = []
    meta = None
    for folder in finals:
        seed = int(folder.name.split("-")[-1])
        a = load_predictions(folder / "predictions.npz")
        b = load_predictions(old_path(dataset, model, seed) / "predictions.npz")
        for k in ["row_id", "y_true", "group_id", "player_id", "match_id", "time", "domain"]:
            if k in a and k in b:
                np.testing.assert_array_equal(a[k], b[k])
        metric = "absolute_error" if dataset == "future-rating" else "log_loss"
        differences.append(per_example_loss(a, metric) - per_example_loss(b, metric))
        meta = a
    units = (
        {"player": meta["player_id"], "match": meta["match_id"]}
        if dataset == "future-rating"
        else {"player": meta["player_id"], "day": meta["time"] // 86400}
        if dataset.startswith("dbd-")
        else {"day": meta["time"] // 86400}
        if dataset == "overwatch"
        else {"roster_group": meta["group_id"]}
    )
    return [
        {
            "unit": unit,
            **paired_cluster_bootstrap(
                np.stack(differences), clusters, replicates=1000, seed=20261114, metric=metric
            ),
        }
        for unit, clusters in units.items()
    ]


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    protocols = [read(p / "protocol.json") for p in RUNS]
    if protocols[0] != protocols[1]:
        raise ValueError("Broad search protocols differ")
    for run, protocol in zip(RUNS, protocols):
        source_identity(run)
    before = read(ROOT / "reports/model-rankings/rankings.json")
    pairs = {}
    missing = []
    trial_count = 0
    audit_count = 0
    for dataset, models in MATRIX.items():
        for model in models:
            candidates = [
                p / dataset / model for p in RUNS if (p / dataset / model / "status.json").exists()
            ]
            if len(candidates) > 1:
                raise ValueError(f"Duplicated model/dataset output: {dataset}/{model}")
            if not candidates or read(candidates[0] / "status.json")["status"] != "completed":
                missing.append([dataset, model])
                continue
            folder = candidates[0]
            broad_selection(folder, protocols[0])
            selection = read(folder / "selection.json")
            records = {}
            for path in sorted((folder / "search").glob("trial-*/seed-*/metrics.json")):
                item = read(path)
                if (
                    item["test_evaluated"] is not False
                    or (path.parent / "predictions.npz").exists()
                ):
                    raise ValueError(f"Search contains test evaluation: {path}")
                trial = int(path.parent.parent.name.split("-")[-1])
                records.setdefault(trial, []).append(item)
                trial_count += 1
            if select_trial(records, selection["selection_seeds"]) != selection["trial"]:
                raise ValueError(f"Selection cannot be replayed: {folder}")
            expected = (
                [0]
                if selection["configuration"]["tabular"] and not dataset.startswith("synthetic-")
                else list(range(5))
            )
            finals = sorted((folder / "final").glob("seed-*/result.json"))
            if [int(p.parent.name.split("-")[-1]) for p in finals] != expected:
                raise ValueError(f"Incomplete or undeclared final seed set: {folder}")
            if not all(
                (folder / "selection.json").stat().st_mtime <= p.stat().st_mtime for p in finals
            ):
                raise ValueError(f"Final results predate locked selection: {folder}")
            results = [audit_result(p.parent, sha(folder / "selection.json")) for p in finals]
            audit_count += len(results)
            metric = (
                "mean_decision_regret"
                if dataset.startswith("synthetic-")
                else "mae"
                if dataset == "future-rating"
                else "log_loss"
            )
            field = "decision_metrics" if dataset.startswith("synthetic-") else "metrics"
            values = [r[field][metric] for r in results]
            pairs[dataset, model] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "values": values,
                "seed_count": len(values),
                "selection": selection,
                "folder": str(folder.relative_to(ROOT)),
                "final_seconds": sum(r["seconds"] for r in results),
                "cap_hits": sum(
                    r["epochs_completed"] >= r["configuration"].get("epochs", 10**9)
                    for r in results
                ),
                "all_metrics": {
                    m: {"mean": float(np.mean([r["metrics"][m] for r in results]))}
                    for m in results[0]["metrics"]
                    if all(isinstance(r["metrics"][m], (float, int)) for r in results)
                },
            }
    status = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "complete": not missing,
        "completed_model_dataset_pairs": len(pairs),
        "expected_pairs": 46,
        "completed_search_fits": trial_count,
        "audited_final_fits": audit_count,
        "missing": missing,
    }

    def write(p, v):
        p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    write(OUT / "status.json", status)
    ready_datasets = {
        dataset
        for dataset, models in MATRIX.items()
        if all((dataset, model) in pairs for model in models)
    }
    groups = []
    for old in [g for g in before["groups"] if g["kind"] == "primary"]:
        dataset = GROUP_MAPPING.get(old["key"], old["key"])
        if dataset not in ready_datasets:
            continue
        rows = []
        for r in old["rows"]:
            tuned = pairs.get((dataset, r["model"]))
            after = tuned["mean"] if tuned else r["mean"]
            if tuned:
                all_metrics = tuned["all_metrics"]
            else:
                all_metrics = dict(r["all_metrics"])
                reference = (
                    ROOT / "runs/overwatch-ratings-v1" / r["model"]
                    if dataset == "overwatch"
                    else old_path(dataset, r["model"], 0)
                )
                with np.load(reference / "predictions.npz", allow_pickle=False) as saved:
                    all_metrics.update(
                        {k: {"mean": v} for k, v in extended_metrics(dict(saved)).items()}
                    )
            intervals = (
                old_new_intervals(
                    dataset,
                    r["model"],
                    [
                        p.parent
                        for p in sorted(
                            (ROOT / tuned["folder"] / "final").glob("seed-*/result.json")
                        )
                    ],
                )
                if tuned
                else []
            )
            rows.append(
                {
                    "model": r["model"],
                    "label": {
                        "owo": "OwO / CUPID adaptation",
                        "dawn": "DAWN / CHAMP adaptation",
                    }.get(r["model"], r["label"]),
                    "before": r["mean"],
                    "after": after,
                    "delta": after - r["mean"],
                    "relative_improvement_percent": 100 * (r["mean"] - after) / r["mean"],
                    "tuned": bool(tuned),
                    "after_sd": tuned["sd"] if tuned else r["seed_sample_sd"],
                    "details": tuned,
                    "intervals": intervals,
                    "all_metrics": all_metrics,
                }
            )
        rows.sort(key=lambda r: (r["after"], r["model"]))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        groups.append(
            {"dataset": dataset, "title": old["title"], "metric": old["metric"], "rows": rows}
        )
    payload = {
        **status,
        "complete_task_count": len(groups),
        "protocol_sha256": sha(RUNS[0] / "protocol.json"),
        "old_rankings_sha256": sha(ROOT / "reports/model-rankings/rankings.json"),
        "groups": groups,
        "interpretation": "Exploratory validation-selected follow-up. Intervals condition on selected configuration; no selection correction, no multiplicity correction, no retraining bootstrap.",
    }
    if missing:
        write(OUT / "partial-results.json", payload)
        render_report(payload, filename="progress.md")
        print(json.dumps(status, ensure_ascii=False))
        return
    write(OUT / "results.json", payload)
    render_report(payload)
    render_chart(groups)
    print(json.dumps({k: v for k, v in status.items() if k != "missing"}, ensure_ascii=False))


def render_report(payload, filename="README.md"):
    lines = [
        "# 超参搜索后的模型性能" if payload["complete"] else "# 超参搜索阶段报告",
        "",
        f"完成 {payload['completed_model_dataset_pairs']}/{payload['expected_pairs']} 组模型 × 数据集；已完成组含 {payload['completed_search_fits']} 次搜索/复核拟合，{payload['audited_final_fits']} 次最终拟合。",
        "",
        "全部测试指标从逐例预测重新计算，选模记录与检查点重载已核验。按各任务主指标均值排序，越低越好。",
        "",
        (
            "[调参前后图](before-after.png) · [矢量图](before-after.svg) · [完整数值、超参与区间](results.json)"
            if payload["complete"]
            else f"**全套搜索尚未完成。** 下表只汇总 {payload['complete_task_count']} 个已完成全部待调模型与规定种子的任务；其余任务等待完成后再排序。[阶段数值](partial-results.json) · [待完成清单](status.json)"
        ),
        "",
        f"导出时间：{payload['generated_utc']}。进行中的部分拟合未计入上述已完成组统计。",
        "",
        "本轮是看到上一轮测试结果后开展的探索性调参，不能当作新的独立确认实验。改变了优化器、正则化、容量和训练预算等多项因素，收益不能单独归因于 dropout。",
        "",
        "## 各任务前三名",
        "",
        "任务 | 第 1 名 | 第 2 名 | 第 3 名",
        "--- | --- | --- | ---",
    ]
    for g in payload["groups"]:
        lines.append(
            " | ".join([g["title"], *[f"{r['label']}：{r['after']:.6f}" for r in g["rows"][:3]]])
        )
    for g in payload["groups"]:
        lines += [
            "",
            f"## {g['title']}",
            "",
            f"主指标：{g['metric']} ↓。未调参的固定参照保持原值。",
            "",
            "名次 | 模型 | 原固定配置 | 调参后 | 变化（后−前） | 重训标准差 | 是否调参",
            "---: | --- | ---: | ---: | ---: | ---: | ---",
        ]
        for r in g["rows"]:
            sd = "—" if r["after_sd"] is None else f"{r['after_sd']:.6f}"
            lines.append(
                f"{r['rank']} | {r['label']} | {r['before']:.6f} | {r['after']:.6f} | {r['delta']:+.6f} | {sd} | {'是' if r['tuned'] else '固定参照'}"
            )
        keys = (
            ["mae", "rmse", "r2", "spearman"]
            if g["dataset"] == "future-rating"
            else [
                "log_loss",
                "expected_count_mae",
                "expected_count_rmse",
                "accuracy",
                "probability_rmse",
            ]
            if g["dataset"].startswith("dbd-")
            else [
                "log_loss",
                "expected_result_score_mae",
                "expected_result_score_rmse",
                "accuracy",
                "probability_rmse",
            ]
            if g["dataset"] == "overwatch"
            else ["log_loss", "probability_mae", "probability_rmse", "accuracy", "auc"]
        )
        lines += [
            "",
            "### 多指标结果",
            "",
            "模型 | " + " | ".join(keys),
            "--- | " + " | ".join(["---:"] * len(keys)),
        ]
        for r in g["rows"]:
            cells = [
                f"{r['all_metrics'][k]['mean']:.6f}" if k in r["all_metrics"] else "—" for k in keys
            ]
            lines.append(" | ".join([r["label"], *cells]))
        lines += [
            "",
            "### 选中的配置",
            "",
            "模型 | 配置号 | 验证集目标 | Dropout | 学习率 | 权重衰减 | 容量／其他设置",
            "--- | ---: | ---: | ---: | ---: | ---: | ---",
        ]
        for r in g["rows"]:
            if not r["tuned"]:
                continue
            s = r["details"]["selection"]
            c = s["configuration"]
            capacity = (
                {
                    k: c[k]
                    for k in ["C", "alpha", "num_leaves", "reg_lambda", "min_child_samples"]
                    if k in c
                }
                if c["tabular"]
                else {
                    k: c[k]
                    for k in (
                        ["width", "layers", "batch_size", "optimizer", "loss"]
                        if g["dataset"] in {"overwatch", "future-rating"}
                        else [
                            "hidden_dim",
                            "embedding_dim",
                            "interaction_hidden",
                            "batch_size",
                            "optimizer",
                        ]
                    )
                }
            )
            lines.append(
                f"{r['label']} | {s['trial']} | {s['validation_mean']:.6f} | {c.get('dropout', '—')} | {c.get('learning_rate', '—')} | {c.get('weight_decay', '—')} | {json.dumps(capacity, ensure_ascii=False)}"
            )
        lines += [
            "",
            "### 调参前后的探索性差值区间",
            "",
            "模型 | 重采样单位 | 后−前均值 | 95% 区间",
            "--- | --- | ---: | ---",
        ]
        for r in g["rows"]:
            for ci in r["intervals"]:
                lo, hi = ci["percentile_interval"]
                lines.append(
                    f"{r['label']} | {ci['unit']} | {ci['delta_first_minus_second']:+.6f} | [{lo:+.6f}, {hi:+.6f}]"
                )
    lines += [
        "",
        "## 实验边界",
        "",
        "- 8 个初始配置，神经模型前 3 个配置用第 2 个种子复核；最终 5 个新拟合。确定性表格模型单次最终拟合。",
        "- 搜索只使用训练和验证数据；在测试评估前保存并锁定 selection.json。分类选模按原始验证 NLL，评级按验证 MAE。",
        "- DBD/Overwatch 最终按验证集温度校准后的概率评估；英雄概率与合成分队使用原始分数。不同任务的指标不能直接合并。",
        "- 合成调参只用历史样本的验证 NLL，不使用隐藏机制的决策遗憾。前两世界用于选模，最终五世界全部报告。",
        "- 调参后仍可能触及 epoch 上限；完整配置和 cap_hits 在 JSON 中，不宣称全局最优。",
        "- 未来评级测试集仍为 942 条、时间不足一天；目标是观测评级代理。OwO/DAWN/MMR-Net 仍是公开数据适配。",
        "- 差值区间是选择后的探索性区间，不包含超参选择不确定性，不做多重比较校正；种子标准差不是置信区间。",
        "- 二分类 probability_MAE/RMSE 比较预测胜率与 0/1 标签，RMSE 等于 sqrt(Brier)。三/五分类 probability_MAE/RMSE 对 one-hot 标签的全部类别分量取均值。",
        "- Overwatch expected_result_score 将负/平/胜映射为 0/0.5/1；DBD expected_count 是 0–4 击杀数的预测期望。它们与分类概率误差不同，不能混名。",
        "- 未来评级是连续回归，没有预测密度时不定义 LogLoss；不把 MAE/RMSE 人为改写成分类指标。",
        "- OwO 是 CUPID 的预测器，DAWN 是 CHAMP 的预测器；这里均为公开数据适配，不是完整工业系统复现。",
        "",
        "协议：[神经模型](../../runs/hpo-v1/protocol.json)、[表格基线](../../runs/hpo-v1-baselines/protocol.json)。方法：[搜索说明](../../docs/hyperparameter-search.md)。",
    ]
    (OUT / filename).write_text("\n".join(lines) + "\n")


def render_chart(groups):
    labels = {
        "title": (
            "GameMatch · 超参搜索前后比较",
            "GameMatch - Before and after hyperparameter search",
        ),
        "subtitle": (
            "按调参后均值排序  |  灰点：原固定配置；青点：降低；橙点：升高；右侧：原值 → 调参后",
            "Sorted by tuned mean | Gray: fixed; teal: lower; orange: higher | Values: fixed -> tuned",
        ),
        "lower": (" · 越低越好", " - lower is better"),
        "seeds": (
            "验证集选配置后独立重训；神经模型 5 个种子。固定参照保持原值。均值名次不代表显著性。",
            "Validation-selected configurations, five neural refits. Fixed references unchanged. Mean ranks do not imply significance.",
        ),
        "scope": (
            "探索性结果：既有测试集已在上一轮使用；训练预算和多个超参同时改变，不能把变化单独归因于 dropout。",
            "Exploratory: the test sets were previously used. Budgets and multiple settings changed; differences cannot be attributed to dropout alone.",
        ),
    }
    english_titles = {
        "uci-dota2": "UCI Dota2 - hero outcomes",
        "neuralac-dota2018": "Dota2018 - hero outcomes",
        "neuralac-lol": "LoL - hero outcomes",
        "overwatch": "Overwatch - loss/draw/win",
        "dbd-2022": "DBD 2022 - five outcomes",
        "dbd-2023": "DBD 2023 - five outcomes",
        "future-rating": "Dota2 - future observed rating",
        "synthetic-additive": "Synthetic additive - assignment",
        "synthetic-cooperative": "Synthetic cooperative - assignment",
    }
    chinese = configure_plot_font(
        [pair[0] for pair in labels.values()]
        + [g["title"] for g in groups]
        + [r["label"] for g in groups for r in g["rows"]]
    )

    def text(key):
        return labels[key][0 if chinese else 1]

    fig, axes = plt.subplots(3, 3, figsize=(23, 17), facecolor="#f7f9fc")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.86, wspace=0.72, hspace=0.58)
    for ax, g in zip(axes.flat, groups):
        rows = g["rows"]
        all_values = np.array([r[k] for r in rows for k in ["before", "after"]])
        low, high = all_values.min(), all_values.max()
        span = max(high - low, abs(low) * 0.006)
        ax.set_xlim(low - 0.1 * span, high + 0.75 * span)
        ax.set_ylim(len(rows) - 0.5, -0.7)
        for i, r in enumerate(rows):
            color = "#007f86" if r["delta"] <= 0 else "#b36b32"
            ax.axhline(i, color="#edf1f6", lw=0.8, zorder=0)
            ax.plot([r["before"], r["after"]], [i, i], color=color, lw=1.6)
            ax.plot(r["before"], i, "o", color="#a6afbc", ms=4.3)
            ax.plot(r["after"], i, "o", color=color if r["tuned"] else "#6a7890", ms=5.5)
            ax.text(
                0.985,
                i,
                f"{r['before']:.5f} {'→' if chinese else '->'} {r['after']:.5f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                color="#29415c",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            )
        ax.set_yticks(
            range(len(rows)),
            [
                f"{r['rank']:02d}  "
                + (
                    r["label"]
                    if chinese or r["label"].isascii()
                    else LABELS.get(r["model"], r["model"].replace("_", " "))
                )
                for r in rows
            ],
            fontsize=9.3,
        )
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9, colors="#718099")
        for edge in ["top", "right", "left"]:
            ax.spines[edge].set_visible(False)
        ax.spines["bottom"].set_color("#c6d0dc")
        ax.set_title(
            g["title"] if chinese else english_titles[g["dataset"]],
            loc="left",
            fontsize=13,
            pad=14,
            color="#20344e",
        )
        ax.set_xlabel(
            {"log_loss": "NLL", "mae": "MAE", "mean_decision_regret": "Decision regret"}[
                g["metric"]
            ]
            + text("lower"),
            fontsize=10,
        )
    fig.text(0.035, 0.965, text("title"), fontsize=26, color="#18334c", va="top")
    fig.text(
        0.035,
        0.923,
        text("subtitle"),
        fontsize=12,
        color="#4b6079",
        va="top",
    )
    fig.text(
        0.035,
        0.044,
        text("seeds"),
        fontsize=11,
        color="#4b6079",
    )
    fig.text(
        0.035,
        0.018,
        text("scope"),
        fontsize=10.5,
        color="#4b6079",
    )
    fig.savefig(OUT / "before-after.png", dpi=165)
    fig.savefig(OUT / "before-after.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
