# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Report OwO/CHAMP results across frozen reference and two search stages."""

import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from audit_tuning_integrity import broad_selection, source_identity, specialist_selection
from export_tuning_results import audit_result, read, sha
from extended_metrics import extended_metrics
from plotting_fonts import configure_plot_font
from scipy.special import softmax

from gamematch.evaluation.comparison import paired_cluster_bootstrap, per_example_loss
from gamematch.evaluation.multiclass import multiclass_metrics
from gamematch.tuning.data import load_bundle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/specialist-search"
MODELS = ["static_deepsets", "history_deepsets", "owo_transformer", "owo", "dawn"]
NAMES = {
    "static_deepsets": "Static Deep Sets",
    "history_deepsets": "History Deep Sets",
    "owo_transformer": "Transformer adaptation",
    "owo": "OwO / CUPID adaptation",
    "dawn": "DAWN / CHAMP adaptation",
}
STAGES = {
    "reference": ROOT / "runs/overwatch-history-v1",
    "search": ROOT / "runs/hpo-v1/overwatch",
    "specialist": ROOT / "runs/specialist-hpo-v2",
}


def summary(values):
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "values": values,
    }


def metrics(data, mask=None, raw=False):
    y = data["y_true"]
    p = softmax(data.get("raw", data.get("logits")), axis=1) if raw else data["probabilities"]
    if mask is not None:
        y, p = y[mask], p[mask]
    return {**multiclass_metrics(y, p), **extended_metrics({"y_true": y, "probabilities": p})}


def aggregate(rows):
    return {
        k: summary([r[k] for r in rows])
        for k in rows[0]
        if all(isinstance(r[k], (int, float)) for r in rows)
    }


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    entries = []
    arrays = {}
    pending = []
    bundle = load_bundle(ROOT, "overwatch")
    test = bundle.parts["test"]
    counts = test.inputs[2].numpy().sum(-1)
    all_seen = (counts > 0).all(axis=(1, 2))
    history_cohorts = {
        "all_players_previously_observed": all_seen,
        "some_players_unobserved": ~all_seen,
    }
    row_ids, labels = test.row_id.copy(), test.target.copy()
    del test, bundle
    protocols = {}
    for stage, base in STAGES.items():
        if stage == "reference":
            continue
        root = base.parent if stage == "search" else base
        if (root / "protocol.json").exists():
            protocols[stage] = source_identity(root, specialist=stage == "specialist")
    for stage, base in STAGES.items():
        for name in MODELS:
            if stage != "reference":
                status = base / name / "status.json"
                if not status.exists() or read(status)["status"] != "completed":
                    pending.append([stage, name])
                    continue
                selection = read(base / name / "selection.json")
                audit = specialist_selection if stage == "specialist" else broad_selection
                audit(base / name, protocols[stage])
                folders = [base / name / "final" / f"seed-{seed}" for seed in range(5)]
            else:
                selection = None
                folders = [base / f"seed-{seed}-{name}" for seed in range(5)]
            saved = []
            fitted = []
            for folder in folders:
                if stage != "reference":
                    fitted.append(audit_result(folder, sha(base / name / "selection.json")))
                else:
                    fitted.append(read(folder / "metrics.json"))
                with np.load(folder / "predictions.npz", allow_pickle=False) as f:
                    d = dict(f)
                np.testing.assert_array_equal(d["row_id"], row_ids)
                np.testing.assert_array_equal(d["y_true"], labels)
                if saved:
                    for key in ["row_id", "y_true", "time", "domain"]:
                        np.testing.assert_array_equal(saved[0][key], d[key])
                saved.append(d)
            arrays[stage, name] = saved
            entries.append(
                {
                    "stage": stage,
                    "model": name,
                    "label": NAMES[name],
                    "selection": selection,
                    "training": {
                        "parameter_count": [r.get("parameter_count") for r in fitted],
                        "best_epoch": [r.get("best_epoch") for r in fitted],
                        "epochs_completed": [r.get("epochs_completed") for r in fitted],
                        "seconds": [r.get("seconds", r.get("fit_seconds")) for r in fitted],
                    },
                    "metrics": aggregate([metrics(d) for d in saved]),
                    "raw_metrics": aggregate([metrics(d, raw=True) for d in saved]),
                    "by_domain": {
                        str(domain): aggregate([metrics(d, d["domain"] == domain) for d in saved])
                        for domain in [0, 1]
                    },
                    "by_history_cohort": {
                        cohort: aggregate([metrics(d, mask) for d in saved])
                        for cohort, mask in history_cohorts.items()
                        if mask.any()
                    },
                }
            )
    comparisons = []
    for newer, older in [
        ("search", "reference"),
        ("specialist", "search"),
        ("specialist", "reference"),
    ]:
        for name in MODELS:
            if (newer, name) not in arrays or (older, name) not in arrays:
                continue
            delta = []
            for a, b in zip(arrays[newer, name], arrays[older, name]):
                for key in ["row_id", "y_true", "time", "domain"]:
                    np.testing.assert_array_equal(a[key], b[key])
                delta.append(per_example_loss(a) - per_example_loss(b))
            meta = arrays[newer, name][0]
            units = {
                "all": np.ones(len(meta["y_true"]), dtype=bool),
                "domain_0": meta["domain"] == 0,
                "domain_1": meta["domain"] == 1,
            }
            for cohort, mask in units.items():
                ci = paired_cluster_bootstrap(
                    np.stack(delta)[:, mask],
                    meta["time"][mask] // 86400,
                    replicates=1000,
                    seed=20261114,
                )
                comparisons.append(
                    {"newer": newer, "older": older, "model": name, "cohort": cohort, **ci}
                )
    model_comparisons = []
    for stage in STAGES:
        pairs = [(m, baseline) for m in MODELS[2:] for baseline in MODELS[:2]] + [("dawn", "owo")]
        for first, second in pairs:
            if (stage, first) not in arrays or (stage, second) not in arrays:
                continue
            delta = []
            for a, b in zip(arrays[stage, first], arrays[stage, second]):
                for key in ["row_id", "y_true", "time", "domain"]:
                    np.testing.assert_array_equal(a[key], b[key])
                delta.append(per_example_loss(a) - per_example_loss(b))
            meta = arrays[stage, first][0]
            ci = paired_cluster_bootstrap(
                np.stack(delta), meta["time"] // 86400, replicates=1000, seed=20261114
            )
            model_comparisons.append({"stage": stage, "first": first, "second": second, **ci})
    payload = {
        "complete": not pending,
        "entries": entries,
        "pending": pending,
        "paired_day_intervals": comparisons,
        "within_stage_model_intervals": model_comparisons,
        "scope": "Same 8732 WDL test games, 5 seeds. All results are public-data adaptations and exploratory after prior test access. No selection or multiplicity correction.",
    }
    (OUT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# OwO／CHAMP 专项结果",
        "",
        f"状态：{'全部完成' if not pending else '仍在运行；只列已完成的五种子组合'}。",
        "",
        "OwO 对应 CUPID，DAWN 对应 CHAMP。这里是公开数据适配，非原论文工业结果复现。所有主结果保留平局。",
        "",
        "## 三阶段、多指标比较",
        "",
        "阶段 | 模型 | LogLoss ↓ | 概率 RMSE ↓ | 概率 MAE ↓ | 结果分数 MAE ↓ | 结果分数 RMSE ↓ | Accuracy ↑",
        "--- | --- | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    stage_labels = {
        "reference": "原固定配置",
        "search": "第一轮调参",
        "specialist": "专项调参/结构变体",
    }
    for e in entries:
        keys = [
            "log_loss",
            "probability_rmse",
            "probability_mae",
            "expected_result_score_mae",
            "expected_result_score_rmse",
            "accuracy",
        ]
        lines.append(
            " | ".join(
                [
                    stage_labels[e["stage"]],
                    e["label"],
                    *[f"{e['metrics'][k]['mean']:.6f}" for k in keys],
                ]
            )
        )
    lines += [
        "",
        "## 分出胜负的比赛：条件二分类",
        "",
        "这部分将胜率条件化为 p(win)/(p(loss)+p(win))，并明确排除平局，只作次要分析；不能与主三分类数值直接混比。",
        "",
        "阶段 | 模型 | 比赛数 | LogLoss ↓ | RMSE ↓ | MAE ↓ | Accuracy ↑ | AUC ↑",
        "--- | --- | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for e in entries:
        keys = [
            "decisive_n",
            "decisive_log_loss",
            "decisive_probability_rmse",
            "decisive_probability_mae",
            "decisive_accuracy",
            "decisive_auc",
        ]
        lines.append(
            " | ".join(
                [
                    stage_labels[e["stage"]],
                    e["label"],
                    *[f"{e['metrics'][k]['mean']:.6f}" if k in e["metrics"] else "—" for k in keys],
                ]
            )
        )
    lines += [
        "",
        "## 各模式结果",
        "",
        "阶段 | 模型 | 模式 | 样本数 | LogLoss ↓ | 概率 RMSE ↓ | Accuracy ↑",
        "--- | --- | --- | ---: | ---: | ---: | ---:",
    ]
    for e in entries:
        for domain, m in e["by_domain"].items():
            lines.append(
                " | ".join(
                    [
                        stage_labels[e["stage"]],
                        e["label"],
                        "竞技" if domain == "1" else "非竞技",
                        str(int(m["n"]["mean"])),
                        *[
                            f"{m[k]['mean']:.6f}"
                            for k in ["log_loss", "probability_rmse", "accuracy"]
                        ],
                    ]
                )
            )
    lines += [
        "",
        "## 历史覆盖分层",
        "",
        "沿用先前数据审计的分层：全部玩家至少有一场已完成历史，对比至少一名玩家未观测过历史。前者只有 288 场，仅作描述，不能支持丰富历史或冷启动效果的强结论。",
        "",
        "阶段 | 模型 | 历史覆盖 | 样本数 | LogLoss ↓ | 概率 RMSE ↓",
        "--- | --- | --- | ---: | ---: | ---:",
    ]
    for e in entries:
        for cohort, m in e["by_history_cohort"].items():
            lines.append(
                " | ".join(
                    [
                        stage_labels[e["stage"]],
                        e["label"],
                        "全部玩家已有观测"
                        if cohort == "all_players_previously_observed"
                        else "含未观测玩家",
                        str(int(m["n"]["mean"])),
                        *[f"{m[k]['mean']:.6f}" for k in ["log_loss", "probability_rmse"]],
                    ]
                )
            )
    lines += [
        "",
        "## 选中的专项配置",
        "",
        "只由验证集选定。team pooling 是保留两队对比的新适配变体；不将其冒充原作者模型。",
    ]
    for e in entries:
        if e["stage"] == "specialist":
            c = e["selection"]["configuration"]
            lines += [
                "",
                f"- {e['label']}："
                + json.dumps(
                    {
                        k: c[k]
                        for k in [
                            "width",
                            "heads",
                            "layers",
                            "dropout",
                            "learning_rate",
                            "weight_decay",
                            "optimizer",
                            "batch_size",
                            "pooling",
                            "norm_first",
                        ]
                    },
                    ensure_ascii=False,
                ),
            ]
    lines += [
        "",
        "## 校准前后与训练预算",
        "",
        "选模使用原始验证 NLL；测试集温度只由验证数据拟合。训练耗时为并发运行记录，不能作为独立速度排名。",
        "",
        "阶段 | 模型 | 原始测试 NLL | 温度校准 NLL | 参数量 | 最佳 epoch（五种子）",
        "--- | --- | ---: | ---: | --- | ---",
    ]
    for e in entries:
        lines.append(
            " | ".join(
                [
                    stage_labels[e["stage"]],
                    e["label"],
                    f"{e['raw_metrics']['log_loss']['mean']:.6f}",
                    f"{e['metrics']['log_loss']['mean']:.6f}",
                    str(e["training"]["parameter_count"][0]),
                    str(e["training"]["best_epoch"]),
                ]
            )
        )
    lines += [
        "",
        "## 同一阶段的模型差异",
        "",
        "主指标为三分类 LogLoss。差值 = 前者 − 后者，负数有利于前者；区间按日期簇与配对种子重采样，未作多重比较或搜索校正。",
        "",
        "阶段 | 前者 | 后者 | 平均差值 | 95% 探索性区间",
        "--- | --- | --- | ---: | ---",
    ]
    for c in model_comparisons:
        lo, hi = c["percentile_interval"]
        lines.append(
            " | ".join(
                [
                    stage_labels[c["stage"]],
                    NAMES[c["first"]],
                    NAMES[c["second"]],
                    f"{c['delta_first_minus_second']:+.6f}",
                    f"[{lo:+.6f}, {hi:+.6f}]",
                ]
            )
        )
    lines += [
        "",
        "## 分析限制",
        "",
        "- 三分类概率 RMSE = sqrt(Brier_sum / 3)。结果分数使用负/平/胜 = 0/0.5/1；二者不是同一误差。",
        "- NLL 是选模主指标，其他指标同步报告；概率 MAE 仅作描述。连续评级的 LogLoss 不适用。",
        "- 探索性日期簇配对区间保存在 JSON；不包含搜索选择不确定性或多重比较校正。",
        "- 同等试验数不等于同等参数量或 GPU 时间。完整配置、曲线、选模与检查点在 runs 中保留。",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    if not pending:
        chart(entries)
        latex(entries)
    print(
        json.dumps(
            {"complete": not pending, "completed_entries": len(entries), "pending": pending},
            ensure_ascii=False,
        )
    )


def chart(entries):
    text_pairs = {
        "log_loss": ("LogLoss · 三分类", "LogLoss - three classes"),
        "probability_rmse": ("RMSE · 三分类概率", "RMSE - class probabilities"),
        "expected_result_score_mae": ("MAE · 负/平/胜结果得分", "MAE - loss/draw/win score"),
        "reference": ("原固定配置", "Fixed configuration"),
        "search": ("第一轮调参", "First search"),
        "specialist": ("专项调参/结构变体", "Specialist search / variants"),
        "lower": ("越低越好", "Lower is better"),
        "title": (
            "OwO / CHAMP · 调参与结构变体结果",
            "OwO / CHAMP - Hyperparameter search and structural variants",
        ),
        "subtitle": (
            "同一测试集，五个重训种子；各子图按专项结果的 LogLoss 顺序排列",
            "Same test set, five refit seeds. All panels follow the specialist LogLoss ordering.",
        ),
        "scope": (
            "误差线为种子标准差。专项结果可能选择 team pooling 变体；全部都是公开数据适配与探索性证据。",
            "Bars show seed standard deviations. Specialist winners may use team pooling. All results are exploratory public-data adaptations.",
        ),
    }
    labels = {
        "static_deepsets": "Static Deep Sets",
        "history_deepsets": "History Deep Sets",
        "owo_transformer": "Transformer",
        "owo": "OwO / CUPID",
        "dawn": "DAWN / CHAMP",
    }
    chinese = configure_plot_font(
        [pair[0] for pair in text_pairs.values()] + list(labels.values())
    )

    def text(key):
        return text_pairs[key][0 if chinese else 1]

    lookup = {(r["stage"], r["model"]): r for r in entries}
    order = sorted(MODELS, key=lambda m: lookup["specialist", m]["metrics"]["log_loss"]["mean"])
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor="#f7f9fc")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.77, bottom=0.22, wspace=0.76)
    for ax, key in zip(
        axes,
        ["log_loss", "probability_rmse", "expected_result_score_mae"],
    ):
        for offset, stage, color in [
            (-0.18, "reference", "#a6afbc"),
            (0, "search", "#547aa4"),
            (0.18, "specialist", "#007f86"),
        ]:
            means = [lookup[stage, m]["metrics"][key]["mean"] for m in order]
            sd = [lookup[stage, m]["metrics"][key]["sd"] for m in order]
            ax.errorbar(
                means,
                np.arange(5) + offset,
                xerr=sd,
                fmt="o",
                ms=5,
                color=color,
                capsize=2,
                label=text(stage),
            )
        ax.set_yticks(range(5), [labels[m] for m in order], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel(text("lower"), fontsize=10)
        ax.set_title(text(key), loc="left", fontsize=13, pad=16)
        ax.tick_params(axis="y", length=0)
        for edge in ["top", "right", "left"]:
            ax.spines[edge].set_visible(False)
        ax.spines["bottom"].set_color("#c6d0dc")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.25), ncol=3, frameon=False, fontsize=10)
    fig.text(
        0.035, 0.96, text("title"), fontsize=25, color="#18334c", va="top"
    )
    fig.text(
        0.035,
        0.885,
        text("subtitle"),
        fontsize=12,
        color="#4b6079",
        va="top",
    )
    fig.text(
        0.035,
        0.045,
        text("scope"),
        fontsize=11,
        color="#4b6079",
    )
    fig.savefig(OUT / "specialist-comparison.png", dpi=165)
    fig.savefig(OUT / "specialist-comparison.svg")
    plt.close(fig)


def latex(entries):
    lookup = {(r["stage"], r["model"]): r for r in entries}
    labels = {
        "static_deepsets": "Static Deep Sets",
        "history_deepsets": "History Deep Sets",
        "owo_transformer": "Transformer",
        "owo": "OwO (adapted)",
        "dawn": "DAWN (adapted)",
    }
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{lrrr}\toprule",
        r"Model & Fixed NLL & Search NLL & Specialist NLL\\\midrule",
    ]
    for name in MODELS:
        cells = [
            f"{lookup[s, name]['metrics']['log_loss']['mean']:.5f}"
            for s in ["reference", "search", "specialist"]
        ]
        lines.append(" & ".join([labels[name], *cells]) + r"\\")
    lines += [
        r"\bottomrule\end{tabular}",
        r"\caption{Overwatch three-class NLL across three protocols. Every entry averages five fitted predictors; specialist variants are identified in the archived configurations.}\end{table}",
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{lrrrr}\toprule",
        r"Model & NLL & Prob. RMSE & Score MAE & Accuracy\\\midrule",
    ]
    for name in MODELS:
        cells = [
            f"{lookup['specialist', name]['metrics'][k]['mean']:.5f}"
            for k in ["log_loss", "probability_rmse", "expected_result_score_mae", "accuracy"]
        ]
        lines.append(" & ".join([labels[name], *cells]) + r"\\")
    lines += [
        r"\bottomrule\end{tabular}",
        r"\caption{Specialist-search multi-metric results. Probability RMSE uses the three-class one-hot vector; score MAE uses loss/draw/win values 0, 1/2, 1.}\end{table}",
    ]
    (ROOT / "manuscript/tables/specialist-metrics.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
