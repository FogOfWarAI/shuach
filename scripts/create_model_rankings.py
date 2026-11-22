# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Export descriptive within-task model rankings from completed local results.

No fitting, inference, data split changes, or new significance tests are run.
"""

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
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "prior": "Prior",
    "logistic": "Logistic",
    "lightgbm": "LightGBM",
    "fm": "FM",
    "mlp": "MLP",
    "deepsets": "Deep Sets",
    "neuralac": "NeuralAC",
    "history_frequency": "History frequency",
    "ordinal": "Ordinal MLP",
    "history_deepsets": "History Deep Sets",
    "static_deepsets": "Static Deep Sets",
    "owo": "OwO adaptation",
    "dawn": "DAWN adaptation",
    "owo_transformer": "OwO / Transformer",
    "calibrated_prior-pooled": "Calibrated prior",
    "elo-pooled": "Elo / pooled",
    "elo-separate": "Elo / separate",
    "openskill-pooled": "OpenSkill / pooled",
    "openskill-separate": "OpenSkill / separate",
    "trueskill-pooled": "TrueSkill / pooled",
    "trueskill-separate": "TrueSkill / separate",
    "current": "Current rating",
    "mean_delta": "Mean delta",
    "ridge": "Ridge",
    "gru": "GRU",
    "transformer": "Transformer",
    "mmrnet": "MMR-Net adaptation",
    "neuralac_author_bias": "NeuralAC / author bias",
    "neuralac_no_attention": "NeuralAC / no attention",
    "neuralac_no_competition": "NeuralAC / no competition",
    "neuralac_no_cooperation": "NeuralAC / no cooperation",
}
DATASETS = {
    "neuralac-dota2018": "Dota2018",
    "neuralac-lol": "LoL",
    "uci-dota2": "UCI Dota2",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "reports/analysis/results.json"
    data = json.loads(source.read_text())
    out = ROOT / "reports/model-rankings"
    out.mkdir(exist_ok=True)
    groups = []

    def add(key, title, metric, rows, note, kind="primary"):
        if not rows:
            raise ValueError(f"Empty ranking: {key}")
        if any(not row.get("complete_seed_set", True) for row in rows):
            raise ValueError(f"Incomplete seed set: {key}")
        ranked = []
        for row in sorted(rows, key=lambda r: (r["metrics"][metric]["mean"], r["model"])):
            entry = row["metrics"][metric]
            if not np.isfinite(entry["mean"]):
                raise ValueError("Nonfinite ranked mean")
            ranked.append(
                {
                    "rank": len(ranked) + 1,
                    "model": row["model"],
                    "label": row.get("display_label", LABELS.get(row["model"], row["model"])),
                    "mean": entry["mean"],
                    "seed_sample_sd": entry.get("seed_sample_sd"),
                    "seeds": row.get("seeds", row.get("world_seeds", [0, 1, 2, 3, 4])),
                    "task": row.get("task"),
                    "all_metrics": row["metrics"],
                }
            )
        groups.append(
            {
                "key": key,
                "title": title,
                "metric": metric,
                "kind": kind,
                "note": note,
                "rows": ranked,
            }
        )

    def summaries(tasks, dataset=None):
        return [
            r
            for r in data["summaries"]
            if r["task"] in tasks and (dataset is None or r["dataset"] == dataset)
        ]

    for dataset, title in DATASETS.items():
        note = "原始概率，固定主实验配置；按 NLL 均值排序。"
        if dataset == "uci-dota2":
            note += "FM、MLP、Deep Sets 相对 Logistic 的现有差值区间均跨零；没有计算所有模型两两之间的区间。"
        add(dataset, title + " · 英雄阵容", "log_loss", summaries(["heroes"], dataset), note)
    add(
        "overwatch",
        "Overwatch · 评级与历史",
        "log_loss",
        summaries(["history", "ratings"]),
        "同一主测试集上的验证集校准 WDL 概率；评级模型为单次确定性回放。History 与 Static Deep Sets 的差值区间跨零。",
    )
    for year in [2022, 2023]:
        add(
            f"dbd-{year}",
            f"DBD {year} · 五类结果",
            "log_loss",
            summaries(["asymmetric"], f"dbd-{year}"),
            "预测 0–4 击杀分布。"
            + (
                "MLP 与 Logistic 的玩家／日期簇差值区间都跨零。"
                if year == 2023
                else "同一数据与时间划分，按主 NLL 排序。"
            ),
        )
    add(
        "future-rating",
        "Dota2 · 未来观测评级",
        "mae",
        summaries(["skill"]),
        "942 条测试记录覆盖不足一天；GRU、MMR-Net 相对 Current rating 的玩家／比赛簇差值区间均跨零。目标不是实际 MMR。",
    )
    for mechanism, title in [("additive", "加性"), ("cooperative", "协作")]:
        rows = [
            r
            for r in data["decision_summaries"]
            if r["mechanism"] == mechanism and r["variant"] == "raw" and r["solver"] == "exact"
        ]
        add(
            f"decision-{mechanism}",
            f"合成{title}机制 · 分队",
            "mean_decision_regret",
            rows,
            "固定原始分数与 252 候选穷举；神经模型主实验上限 25 epochs。NeuralAC 仍受预算限制；延长训练单列。这不是真实玩家重组效果。",
        )

    for variant, caption in [("temperature", "温度校准"), ("affine", "Affine 校准")]:
        for dataset, title in DATASETS.items():
            rows = [
                r
                for r in data["hero_calibration"]["summaries"]
                if r["dataset"] == dataset and r["variant"] == variant
            ]
            add(
                f"{dataset}-{variant}",
                f"{title} · {caption}",
                "log_loss",
                rows,
                "仅原验证集拟合，测试集未改；测试查看后开展的探索性分析，不是新确认实验。",
                "calibration",
            )
    followups = [
        (
            "overwatch-earlier",
            "Overwatch · 较早时间切分",
            ["history-earlier-origin", "ratings-earlier-origin"],
        ),
        ("overwatch-delay", "Overwatch · 额外延迟 24 小时", ["ratings-delay"]),
        ("overwatch-target-only", "Overwatch · 仅目标模式历史", ["history-target-only"]),
        ("neuralac-components", "UCI · NeuralAC 组件消融", ["heroes-ablation"]),
    ]
    for key, title, tasks in followups:
        rows = summaries(tasks)
        if key == "neuralac-components":
            rows += [r for r in summaries(["heroes"], "uci-dota2") if r["model"] == "neuralac"]
        add(
            key,
            title,
            "log_loss",
            rows,
            "独立的敏感性／探索性条件，不与主实验混为同一排名。",
            "followup",
        )
    budget_rows = []
    for task, cap in [("heroes", 50), ("heroes-convergence", 200)]:
        row = next(r for r in summaries([task], "uci-dota2") if r["model"] == "neuralac")
        budget_rows.append({**row, "display_label": f"NeuralAC / cap {cap}"})
    add(
        "uci-budget",
        "UCI · NeuralAC 训练上限",
        "log_loss",
        budget_rows,
        "50 与 200 epoch 上限，均允许提前停止；不代表都训练到上限。差值区间跨零。",
        "budget",
    )
    for mechanism, title in [("additive", "加性"), ("cooperative", "协作")]:
        rows = []
        for field, cap in [("decision_summaries", 25), ("decision_convergence_summaries", 200)]:
            row = next(
                r
                for r in data[field]
                if r["mechanism"] == mechanism
                and r["model"] == "neuralac"
                and r["variant"] == "raw"
                and r["solver"] == "exact"
            )
            rows.append({**row, "display_label": f"NeuralAC / cap {cap}"})
        add(
            f"{mechanism}-budget",
            f"合成{title} · NeuralAC 训练上限",
            "mean_decision_regret",
            rows,
            "25 与 200 epoch 上限，保留原始结果；预算不同，不用于宣称同预算优越。",
            "budget",
        )

    primary = [g for g in groups if g["kind"] == "primary"]
    assert len(primary) == 9 and sum(len(g["rows"]) for g in primary) == 62
    assert all(
        [r["mean"] for r in g["rows"]] == sorted(r["mean"] for r in g["rows"]) for g in groups
    )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source": str(source.relative_to(ROOT)),
        "analysis_sha256": digest(source),
        "script_sha256": digest(Path(__file__)),
        "ranking_policy": "Ascending primary-metric mean within each dataset/protocol. Ordinal row positions are descriptive, not significance groups. No cross-task aggregate ranking.",
        "uncertainty": data["uncertainty"],
        "sd_policy": "Sample SD across recorded seeds/worlds, not confidence intervals; one-run rating SD is absent; repeated deterministic baselines may have zero SD.",
        "primary_groups": len(primary),
        "primary_rows": sum(len(g["rows"]) for g in primary),
        "groups": groups,
    }
    (out / "rankings.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )

    font_path = Path("/Library/Fonts/Arial Unicode.ttf")
    if not font_path.exists():
        font_path = Path("/System/Library/Fonts/STHeiti Medium.ttc")
    fm.fontManager.addfont(str(font_path))
    family = fm.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": [family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "svg.fonttype": "path",
            "savefig.facecolor": "#f7f9fc",
        }
    )

    def chart(selected, filename, title, subtitle, rows, columns, figsize):
        fig, axes = plt.subplots(rows, columns, figsize=figsize, facecolor="#f7f9fc", squeeze=False)
        fig.subplots_adjust(
            left=0.125,
            right=0.975,
            top=0.77 if rows == 1 else 0.87,
            bottom=0.20 if rows == 1 else 0.10,
            wspace=0.74,
            hspace=0.54,
        )
        for ax, group in zip(axes.flat, selected):
            items = group["rows"]
            means = np.array([r["mean"] for r in items])
            std = np.array([r["seed_sample_sd"] or 0 for r in items])
            low, high = min(means - std), max(means + std)
            span = max(high - low, abs(means.mean()) * 0.006, 1e-4)
            ax.set_facecolor("white")
            ax.set_xlim(low - 0.09 * span, high + 0.58 * span)
            ax.set_ylim(len(items) - 0.5, -0.7)
            for i, row in enumerate(items):
                color = "#007f86" if i == 0 else "#5b7291"
                ax.axhline(i, color="#eaf0f6", linewidth=0.75, zorder=0)
                ax.errorbar(
                    row["mean"],
                    i,
                    xerr=std[i],
                    fmt="o",
                    color=color,
                    markersize=6.4 if i == 0 else 5.2,
                    capsize=2.3,
                    linewidth=1.3,
                )
                ax.text(
                    0.985,
                    i,
                    f"{row['mean']:.5f}",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    color=color,
                    fontsize=10.1,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
                )
            ax.set_yticks(
                range(len(items)), [f"{r['rank']:02d}  {r['label']}" for r in items], fontsize=10
            )
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", colors="#718099", labelsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            for edge in ["top", "right", "left"]:
                ax.spines[edge].set_visible(False)
            ax.spines["bottom"].set_color("#c6d0dc")
            metric = {"log_loss": "NLL", "mae": "MAE", "mean_decision_regret": "Decision regret"}[
                group["metric"]
            ]
            ax.set_xlabel(metric + "  ·  越低越好", fontsize=10, labelpad=7)
            ax.set_title(group["title"], loc="left", fontsize=13.3, color="#20344e", pad=14)
        for ax in list(axes.flat)[len(selected) :]:
            ax.axis("off")
        fig.text(0.04, 0.966, title, fontsize=25, color="#18334c", va="top")
        fig.text(
            0.04,
            0.895 if rows == 1 else 0.928,
            subtitle,
            fontsize=12.2,
            color="#4b6079",
            va="top",
        )
        fig.text(
            0.04,
            0.065 if rows == 1 else 0.045,
            "绿点表示均值最低，不表示显著优于其他模型。误差线是种子标准差，不是置信区间。",
            fontsize=11,
            color="#4b6079",
        )
        fig.text(
            0.04,
            0.019,
            (
                "各子图独立横轴；校准后的名次只适用于这一概率评估协议。"
                if rows == 1
                else "各子图独立横轴；评级模型为单次回放。OwO / DAWN / MMR-Net 为公开数据适配。"
            ),
            fontsize=10.4,
            color="#4b6079",
        )
        fig.savefig(out / (filename + ".png"), dpi=165)
        fig.savefig(out / (filename + ".svg"))
        plt.close(fig)

    chart(
        primary,
        "primary-rankings",
        "GameMatch · 主实验模型排名",
        "按各任务主指标均值排序  |  英雄概率未经额外校准；合成决策使用 raw + 穷举，神经训练上限 25 epochs",
        3,
        3,
        (21, 16),
    )
    affine = [g for g in groups if g["kind"] == "calibration" and g["key"].endswith("-affine")]
    chart(
        affine,
        "affine-calibrated-rankings",
        "英雄阵容 · 统一 Affine 校准后的排名",
        "在各模型原验证集上拟合正斜率与截距；测试集不变。这是测试查看后的探索性诊断。",
        1,
        3,
        (21, 6.5),
    )

    lines = [
        "# GameMatch 模型性能排名",
        "",
        "按各数据集／协议的主要指标均值由低到高排序。名次是描述性顺序，不是统计显著性分组；不同任务不合并总榜。",
        "分类用 NLL，未来观测评级用 MAE，独立合成分队用决策遗憾。均值 ± 种子标准差不是置信区间；确定性模型的重复值可能相同，单次评级回放无种子标准差。",
        "",
        "[主实验排名图](primary-rankings.png) · [校准后排名图](affine-calibrated-rankings.png) · [全部指标 JSON](rankings.json)",
        "",
        "## 主实验概览",
        "",
        "任务 | 均值第 1 | 均值第 2 | 均值第 3 | 指标",
        "--- | --- | --- | --- | ---",
    ]
    for group in primary:
        cells = [f"{r['label']}：{r['mean']:.5f}" for r in group["rows"][:3]]
        lines.append(" | ".join([group["title"], *cells, group["metric"]]))
    for kind, label in [
        ("primary", "主实验完整排序"),
        ("calibration", "验证集校准后排序"),
        ("followup", "时间／信息与组件敏感性"),
        ("budget", "训练上限敏感性"),
    ]:
        lines += ["", f"## {label}", ""]
        for group in [g for g in groups if g["kind"] == kind]:
            lines += [
                f"### {group['title']}",
                "",
                group["note"],
                "",
                f"名次 | 模型 | {group['metric']} ↓ | 种子标准差 | 记录种子数",
                "---: | --- | ---: | ---: | ---:",
            ]
            for row in group["rows"]:
                sd = "—" if row["seed_sample_sd"] is None else f"{row['seed_sample_sd']:.6f}"
                lines.append(
                    f"{row['rank']} | {row['label']} | {row['mean']:.6f} | {sd} | {len(row['seeds'])}"
                )
            lines += [""]
    lines += [
        "## 解读边界",
        "",
        "- Overwatch 的 History 与 Static Deep Sets 差异尚不明确；较早切分中的模型顺序可能改变。",
        "- UCI 的 FM 均值最低，但 FM／MLP／Deep Sets 相对 Logistic 的既有区间均跨零，不能把名次解读为确定的优劣。",
        "- DBD 2023 的 MLP 与 Logistic 差值区间跨零。GRU／MMR-Net 相对当前评级持续性的提升也未得到明确区分。",
        "- 合成主实验中 NeuralAC 仍受训练上限影响。延长上限后的结果另列，不能与较短预算模型拼成公平排名。",
        "- 既有差值区间是探索性的成对种子／评价簇重采样区间，无多重比较校正，也没有在每次重采样中重新训练。",
        "- 当前图表展示均值与种子标准差，没有由标准差反推显著性，也没有新增确认实验。",
        "",
        "来源：[冻结的分析结果](../analysis/results.json)、[配对区间](../analysis/README.md)、[模型卡](../../docs/model-cards.md)。",
        f"分析文件 SHA-256：`{payload['analysis_sha256']}`。",
        "重建命令：`.venv/bin/python scripts/create_model_rankings.py`。",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {"groups": len(groups), "primary_rows": payload["primary_rows"], "output": str(out)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
