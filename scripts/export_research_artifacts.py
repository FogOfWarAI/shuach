# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Generate manuscript tables and scientific figures from analyzed artifacts."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def tex(text):
    return str(text).replace("_", r"\_").replace("&", r"\&")


def architecture_figure(directory):
    """Show the implemented evidence boundaries, independently of numeric results."""
    fig, ax = plt.subplots(figsize=(9, 4.9))
    ax.set(xlim=(0, 9), ylim=(0, 4.9))
    ax.axis("off")

    def box(x, y, width, height, title, detail, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.06,rounding_size=.08",
                facecolor=color,
                edgecolor="#536172",
                linewidth=0.8,
            )
        )
        ax.text(
            x + width / 2,
            y + height - 0.17,
            title,
            ha="center",
            va="top",
            fontsize=10,
            weight="bold",
            color="#172A40",
        )
        ax.text(
            x + width / 2,
            y + height / 2 - 0.12,
            detail,
            ha="center",
            va="center",
            fontsize=8.7,
            linespacing=1.4,
            color="#283848",
        )

    def arrow(a, b, label=None):
        ax.add_patch(
            FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, linewidth=1, color="#536172")
        )
        if label:
            ax.text(
                (a[0] + b[0]) / 2 + 0.08,
                (a[1] + b[1]) / 2,
                label,
                fontsize=8,
                va="center",
                color="#536172",
            )

    box(
        0.1,
        3.45,
        2.45,
        1.05,
        "Record contracts",
        "Entities and outcomes\nFeature / feedback times",
        "#EDF2F8",
    )
    box(
        3.2,
        3.45,
        2.45,
        1.05,
        "Learning protocol",
        "Group / temporal splits\nTrain-fit transforms\nValidation selection",
        "#EDF2F8",
    )
    box(
        6.35,
        3.45,
        2.45,
        1.05,
        "Prediction evidence",
        "Saved test predictions\nProper scores\nPaired uncertainty",
        "#EDF2F8",
    )
    arrow((2.65, 3.98), (3.08, 3.98))
    arrow((5.75, 3.98), (6.23, 3.98))
    box(
        0.1,
        1.65,
        2.45,
        1.05,
        "Feasible assignments",
        "Sizes, parties and roles\nSide-exchangeability rule",
        "#E9F4F0",
    )
    box(
        3.2,
        1.65,
        2.45,
        1.05,
        "Scorer + search",
        "Frozen predictor\nExplicit objective\nBatched exhaustive search",
        "#E9F4F0",
    )
    box(
        6.35,
        1.65,
        2.45,
        1.05,
        "Decision evidence",
        "Gap to model optimum\nRegret to hidden optimum",
        "#E9F4F0",
    )
    arrow((2.65, 2.18), (3.08, 2.18))
    arrow((5.75, 2.18), (6.23, 2.18))
    arrow((4.42, 3.35), (4.42, 2.82), "checkpoint")
    ax.text(
        7.57,
        0.85,
        "Hidden-world oracle",
        ha="center",
        fontsize=9,
        weight="bold",
        color="#8B4C32",
    )
    arrow((7.57, 1.0), (7.57, 1.53))
    ax.text(
        0.1,
        0.58,
        "Every track retains configuration, source / data hashes, split identities and auditable outputs.",
        fontsize=9,
        color="#283848",
    )
    ax.text(
        0.1,
        0.17,
        "Logged outcomes evaluate observed teams. They do not supply outcomes for newly assigned teams.",
        fontsize=8.5,
        color="#536172",
    )
    for suffix in ["pdf", "png"]:
        fig.savefig(directory / f"architecture.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def optimizer_update_table(tables):
    """Read standalone engineering reports; do not add them to model-job counts."""
    lines = [
        r"\begin{longtable}{@{}lrrrr@{}}",
        r"\toprule",
        r"Device & Batch & Previous ms & Packed ms & Time ratio\\",
        r"\midrule\endhead",
    ]
    found = False
    for device in ["cpu", "mps"]:
        source = Path(f"reports/neuralac-update-performance-{device}.json")
        if not source.exists():
            continue
        report = json.loads(source.read_text())
        if report["status"] != "completed" or report["device"] != device:
            raise ValueError("Invalid optimizer-update performance report")
        found = True
        lines.append("% Source SHA-256: " + hashlib.sha256(source.read_bytes()).hexdigest())
        for row in report["summaries"]:
            previous = row["median_seconds_per_update"]["previous"]
            current = row["median_seconds_per_update"]["optimized"]
            if not np.isfinite([previous, current]).all() or min(previous, current) <= 0:
                raise ValueError("Expected positive finite update timings")
            lines.append(
                f"{device.upper()} & {row['batch_size']} & {previous * 1000:.3f} & {current * 1000:.3f} & {previous / current:.2f}"
                + r"\\"
            )
    if not found:
        lines.append(r"\multicolumn{5}{l}{Optimizer-update measurements pending.}\\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (tables / "neuralac-update-performance.tex").write_text("\n".join(lines) + "\n")


def main():
    source = Path("reports/analysis/results.json")
    data = json.loads(source.read_text())
    tables, figures = Path("manuscript/tables"), Path("manuscript/figures")
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    status_lines = [
        r"\begin{tabular}{@{}llrrl@{}}",
        r"\toprule",
        r"Track & Unit & Done & Declared & Status\\",
        r"\midrule",
    ]
    for name, status in data["status"].items():
        status_lines.append(
            f"{tex(name)} & {tex(status.get('unit', 'model jobs'))} & {status['jobs']} & {status['expected_jobs']} & {tex(status['status'])}"
            + r"\\"
        )
    status_lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "run-status.tex").write_text("\n".join(status_lines) + "\n")
    names = {
        "uci-dota2": "UCI Dota2",
        "neuralac-dota2018": "NAC Dota2018",
        "neuralac-lol": "NAC LoL",
        "overwatch": "Overwatch",
        "dbd-2022": "DBD 2022",
        "dbd-2023": "DBD 2023",
        "dota2-future-rating": "Dota future rating",
    }
    labels = {
        "logistic": "Logistic",
        "lightgbm": "LightGBM",
        "deepsets": "Deep Sets",
        "history_deepsets": "History Deep Sets",
        "static_deepsets": "Static Deep Sets",
        "owo_transformer": "OwO / Transformer",
        "owo": "OwO",
        "dawn": "DAWN",
        "mmrnet": "MMR-Net",
        "mlp": "MLP",
        "fm": "FM",
        "gru": "GRU",
        "neuralac": "NeuralAC",
        "current": "Current rating",
        "mean_delta": "Mean residual",
        "history_frequency": "Historical frequency",
        "prior": "Prior",
        "ordinal": "Ordinal MLP",
        "ridge": "Ridge",
        "transformer": "Transformer",
        "neuralac_no_cooperation": "No cooperation",
        "neuralac_no_competition": "No competition",
        "neuralac_no_attention": "No attention",
        "neuralac_author_bias": "Author attention bias",
    }
    for task, title in [
        ("heroes", "Hero outcomes"),
        ("history", "Overwatch neural history"),
        ("ratings", "Overwatch ratings"),
        ("ratings-delay", "Ratings with 24-hour additional delay"),
        ("asymmetric", "Asymmetric predator outcomes"),
        ("skill", "Future observed rating"),
        ("heroes-ablation", "NeuralAC component sensitivity"),
        ("history-target-only", "Target-mode-only information sensitivity"),
        ("heroes-convergence", "NeuralAC convergence sensitivity"),
        ("history-earlier-origin", "Earlier-origin history sensitivity"),
        ("ratings-earlier-origin", "Earlier-origin rating/prior baseline"),
    ]:
        rows = [s for s in data["summaries"] if s["task"] == task and s["complete_seed_set"]]
        lines = [
            r"\begin{longtable}{@{}llrrr@{}}",
            r"\toprule",
            r"Dataset & Model & Seeds & "
            + ("Mean MAE" if task == "skill" else "Mean NLL")
            + r" & Seed SD\\",
            r"\midrule\endhead",
        ]
        for row in rows:
            key = "mae" if task == "skill" else "log_loss"
            m = row["metrics"][key]
            sd = "--" if m["seed_sample_sd"] is None else f"{m['seed_sample_sd']:.5f}"
            lines.append(
                f"{tex(names.get(row['dataset'], row['dataset']))} & {tex(labels.get(row['model'], row['model']))} & {len(row['seeds'])} & {m['mean']:.5f} & {sd}"
                + r"\\"
            )
        if not rows:
            lines.append(r"\multicolumn{5}{l}{No complete seed set at this export time.}\\")
        lines.extend([r"\bottomrule", r"\end{longtable}"])
        (tables / f"{task}.tex").write_text("\n".join(lines) + "\n")
    for task in [
        "heroes",
        "history",
        "history-vs-prior",
        "asymmetric",
        "skill",
        "heroes-ablation",
        "heroes-convergence",
        "history-target-only",
        "feedback-delay",
        "history-earlier-origin",
        "earlier-origin-vs-prior",
        "history-vs-ratings",
        "earlier-origin-vs-ratings",
    ]:
        rating_reference = task in {"history-vs-ratings", "earlier-origin-vs-ratings"}
        lines = [
            r"\begin{longtable}{@{}p{.18\textwidth}p{.28\textwidth}lrr@{}}",
            r"\toprule",
            r"Dataset & "
            + ("Rating reference" if rating_reference else "First model")
            + r" & Cluster & Difference & 95\% interval\\",
            r"\midrule\endhead",
        ]
        for row in data["paired_comparisons"]:
            if row["task"] != task:
                continue
            a, b = row["percentile_interval"]
            model = row["second"] if rating_reference else row["first"]
            name = labels.get(model, model)
            unit = {
                "calendar_day": "Day",
                "group_id": "Roster",
                "player_id": "Player",
                "match_id": "Match",
            }[row["cluster_unit"]]
            lines.append(
                f"{tex(names[row['dataset']])} & {tex(name)} & {unit} & {row['delta_first_minus_second']:.5f} & [{a:.5f}, {b:.5f}]"
                + r"\\"
            )
        lines.extend([r"\bottomrule", r"\end{longtable}"])
        (tables / f"paired-{task}.tex").write_text("\n".join(lines) + "\n")
    calibration = data.get("hero_calibration")
    lines = [
        r"\begin{longtable}{@{}llrrr@{}}",
        r"\toprule",
        r"Dataset & Model & Raw NLL & Temperature & Affine\\",
        r"\midrule\endhead",
    ]
    if calibration:
        calibrated = {(r["dataset"], r["model"], r["variant"]): r for r in calibration["summaries"]}
        for row in data["summaries"]:
            if row["task"] != "heroes" or not row["complete_seed_set"]:
                continue
            key = (row["dataset"], row["model"])
            values = [row["metrics"]["log_loss"]] + [
                calibrated[(*key, mode)]["metrics"]["log_loss"]
                for mode in ["temperature", "affine"]
            ]
            formatted = [
                f"${value['mean']:.5f}\\pm{value['seed_sample_sd']:.5f}$" for value in values
            ]
            lines.append(
                f"{tex(names[row['dataset']])} & {tex(labels.get(row['model'], row['model']))} & "
                + " & ".join(formatted)
                + r"\\"
            )
    else:
        lines.append(r"\multicolumn{5}{l}{Calibration follow-up not yet complete.}\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    (tables / "hero-calibration.tex").write_text("\n".join(lines) + "\n")
    lines = [
        r"\begin{longtable}{@{}lllrr@{}}",
        r"\toprule",
        r"Dataset & Model & Calibration & NLL difference & 95\% interval\\",
        r"\midrule\endhead",
    ]
    if calibration:
        for row in calibration["paired_comparisons"]:
            lo, hi = row["percentile_interval"]
            lines.append(
                f"{tex(names[row['dataset']])} & {tex(labels.get(row['model'], row['model']))} & {tex(row['variant'])} & {row['delta_first_minus_second']:.5f} & [{lo:.5f}, {hi:.5f}]"
                + r"\\"
            )
    lines += [r"\bottomrule", r"\end{longtable}"]
    (tables / "paired-hero-calibration.tex").write_text("\n".join(lines) + "\n")
    cross_calibration = data.get("calibration_cross_model")
    lines = [
        r"\begin{longtable}{@{}lllrr@{}}",
        r"\toprule",
        r"Dataset & First model & Calibration & NLL difference & 95\% interval\\",
        r"\midrule\endhead",
    ]
    if cross_calibration:
        for row in cross_calibration["comparisons"]:
            lo, hi = row["percentile_interval"]
            lines.append(
                f"{tex(names[row['dataset']])} & {tex(labels.get(row['first'], row['first']))} & {tex(row['variant'])} & {row['delta_first_minus_second']:.5f} & [{lo:.5f}, {hi:.5f}]"
                + r"\\"
            )
    lines += [r"\bottomrule", r"\end{longtable}"]
    (tables / "paired-common-calibration.tex").write_text("\n".join(lines) + "\n")
    lines = [
        r"\begin{longtable}{@{}llrr@{}}",
        r"\toprule",
        r"Hidden mechanism & NeuralAC budget & NLL & Decision regret\\",
        r"\midrule\endhead",
    ]
    for cap, rows in [
        (25, data["decision_summaries"]),
        (200, data.get("decision_convergence_summaries", [])),
    ]:
        for row in rows:
            if (
                row["model"] != "neuralac"
                or row["variant"] != "raw"
                or row["solver"] != "exact"
                or not row.get("complete_seed_set", True)
            ):
                continue
            values = [row["metrics"][metric] for metric in ["log_loss", "mean_decision_regret"]]
            cells = [f"${value['mean']:.5f}\\pm{value['seed_sample_sd']:.5f}$" for value in values]
            lines.append(
                f"{tex(row['mechanism'])} & At most {cap} epochs & " + " & ".join(cells) + r"\\"
            )
    lines += [r"\bottomrule", r"\end{longtable}"]
    (tables / "decision-convergence.tex").write_text("\n".join(lines) + "\n")
    for group in (data.get("cohorts") or {}).get("groups", []):
        future = group["name"] == "future-rating"
        keys = (
            ["limited_observed_history", "other_returning_players"]
            if future
            else ["at_least_one_first_observed_player", "all_players_previously_observed"]
        )
        titles = (
            ["Limited observed history", "Other returning players"]
            if future
            else ["Any first observed", "All previously observed"]
        )
        headers = [
            r"\shortstack{" + title + r"\\($n=" + str(group["cohort_counts"][key]) + r"$)}"
            for key, title in zip(keys, titles)
        ]
        lines = [
            r"\begin{longtable}{@{}p{.32\textwidth}rr@{}}",
            r"\toprule",
            "Model & " + " & ".join(headers) + r"\\",
            r"\midrule\endhead",
        ]
        grouped = {}
        for row in group["rows"]:
            grouped.setdefault(row["model"], {})[row["cohort"]] = row
        for model, values in sorted(grouped.items()):
            cells = []
            for key in keys:
                row = values[key]
                value = f"{row['mean']:.5f}"
                if row["seed_sample_sd"] is not None:
                    value += f"\\pm{row['seed_sample_sd']:.5f}"
                cells.append("$" + value + "$")
            lines.append(tex(labels.get(model, model)) + " & " + " & ".join(cells) + r"\\")
        lines.extend([r"\bottomrule", r"\end{longtable}"])
        (tables / f"cohorts-{group['name']}.tex").write_text("\n".join(lines) + "\n")
    performance = data.get("software_performance", {})
    lines = [
        r"\begin{longtable}{@{}llrrrr@{}}",
        r"\toprule",
        r"Device & Operation & Batch & Author ms & Local ms & Time ratio\\",
        r"\midrule\endhead",
    ]
    for device in ["cpu", "mps"]:
        record = performance.get(f"neuralac-{device}")
        if not record:
            continue
        measurements = {
            (r["batch_size"], r["mode"], r["implementation"]): r for r in record["measurements"]
        }
        for batch in [256, 1024, 2048]:
            for mode in ["inference", "forward_backward"]:
                reference = measurements[(batch, mode, "author_zero_attention_bias")][
                    "median_seconds"
                ]
                current = measurements[(batch, mode, "gamematch_vectorized")]["median_seconds"]
                label = "Inference" if mode == "inference" else "Forward/backward"
                lines.append(
                    f"{device.upper()} & {label} & {batch} & {reference * 1000:.3f} & {current * 1000:.3f} & {reference / current:.2f}"
                    + r"\\"
                )
    if not performance:
        lines.append(r"\multicolumn{6}{l}{Matched implementation measurements pending.}\\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (tables / "neuralac-performance.tex").write_text("\n".join(lines) + "\n")
    lines = [
        r"\begin{longtable}{@{}llrrrr@{}}",
        r"\toprule",
        r"Device & Mode & Batch & Previous ms & Packed ms & Ratio\\",
        r"\midrule\endhead",
    ]
    for device in ["cpu", "mps"]:
        records = performance.get(f"neuralac-{device}", {}).get("measurements", [])
        measurements = {(r["batch_size"], r["mode"], r["implementation"]): r for r in records}
        for batch in [256, 1024, 2048]:
            for mode in ["inference", "forward_backward"]:
                previous = measurements.get((batch, mode, "gamematch_previous_layout"))
                current = measurements.get((batch, mode, "gamematch_vectorized"))
                if previous and current:
                    old, new = previous["median_seconds"], current["median_seconds"]
                    label = "Inference" if mode == "inference" else "Forward/backward"
                    lines.append(
                        f"{device.upper()} & {label} & {batch} & {old * 1000:.3f} & {new * 1000:.3f} & {old / new:.2f}"
                        + r"\\"
                    )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (tables / "neuralac-layout-performance.tex").write_text("\n".join(lines) + "\n")
    optimizer_update_table(tables)
    lines = [
        r"\begin{longtable}{@{}rrlrr@{}}",
        r"\toprule",
        r"Team size & Candidates & Implementation & Seconds & Traced peak MiB\\",
        r"\midrule\endhead",
    ]
    for row in performance.get("search", {}).get("measurements", []):
        label = (
            "Materialized"
            if row["implementation"] == "materialized_reference"
            else "Stream, batch 256"
        )
        lines.append(
            f"{row['team_size']} & {row['candidate_count']} & {label} & {row['median_seconds']:.4f} & {row['median_traced_peak_bytes'] / 2**20:.3f}"
            + r"\\"
        )
    if "search" not in performance:
        lines.append(r"\multicolumn{5}{l}{Matched search measurements pending.}\\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (tables / "search-performance.tex").write_text("\n".join(lines) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    architecture_figure(figures)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), layout="constrained")
    colors = ["#2364AA", "#37A58C", "#D36547"]
    for row_index, mechanism in enumerate(["additive", "cooperative"]):
        selected = {
            r["variant"]: r
            for r in data["decision_summaries"]
            if r["mechanism"] == mechanism and r["model"] == "logistic" and r["solver"] == "exact"
        }
        for column, metric in enumerate(["log_loss", "mean_decision_regret"]):
            ax = axes[row_index, column]
            for i, variant in enumerate(["raw", "temperature2", "offset1"]):
                value = selected[variant]["metrics"][metric]
                ax.bar(i, value["mean"], color=colors[i], alpha=0.88, width=0.65)
                ax.errorbar(
                    i,
                    value["mean"],
                    yerr=value["seed_sample_sd"],
                    color="#222222",
                    capsize=4,
                    linewidth=1,
                )
                ax.scatter(
                    np.full(len(value["values"]), i)
                    + np.linspace(-0.12, 0.12, len(value["values"])),
                    value["values"],
                    s=14,
                    facecolor="white",
                    edgecolor="#333333",
                    linewidth=0.5,
                    zorder=5,
                )
            ax.set_xticks(range(3), ["Raw", "Logit / 2", "Logit + 1"])
            ax.set_ylabel("Prediction NLL" if column == 0 else "Decision regret")
            ax.set_title(mechanism.capitalize() + " hidden worlds")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.18)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
    fig.savefig(figures / "decision-diagnostics.pdf", bbox_inches="tight")
    fig.savefig(figures / "decision-diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    curves = data.get("convergence_curves", [])
    if all(
        sum(row["setting"] == setting for row in curves) == 5
        for setting in ["uci-dota2", "additive", "cooperative"]
    ):
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), layout="constrained")
        for axis, setting in zip(axes, ["uci-dota2", "additive", "cooperative"]):
            selected = sorted(
                [row for row in curves if row["setting"] == setting], key=lambda row: row["seed"]
            )
            for row in selected:
                history = row["history"]
                axis.plot(
                    [p["epoch"] for p in history],
                    [p["valid_log_loss"] for p in history],
                    linewidth=1,
                    alpha=0.75,
                    label=f"Seed {row['seed']}",
                )
                best = history[row["best_epoch"] - 1]
                axis.scatter(best["epoch"], best["valid_log_loss"], s=13)
            axis.axvline(selected[0]["primary_cap"], color="black", linestyle="--", linewidth=0.9)
            axis.set_title(
                {
                    "uci-dota2": "UCI Dota2",
                    "additive": "Additive worlds",
                    "cooperative": "Cooperative worlds",
                }[setting]
            )
            axis.set_xlabel("Epoch")
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=0.15)
        axes[0].set_ylabel("Validation NLL")
        axes[-1].legend(fontsize=7)
        fig.savefig(figures / "convergence-curves.pdf", bbox_inches="tight")
        fig.savefig(figures / "convergence-curves.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    metadata = {
        "analysis_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "analysis_generated_utc": data["generated_utc"],
        "plot_uncertainty": "Dots are five independent world/seed replicates; bars are means and error bars sample SD, not confidence intervals",
        "table_policy": "Only complete declared seed sets exported; NLL except future-rating MAE; deterministic repeated runs yield zero SD",
    }
    (figures / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
