# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Export manuscript follow-up status and complete-result tables from audits."""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "manuscript/tables"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    roots = [ROOT / "runs/hpo-v1", ROOT / "runs/hpo-v1-baselines"]
    status = read(ROOT / "reports/hyperparameter-search/status.json")
    search_count = sum(len(list(r.glob("*/*/search/trial-*/seed-*/metrics.json"))) for r in roots)
    final_count = sum(len(list(r.glob("*/*/final/seed-*/result.json"))) for r in roots)
    specialist = ROOT / "runs/specialist-hpo-v2"
    specialist_search = len(list(specialist.glob("*/rung-*/trial-*/seed-*/metrics.json")))
    specialist_final = len(list(specialist.glob("*/final/seed-*/result.json")))
    completed = sum(read(p)["status"] == "completed" for p in specialist.glob("*/status.json"))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"At this export ({stamp}), the broad search has completed {status['completed_model_dataset_pairs']} of 46 declared model--dataset groups. "
        f"There are {search_count} saved validation-only search fits and {final_count} completed final test evaluations "
        "(planned totals: 473 and 186). "
        f"The specialist search has completed {completed} of five model groups, with {specialist_search} validation-only fits and {specialist_final} final evaluations "
        "(planned totals: 140 and 25). These are fitted-artifact counts, not independent statistical replicates.",
    ]
    if not status["complete"] or completed < 5:
        lines.append(
            "The expanded experiment is incomplete at this export; partial seed sets are not interpreted as final tuned rankings."
        )
    else:
        lines.append(
            "Both declared searches are complete. All selected configurations are retained, including cases in which test performance deteriorates."
        )
    (TABLES / "tuning-status.tex").write_text("\n\n".join(lines) + "\n")
    if not status["complete"]:
        return
    data = read(ROOT / "reports/hyperparameter-search/results.json")
    names = {
        "uci-dota2": "UCI Dota2",
        "neuralac-dota2018": "Dota2018",
        "neuralac-lol": "LoL",
        "overwatch": "Overwatch",
        "dbd-2022": "DBD 2022",
        "dbd-2023": "DBD 2023",
        "future-rating": "Future rating",
        "synthetic-additive": "Synthetic additive",
        "synthetic-cooperative": "Synthetic cooperative",
    }
    tuned = [r for g in data["groups"] for r in g["rows"] if r["tuned"]]
    improvements = sum(r["delta"] < 0 for r in tuned)
    lines = [
        f"The mean test score improves for {improvements} of the {len(tuned)} searched model--dataset groups and does not improve for the others. "
        "This descriptive count does not test a general tuning benefit. The table ranks each task separately, including unchanged reference baselines; it does not select an architecture for independent deployment.",
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{llrr}\toprule",
        r"Task & Lowest observed mean & Fixed & Follow-up\\\midrule",
    ]
    for group in data["groups"]:
        row = group["rows"][0]
        label = row["label"].replace("_", r"\_")
        lines.append(
            f"{names[group['dataset']]} & {label} & {row['before']:.5f} & {row['after']:.5f}"
            + r"\\"
        )
    lines += [
        r"\bottomrule\end{tabular}",
        r"\caption{Descriptive leaders after the broad search: NLL for classification, MAE for future rating and independent decision regret for synthetic tasks. The fixed column is the same model under its original protocol, not the original task leader. All differences, seeds and selected configurations are retained in the accompanying result artifacts.}\end{table}",
    ]
    (TABLES / "tuning-summary.tex").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "broad_complete": True,
                "specialist_complete": completed == 5,
                "improved_groups": improvements,
            }
        )
    )


if __name__ == "__main__":
    main()
