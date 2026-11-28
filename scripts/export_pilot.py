# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Export completed pilot measurements without turning them into final results."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/local-pilot"))
    parser.add_argument("--latex", type=Path, default=Path("reports/local-pilot/pilot.tex"))
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["config"]["protocol"] != "pilot":
        raise ValueError("This exporter accepts completed development pilots only")
    results = json.loads((args.run / "results.json").read_text())
    expected = (
        len(manifest["config"]["datasets"])
        * len(manifest["config"]["models"])
        * len(manifest["config"]["seeds"])
    )
    if len(results) != expected:
        raise ValueError("Incomplete experiment matrix")
    args.output.mkdir(parents=True, exist_ok=True)
    for source, target in [
        ("results.json", "results.json"),
        ("manifest.json", "provenance.json"),
        ("resolved_config.json", "config.json"),
        ("report.md", "README.md"),
    ]:
        shutil.copyfile(args.run / source, args.output / target)
    labels = {
        "neuralac-dota2018": "NAC Dota2018",
        "neuralac-lol": "NAC LoL",
        "uci-dota2": "UCI Dota2",
    }
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Dataset & Model & Seed & LogLoss & AUC \\",
        r"\hline",
    ]
    for result in results:
        metrics = result["metrics"]["evaluation"]
        auc = "--" if metrics["auc"] is None else f"{metrics['auc']:.4f}"
        lines.append(
            f"{labels[result['dataset']]} & {result['model']} & {result['seed']} & "
            f"{metrics['log_loss']:.4f} & {auc} " + r"\\"
        )
    config = manifest["config"]
    train_limit = config.get("max_train_rows") or "all eligible"
    caption = (
        f"Development pilot only: training limit {train_limit} records per dataset, "
        f"{len(config['seeds'])} seed(s), at most {config.get('training', {}).get('epochs', 10)} epochs. "
        "Evaluation uses an inner development holdout; publication test partitions remain unused. "
        "These measurements do not establish model superiority."
    )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{" + caption + "}",
        r"\label{tab:pilot}",
        r"\end{table}",
        "",
    ]
    args.latex.parent.mkdir(parents=True, exist_ok=True)
    args.latex.write_text("\n".join(lines))
    print(f"Exported {len(results)} pilot rows to {args.output} and {args.latex}")


if __name__ == "__main__":
    main()
