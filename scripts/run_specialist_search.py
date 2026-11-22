# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Equal-budget multi-fidelity OwO/DAWN search with named readout variants."""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gamematch.configuration import runtime_identity
from gamematch.data.heroes import file_hash
from gamematch.runner import code_hash, write_json
from gamematch.tuning.data import load_bundle
from gamematch.tuning.engine import evaluate, fit, predict_raw, reload_model
from gamematch.tuning.search import audit_primary_split, save_data_identity, select_trial

ROOT = Path(os.environ.get("GAMEMATCH_WORKSPACE", Path(__file__).resolve().parents[1])).resolve()
OUT = ROOT / "runs/specialist-hpo-v2"
MODELS = ["static_deepsets", "history_deepsets", "owo_transformer", "owo", "dawn"]


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def search_space(name):
    path = ROOT / "runs/hpo-v1/overwatch" / name / "selection.json"
    if not path.exists() or read(path.parent / "status.json")["status"] != "completed":
        raise ValueError(f"Complete the frozen first-round search for {name} first")
    base = {**read(path)["configuration"], "pooling": "global", "norm_first": False, "patience": 10}
    result = [base]
    rng = np.random.default_rng(20261114)
    for i in range(15):
        width = int(rng.choice([16, 32, 64, 96]))
        cfg = {
            **base,
            "width": width,
            "heads": int(rng.choice([2, 4, 8])),
            "layers": int(rng.choice([1, 2, 3])),
            "learning_rate": float(rng.choice([0.0001, 0.0003, 0.001, 0.003])),
            "weight_decay": float(rng.choice([0.0, 0.00001, 0.0001, 0.001, 0.01])),
            "dropout": float(rng.choice([0.0, 0.1, 0.2, 0.35, 0.5])),
            "optimizer": "adamw" if i % 3 else "adam",
            "scheduler": True,
            "norm_first": bool(i % 2),
            "pooling": "team" if i % 2 else "global",
            "batch_size": 128 if width >= 64 else 256,
        }
        result.append(cfg)
    # A matched readout counterpart to the first-round winner guarantees coverage
    # of this specific modeling question, before looking at any new score.
    for index, width, batch, prenorm in [(13, 128, 128, False), (14, 256, 32, True)]:
        result[index] = {
            **base,
            "width": width,
            "heads": 8,
            "layers": 3,
            "batch_size": batch,
            "learning_rate": 0.0003,
            "weight_decay": 0.001,
            "dropout": 0.1,
            "optimizer": "adamw",
            "scheduler": True,
            "norm_first": prenorm,
            "pooling": "team",
        }
    if name in {"owo", "dawn", "owo_transformer"}:
        result[-1] = {**base, "pooling": "team"}
    return result


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    spaces = {m: search_space(m) for m in MODELS}
    protocol = {
        "spaces": spaces,
        "rungs": [
            {"epochs": 4, "trials": 16, "seeds": [200]},
            {"epochs": 12, "trials": 6, "seeds": [200]},
            {"epochs": 48, "trials": 3, "seeds": [200, 201]},
        ],
        "promotion": "Validation NLL; always retain first-round selected configuration (trial 0), filling remaining slots by validation rank.",
        "final_seeds": [0, 1, 2, 3, 4],
        "final_epoch_cap": 48,
        "stopping": "10 non-improving validation epochs; Adam/AdamW, gradient clipping, optional validation plateau LR scheduling.",
        "selection": "Raw validation NLL, mean of seeds 200/201 in final rung. No test or oracle score used for promotion or selection.",
        "readout_variant": "Global mean or ordered per-team/per-view means; team variant is a new public-data adaptation, not an original-author reproduction.",
        "norm_variant": "Pre/post Transformer normalization is a search dimension; OmniNet residual implementation remains explicitly adapted.",
        "fairness": "Same trial and rung counts for OwO, DAWN, Transformer and both Deep Sets baselines; parameter counts and optimizer steps differ.",
        "source_sha256": code_hash(),
        "driver_sha256": file_hash(Path(__file__)),
        "runtime": runtime_identity(),
        "prior_selection_sha256": {
            m: file_hash(ROOT / "runs/hpo-v1/overwatch" / m / "selection.json") for m in MODELS
        },
        "evidence": "Exploratory follow-up; prior test results observed. Low-fidelity pruning can discard slow-learning configurations; no global-optimum claim.",
    }
    marker = OUT / "protocol.json"
    if marker.exists():
        if read(marker) != protocol:
            raise ValueError("Specialist search protocol changed")
    else:
        write_json(marker, protocol)
        shutil.copytree(
            ROOT / "src/gamematch",
            OUT / "source-snapshot/gamematch",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(__file__, OUT / "source-driver.py")
    bundle = load_bundle(ROOT, "overwatch")
    save_data_identity(OUT / "data", bundle, audit_primary_split(ROOT, bundle, 0))
    for name in MODELS:
        target = OUT / name
        target.mkdir(exist_ok=True)
        status_path = target / "status.json"
        if status_path.exists() and read(status_path)["status"] == "completed":
            continue
        status = {"model": name, "status": "searching", "started_utc": stamp()}
        write_json(status_path, status)
        active = list(range(16))
        stages = []
        try:
            for rung, setup in enumerate(protocol["rungs"]):
                records = {}
                for i in active:
                    records[i] = []
                    cfg = {**spaces[name][i], "epochs": setup["epochs"]}
                    for seed in setup["seeds"]:
                        directory = target / f"rung-{rung}" / f"trial-{i:02d}" / f"seed-{seed}"
                        if (directory / "metrics.json").exists():
                            result = read(directory / "metrics.json")
                            if (
                                result["configuration"] != cfg
                                or result["test_evaluated"] is not False
                            ):
                                raise ValueError(f"Invalid specialist search resume: {directory}")
                        else:
                            print(
                                f"SPECIALIST {name} rung={rung} trial={i} seed={seed}", flush=True
                            )
                            model, result = fit(
                                bundle.task,
                                name,
                                bundle.specification,
                                bundle.parts["train"],
                                bundle.parts["valid"],
                                cfg,
                                seed,
                                directory,
                            )
                            del model
                            print(
                                f"VALID {name} rung={rung} trial={i} score={result['validation_primary']:.7f} seconds={result['seconds']:.1f}",
                                flush=True,
                            )
                        records[i].append(result)
                ranking = sorted(
                    records,
                    key=lambda i: (np.mean([r["validation_primary"] for r in records[i]]), i),
                )
                stages.append({"rung": rung, "active": active, "ranking": ranking})
                write_json(target / "promotions.json", stages)
                if rung < 2:
                    count = protocol["rungs"][rung + 1]["trials"]
                    active = [0] + [i for i in ranking if i != 0][: count - 1]
            winner = select_trial(records, [200, 201])
            cfg = {**spaces[name][winner], "epochs": 48}
            selection = {
                "trial": winner,
                "configuration": cfg,
                "selection_seeds": [200, 201],
                "validation_mean": float(
                    np.mean([r["validation_primary"] for r in records[winner]])
                ),
                "protocol_sha256": file_hash(marker),
                "test_used": False,
            }
            selected = target / "selection.json"
            if selected.exists():
                if read(selected) != selection:
                    raise ValueError("Locked specialist choice changed")
            else:
                write_json(selected, selection)
            status.update(status="final_evaluation", selection=winner)
            write_json(status_path, status)
            for seed in protocol["final_seeds"]:
                directory = target / "final" / f"seed-{seed}"
                if (directory / "result.json").exists():
                    continue
                print(f"SPECIALIST FINAL {name} trial={winner} seed={seed}", flush=True)
                if (directory / "metrics.json").exists():
                    result = read(directory / "metrics.json")
                    model = reload_model(directory, bundle.task, name, bundle.specification, cfg)
                else:
                    model, result = fit(
                        bundle.task,
                        name,
                        bundle.specification,
                        bundle.parts["train"],
                        bundle.parts["valid"],
                        cfg,
                        seed,
                        directory,
                    )
                result = {
                    **result,
                    **evaluate(model, bundle, cfg, directory),
                    "dataset": "overwatch",
                    "test_evaluated": True,
                    "selection_sha256": file_hash(selected),
                }
                restored = reload_model(directory, bundle.task, name, bundle.specification, cfg)
                a, b = (
                    predict_raw(m, bundle.parts["valid"], bundle.task, cfg)
                    for m in [model, restored]
                )
                if not np.allclose(a, b, atol=1e-6, rtol=1e-5):
                    raise ValueError("Specialist reload mismatch")
                result["reload_max_abs_error"] = float(np.max(np.abs(a - b)))
                write_json(directory / "result.json", result)
                del model, restored
            status.update(status="completed", completed_utc=stamp())
            write_json(status_path, status)
            torch.mps.empty_cache()
        except Exception as exc:
            status.update(
                status="failed", error=f"{type(exc).__name__}: {exc}", updated_utc=stamp()
            )
            write_json(status_path, status)
            raise
    print("SPECIALIST MATRIX COMPLETED", flush=True)


if __name__ == "__main__":
    main()
