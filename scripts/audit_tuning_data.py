# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Rebuild tuning partitions and compare current inputs to frozen manifests."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gamematch.data.heroes import file_hash
from gamematch.tuning.data import load_bundle
from gamematch.tuning.search import audit_primary_split

ROOT = Path(__file__).resolve().parents[1]


def audit_prediction_identity(path, bundle, split):
    """Check labels/row identities against reconstructed data, not a score alone."""
    part = bundle.parts[split]
    target = (
        part.metadata["y_true"] if split == "test" and bundle.task == "regression" else part.target
    )
    expected = {"row_id": part.row_id, "y_true": target}
    if split == "test":
        expected.update(part.metadata)
    with np.load(path, allow_pickle=False) as saved:
        for field, values in expected.items():
            if field not in saved or not np.array_equal(saved[field], values):
                raise ValueError(f"Prediction identity differs from {split} data: {path}: {field}")


def completed_prediction_paths(root, name, world, specialist=False):
    """Only audit completed fits; synthetic seeds select different data worlds."""
    base = root if specialist else root / name
    validations = (
        list(base.glob("*/rung-*/trial-*/seed-*/metrics.json"))
        if specialist
        else list(base.glob("*/search/trial-*/seed-*/metrics.json"))
    )
    finals = list(base.glob("*/final/seed-*/result.json"))
    synthetic = name.startswith("synthetic-")
    for marker in validations:
        seed = int(marker.parent.name.removeprefix("seed-"))
        if not synthetic or world == (1 if seed == 101 else 0):
            yield marker.with_name("validation.npz"), "valid"
    for marker in finals:
        seed = int(marker.parent.name.removeprefix("seed-"))
        if not synthetic or world == seed:
            yield marker.with_name("validation.npz"), "valid"
            yield marker.with_name("predictions.npz"), "test"


def verify_snapshot_revision(original, current):
    """Allow strengthened cache identity only after exact old/new array equality."""
    if original == current:
        return None
    for key in ["source_sha256", "reference_examples_sha256", "snapshot_seconds"]:
        if original["identity"][key] != current["identity"][key]:
            raise ValueError(f"Snapshot input identity changed: {key}")
    reference = ROOT / "datasets/derived/dota2-reference-v2"
    paths = [reference / m.get("cache_file", "snapshots.npz") for m in [original, current]]
    for path, metadata in zip(paths, [original, current]):
        if not path.resolve().is_relative_to(reference.resolve()):
            raise ValueError("Snapshot cache must remain within its reference directory")
        if file_hash(path) != metadata["cached_sha256"]:
            raise ValueError(f"Snapshot cache hash changed: {path}")
    with np.load(paths[0], allow_pickle=False) as old, np.load(paths[1], allow_pickle=False) as new:
        if set(old.files) != set(new.files):
            raise ValueError("Snapshot revision changed available arrays")
        for key in old.files:
            if not np.array_equal(old[key], new[key]):
                raise ValueError(f"Snapshot revision changed experimental inputs: {key}")
        return {
            "original_cache": str(paths[0].relative_to(ROOT)),
            "current_cache": str(paths[1].relative_to(ROOT)),
            "original_sha256": original["cached_sha256"],
            "current_sha256": current["cached_sha256"],
            "arrays_equal": sorted(old.files),
            "rows": len(old["row_id"]),
        }


def main():
    paths = [
        p
        for name in ["hpo-v1", "hpo-v1-baselines"]
        for p in (ROOT / "runs" / name).glob("*/data-world-*/data-manifest.json")
    ]
    specialist = ROOT / "runs/specialist-hpo-v2/data/data-manifest.json"
    if specialist.exists():
        paths.append(specialist)
    groups = {}
    for path in paths:
        value = json.loads(path.read_text())
        key = (value["name"], value["provenance"].get("world", 0))
        groups.setdefault(key, []).append((path, value))
    anchors = json.loads((ROOT / "reports/decision-environment-audit.json").read_text())[0]
    if anchors["run"] != "runs/decisions-v1":
        raise ValueError("Unexpected independent synthetic environment audit")
    environment_hashes = {w["world"]: w["environment_sha256"] for w in anchors["worlds"]}
    results = []
    prediction_count = 0
    snapshot_revisions = []
    for (name, world), records in sorted(groups.items()):
        bundle = load_bundle(ROOT, name, world)
        provenance = dict(bundle.provenance)
        provenance.pop("preprocessor", None)
        if name.startswith("synthetic-"):
            actual = file_hash(Path(provenance["decision_environment"]))
            if actual != environment_hashes[f"{name[10:]}-seed-{world}"]:
                raise ValueError("Synthetic oracle environment changed since independent audit")
        expected = {
            "name": bundle.name,
            "task": bundle.task,
            "specification": bundle.specification,
            "provenance": provenance,
            "split_counts": {p: len(d.target) for p, d in bundle.parts.items()},
            **audit_primary_split(ROOT, bundle, world),
        }
        for path, original in records:
            compared = {**expected, "provenance": dict(provenance)}
            if "decision_environment_sha256" not in original["provenance"]:
                compared["provenance"].pop("decision_environment_sha256", None)
            if name == "future-rating":
                old_snapshot = original["provenance"]["data"]["snapshots"]
                current_snapshot = compared["provenance"]["data"]["snapshots"]
                equivalence = verify_snapshot_revision(old_snapshot, current_snapshot)
                if equivalence is not None:
                    snapshot_revisions.append(
                        {"manifest": str(path.relative_to(ROOT)), **equivalence}
                    )
                    compared["provenance"]["data"] = {
                        **compared["provenance"]["data"],
                        "snapshots": old_snapshot,
                    }
            if compared != original:
                raise ValueError(f"Current inputs differ from frozen manifest: {path}")
            with np.load(path.with_name("split-manifest.npz"), allow_pickle=False) as saved:
                for split, part in bundle.parts.items():
                    np.testing.assert_array_equal(saved[f"{split}_row_id"], part.row_id)
            is_specialist = path == specialist
            run_root = path.parents[1] if is_specialist else path.parents[2]
            for prediction, split in completed_prediction_paths(
                run_root, name, world, specialist=is_specialist
            ):
                audit_prediction_identity(prediction, bundle, split)
                prediction_count += 1
        results.append(
            {
                "dataset": name,
                "world": world,
                "manifests_checked": len(records),
                "split_counts": expected["split_counts"],
            }
        )
        del bundle
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "manifests_checked": len(paths),
        "prediction_archives_checked": prediction_count,
        "snapshot_revision_equivalence": snapshot_revisions,
        "dataset_worlds": results,
        "scope": "Current source files, train-fitted transformation specification, train/valid/test row identities, primary test labels, and saved validation/test labels and evaluation metadata from completed fits; synthetic environments additionally match the earlier independent oracle audit. No model fitting or score-based selection.",
    }
    (ROOT / "reports/tuning-data-integrity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "dataset_worlds"}))


if __name__ == "__main__":
    main()
