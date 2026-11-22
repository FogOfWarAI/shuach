# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Compare corrected DBD inputs against the original frozen adapter across pandas versions."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gamematch.data import asymmetric
from gamematch.data.asymmetric import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from gamematch.data.heroes import file_hash


def arrays(data):
    return {
        "time": data.time,
        "available_at": data.available_at,
        "outcome": data.outcome,
        "row_ids": np.asarray(data.row_ids, dtype=str),
        "player_ids": np.asarray(data.player_ids, dtype=str),
        "numeric": data.features[NUMERIC_FEATURES].to_numpy(),
        "categorical": data.features[CATEGORICAL_FEATURES].to_numpy(dtype=str),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "compare"])
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.workspace.resolve()
    folder = root / ".cache/asymmetric-input-reference"
    report_path = root / "reports/pandas-compatibility.json"
    if args.mode == "build":
        folder.mkdir(parents=True, exist_ok=True)
        frozen = root / "runs/asymmetric-v1/source-snapshot/gamematch/data/asymmetric.py"
        spec = importlib.util.spec_from_file_location("gamematch.data.reference_asymmetric", frozen)
        legacy = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = legacy
        spec.loader.exec_module(legacy)
        report = {
            "status": "old_environment_passed_new_environment_pending",
            "original_pandas": pd.__version__,
            "frozen_adapter_sha256": file_hash(frozen),
            "fixed_adapter_sha256": file_hash(Path(asymmetric.__file__)),
            "checks": [],
        }
        for name in ["dbd-2022", "dbd-2023"]:
            old = legacy.load_asymmetric(name, root / "datasets")
            new = asymmetric.load_asymmetric(name, root / "datasets")
            for key, value in arrays(old).items():
                np.testing.assert_array_equal(value, arrays(new)[key])
            np.savez_compressed(folder / f"{name}.npz", **arrays(old))
            report["checks"].append(
                {
                    "dataset": name,
                    "rows": len(old.time),
                    "old_and_fixed_arrays_exactly_equal": True,
                    "source_sha256": old.manifest["sha256"],
                    "reference_arrays_sha256": file_hash(folder / f"{name}.npz"),
                }
            )
            print(name, "reference and corrected input identical", flush=True)
    else:
        report = json.loads(report_path.read_text())
        if report["fixed_adapter_sha256"] != file_hash(Path(asymmetric.__file__)):
            raise ValueError("Corrected adapter changed between environments")
        for item in report["checks"]:
            name = item["dataset"]
            cached = folder / f"{name}.npz"
            if file_hash(cached) != item["reference_arrays_sha256"]:
                raise ValueError("Reference arrays changed")
            new = asymmetric.load_asymmetric(name, root / "datasets")
            with np.load(cached, allow_pickle=False) as old:
                for key, value in arrays(new).items():
                    np.testing.assert_array_equal(old[key], value)
            item["fresh_pandas_arrays_exactly_equal"] = True
            print(name, "new dependency environment exact equality", flush=True)
        report.update(
            status="passed",
            fresh_pandas=pd.__version__,
            scope="Every row, timestamp, availability, outcome, identity and model feature in both full DBD collections compared across frozen original and corrected adapters; existing benchmark inputs unchanged.",
        )
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
