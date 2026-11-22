# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Independently reload completed compact-history models at a new batch size."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gamematch.configuration import runtime_identity
from gamematch.data.heroes import file_hash
from gamematch.indexed_inference import IndexedHistoryPredictor
from gamematch.runner import code_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


def main(require_complete=False):
    base = ROOT / "runs/league-neural-v1"
    data = ROOT / "datasets/derived/leagueData-v1"
    protocol = json.loads((base / "protocol.json").read_text())
    rows = np.load(data / "test-rows.npy", allow_pickle=False)
    selected = np.sort(
        np.random.default_rng(20261114).choice(len(rows), min(128, len(rows)), replace=False)
    )
    pending, checks = [], []
    for model in protocol["models"]:
        for variant in protocol["variants"]:
            for seed in protocol["seeds"]:
                directory = base / variant / model / f"seed-{seed}"
                key = str(directory.relative_to(ROOT))
                if not (directory / "result.json").exists():
                    pending.append(key)
                    continue
                result = json.loads((directory / "result.json").read_text())
                if (
                    result["prediction_sha256"] != file_hash(directory / "predictions.npz")
                    or result["selection_sha256"] != file_hash(directory / "selection.json")
                    or result["protocol_sha256"] != file_hash(base / "protocol.json")
                ):
                    raise ValueError("Saved final artifact identity changed: " + key)
                predictor = IndexedHistoryPredictor(directory, data, threads=2)
                with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["row_id"], predictor.store.row_id[rows])
                    np.testing.assert_array_equal(saved["y_true"], predictor.store.target[rows])
                    raw = predictor.predict_logits(rows[selected], batch_size=37)
                    p = predictor.predict_proba(rows[selected], batch_size=37)[:, 1]
                    raw_error = float(np.abs(raw - saved["raw"][selected]).max())
                    probability_error = float(np.abs(p - saved["p_positive"][selected]).max())
                if (
                    not np.isfinite([raw_error, probability_error]).all()
                    or max(raw_error, probability_error) > 5e-5
                ):
                    raise ValueError("Reloaded predictor differs: " + key)
                checks.append(
                    {
                        "path": key,
                        "rows": len(selected),
                        "raw_max_abs_error": raw_error,
                        "probability_max_abs_error": probability_error,
                        "checkpoint_sha256": file_hash(directory / "checkpoint.pt"),
                        "result_sha256": file_hash(directory / "result.json"),
                    }
                )
                del predictor
    report = {
        "complete": not pending,
        "passed_completed_fits": len(checks),
        "checks": checks,
        "pending": pending,
        "sample_seed": 20261114,
        "sample_test_positions": selected.tolist(),
        "batch_size": 37,
        "device": "cpu",
        "tolerance": 5e-5,
        "current_source_sha256": code_hash(),
        "driver_sha256": file_hash(Path(__file__)),
        "runtime": runtime_identity(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent inference API reconstruction on 128 sampled test rows per completed fit. No training or selection. This checks numerical reload consistency, not general cross-version serialization or all test predictions.",
    }
    write_json(ROOT / "reports/richer-history/indexed-predictor-reload.json", report)
    print(
        json.dumps(
            {
                "passed": len(checks),
                "pending": len(pending),
                "max_probability_error": max(
                    (r["probability_max_abs_error"] for r in checks), default=None
                ),
            }
        )
    )
    if require_complete and pending:
        raise RuntimeError("Some declared predictors have not finished training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    main(parser.parse_args().require_complete)
