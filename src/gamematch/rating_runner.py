# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Delayed-feedback, temporal rating evaluation with validation-only calibration."""

import hashlib
import heapq
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import completed_fit, seal_artifacts, verify_completions
from .configuration import runtime_identity
from .data.events import load_overwatch, temporal_boundaries
from .evaluation.binary import binary_metrics
from .evaluation.multiclass import multiclass_metrics
from .models.ratings import DEFAULT_RATING_MODELS, RATING_MODELS
from .runner import code_hash, write_json


def replay(events, model_name, separate_domains=False):
    """All predictions at t precede feedback with availability >= t.

    Updates use state at feedback delivery, including when games overlap. Event
    row IDs break equal-delivery-time ties deterministically; they are not model
    features. Counts describe completed, observed history, not account age.
    """
    models, pending = {}, []
    counts, domain_counts = defaultdict(int), defaultdict(int)
    scores, history = [], []

    def model_for(event):
        domain = event.domain if separate_domains else 0
        if domain not in models:
            models[domain] = RATING_MODELS[model_name]()
        return models[domain]

    def update(event):
        model_for(event).update(event.teams, event.outcome)
        for team in event.teams:
            for player in team:
                counts[player] += 1
                domain_counts[(event.domain, player)] += 1

    last_time = -float("inf")
    for index, event in enumerate(events):
        if event.time < last_time:
            raise ValueError("Replay events must be sorted chronologically")
        last_time = event.time
        while pending and pending[0][0] < event.time:
            _, _, previous = heapq.heappop(pending)
            update(previous)
        players = event.teams[0] + event.teams[1]
        n = np.array([counts[p] for p in players])
        in_domain = np.array([domain_counts[(event.domain, p)] for p in players])
        history.append([n.min(), n.mean(), (n == 0).mean(), in_domain.min(), in_domain.mean()])
        scores.append(model_for(event).features(event.teams))
        heapq.heappush(pending, (event.available_at, event.row_id, event))
        if (index + 1) % 10000 == 0:
            print(
                f"  {model_name} separate={separate_domains}: {index + 1}/{len(events)}", flush=True
            )
    while pending:
        update(heapq.heappop(pending)[2])
    return np.asarray(scores), np.asarray(history), models


def calibration_features(score, domain):
    return np.column_stack([score, np.abs(score), domain, score * domain])


def fit_calibration(score, domain, outcome, valid):
    if set(np.unique(outcome[valid])) != {0, 1, 2}:
        raise ValueError("Calibration partition must contain all three outcome classes")
    calibrator = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    calibrator.fit(calibration_features(score[valid], domain[valid]), outcome[valid])
    return calibrator


def snapshot_ratings(models):
    result = {}
    for domain, model in models.items():
        result[str(domain)] = {
            player: (
                {"rating": float(value)}
                if isinstance(value, (int, float))
                else {"mu": float(value.mu), "sigma": float(value.sigma)}
            )
            for player, value in model.ratings.items()
        }
    return result


def evaluate_probabilities(outcome, probabilities, mask):
    if not mask.any():
        return None
    result = multiclass_metrics(outcome[mask], probabilities[mask])
    decisive = mask & (outcome != 1)
    if decisive.any():
        result["decisive_conditional_binary"] = binary_metrics(
            (outcome[decisive] == 2).astype(int),
            probabilities[decisive, 2] / (probabilities[decisive, 0] + probabilities[decisive, 2]),
        )
    return result


def run_ratings(config_path: Path, workspace: Path, output: Path):
    config_path, workspace, output = (Path(p).resolve() for p in (config_path, workspace, output))
    verify_completions(output)
    config = json.loads(config_path.read_text())
    if config.get("protocol") != "temporal_delayed_feedback":
        raise ValueError("Explicit temporal_delayed_feedback protocol is required")
    model_names = config.get("models", DEFAULT_RATING_MODELS)
    if set(model_names) - set(RATING_MODELS):
        raise ValueError("Unknown rating model")
    events, data_manifest = load_overwatch(
        workspace / config.get("source", "datasets/raw/overwatch.jsonl"),
        config.get("feedback_delay_seconds", 0),
    )
    bounds = temporal_boundaries(
        events, config.get("train_fraction", 0.7), config.get("valid_fraction", 0.15)
    )
    timestamp = np.asarray([e.time for e in events])
    available = np.asarray([e.available_at for e in events])
    outcome = np.asarray([e.outcome for e in events], dtype=np.int64)
    domain = np.asarray([e.domain for e in events], dtype=np.int64)
    train = (timestamp < bounds[0]) & (available < bounds[0])
    valid = (timestamp >= bounds[0]) & (timestamp < bounds[1]) & (available < bounds[1])
    test = timestamp >= bounds[1]
    runtime = runtime_identity()
    fingerprint = hashlib.sha256(
        json.dumps(
            {"config": config, "data": data_manifest, "code": code_hash(), "runtime": runtime},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Existing rating output has different source, data or configuration")
    else:
        manifest = {
            "fingerprint": fingerprint,
            "code_sha256": code_hash(),
            "config": config,
            "runtime": runtime,
            "data": data_manifest,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "time_boundaries": bounds,
            "classes": ["loss", "draw", "win"],
            "split_counts": {
                "train_completed": int(train.sum()),
                "calibration": int(valid.sum()),
                "test": int(test.sum()),
                "calibration_boundary_purged": int(
                    ((timestamp >= bounds[0]) & (timestamp < bounds[1]) & ~valid).sum()
                ),
            },
            "dependencies": {
                p: runtime["dependencies"][p]
                for p in ("numpy", "scikit-learn", "trueskill", "openskill")
            },
            "update_policy": "Use outcomes strictly before current prediction timestamp; earlier test outcomes may update rating state online; calibrator remains frozen",
            "calibration": "One multinomial logistic calibrator per model, fit on validation only, features=[score,abs(score),domain,score*domain]",
            "rating_parameters": {
                "elo": {
                    "K": 32,
                    "scale": 400,
                    "aggregation": "mean",
                    "per_player_update": "K*error",
                },
                "trueskill": {
                    "mu": 25,
                    "sigma": 25 / 3,
                    "beta": 25 / 6,
                    "tau": 25 / 300,
                    "draw_probability": 0.02,
                },
                "openskill": "PlackettLuce library defaults",
                "openskill_bt_full": "BradleyTerryFull library defaults",
                "openskill_tm_full": "ThurstoneMostellerFull library defaults",
            },
            "evidence_type": "observed_prequential_prediction_under_timestamp_assumption",
        }
        shutil.copytree(
            Path(__file__).parent,
            output / "source-snapshot/gamematch",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        write_json(output / "resolved_config.json", config)
        np.savez_compressed(
            output / "split-manifest.npz",
            row_id=np.array([e.row_id for e in events]),
            time=timestamp,
            available_at=available,
            train=train,
            valid=valid,
            test=test,
            y_true=outcome,
            domain=domain,
        )
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    try:
        specifications = [("calibrated_prior", False)] + [
            (m, separate)
            for separate in config.get("separate_domains", [False, True])
            for m in model_names
        ]
        for name, separate in specifications:
            tag = name + ("-separate" if separate else "-pooled")
            directory = output / tag
            required = ("metrics.json", "calibrator.joblib", "predictions.npz")
            if name != "calibrated_prior":
                required += ("rating-state.json",)
            if completed_fit(directory, required, output):
                print(f"Skipping completed {tag}", flush=True)
                continue
            print(f"Rating replay {tag}", flush=True)
            directory.mkdir(exist_ok=True)
            start = time.perf_counter()
            if name == "calibrated_prior":
                score, history, models = np.zeros(len(events)), None, {}
            else:
                score, history, models = replay(events, name, separate)
            replay_seconds = time.perf_counter() - start
            calibrator = fit_calibration(score, domain, outcome, valid)
            probability = calibrator.predict_proba(calibration_features(score, domain))
            joblib.dump(calibrator, directory / "calibrator.joblib")
            if models:
                write_json(directory / "rating-state.json", snapshot_ratings(models))
            save = {
                "row_id": np.array([e.row_id for e in events])[test],
                "time": timestamp[test],
                "y_true": outcome[test],
                "domain": domain[test],
                "probabilities": probability[test],
                "raw_conditional_p_win": expit(score[test]),
            }
            if history is not None:
                save["completed_history"] = history[test]
            np.savez_compressed(directory / "predictions.npz", **save)
            result = {
                "model": name,
                "separate_domains": separate,
                "replay_seconds": replay_seconds,
                "metrics": evaluate_probabilities(outcome, probability, test),
                "subgroups": {
                    "competitive": evaluate_probabilities(
                        outcome, probability, test & (domain == 1)
                    ),
                    "noncompetitive": evaluate_probabilities(
                        outcome, probability, test & (domain == 0)
                    ),
                },
            }
            if history is not None:
                result["subgroups"].update(
                    {
                        "all_players_previously_observed": evaluate_probabilities(
                            outcome, probability, test & (history[:, 0] > 0)
                        ),
                        "at_least_one_first_observed_player": evaluate_probabilities(
                            outcome, probability, test & (history[:, 0] == 0)
                        ),
                        "all_players_at_least_5_completed": evaluate_probabilities(
                            outcome, probability, test & (history[:, 0] >= 5)
                        ),
                    }
                )
                decisive = test & (outcome != 1)
                result["uncalibrated_decisive_binary"] = binary_metrics(
                    (outcome[decisive] == 2).astype(int), expit(score[decisive])
                )
            write_json(directory / "metrics.json", result)
            seal_artifacts(directory, required, output)
            print(f"  {tag}: test WDL log loss={result['metrics']['log_loss']:.6f}", flush=True)
        manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        results = [json.loads(p.read_text()) for p in sorted(output.glob("*/metrics.json"))]
        write_json(output / "results.json", results)
    return output / "results.json"
