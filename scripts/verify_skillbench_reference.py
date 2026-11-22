# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Pinned Skillbench module probe; generated fixtures, no paper-result claim."""

import hashlib
import heapq
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import trueskill
from scipy.special import expit

from gamematch.configuration import runtime_identity
from gamematch.data.events import MatchEvent
from gamematch.matching.constraints import constrained_assignments
from gamematch.matching.search import minimize_assignments
from gamematch.models.ratings import RATING_MODELS, TrueSkillRating
from gamematch.rating_runner import replay

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reading/software/skillbench-runtime"


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for item in manifest["files"]:
        if hashlib.sha256((SOURCE / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Pinned Skillbench source changed")
    sys.path.insert(0, str(SOURCE))
    from skillbench import Match, MatchDataset, Simulator, Team, TeamPair
    from skillbench.acquirers import LikeliestDrawAF
    from skillbench.emulators import TrueSkillPlayersEmulator

    if sys.version_info >= (3, 11):
        raise ValueError(
            "Use Python 3.10: pinned TeamPair samples a frozenset, unsupported in 3.11+"
        )

    def gm_model():
        model = TrueSkillRating(draw_probability=0.1)
        # Match Skillbench's documented code settings, including its backend.
        model.environment = trueskill.TrueSkill(draw_probability=0.1, backend="mpmath")
        return model

    def pair(teams, rng):
        a, b = [Team("/".join(sorted(t)), list(t)) for t in teams]
        return TeamPair(a, b, rng), a, b

    rng = random.Random(20261114)
    players = [f"p{i}" for i in range(40)]
    author, ours = TrueSkillPlayersEmulator(), gm_model()
    probability_error, state_error, trace, fixtures = [], [], [], []
    for i in range(200):
        roster = rng.sample(players, 10)
        teams = (tuple(roster[:5]), tuple(roster[5:]))
        outcome = 1 if i % 11 == 0 else rng.choice([0, 2])
        matchup, first, second = pair(teams, rng)
        p_author, p_ours = float(author.emulate(first, second)), float(expit(ours.features(teams)))
        probability_error.append(abs(p_author - p_ours))
        winner = first if outcome == 2 else second if outcome == 0 else None
        author.fit_one_match(matchup, winner)
        ours.update(teams, outcome)
        for player in roster:
            state_error.extend(
                [
                    abs(author.ratings[player].mu - ours.ratings[player].mu),
                    abs(author.ratings[player].sigma - ours.ratings[player].sigma),
                ]
            )
        trace.append(
            {
                "index": i,
                "outcome": outcome,
                "author_probability": p_author,
                "gamematch_probability": p_ours,
            }
        )
        fixtures.append((teams, outcome))
    if max(probability_error) > 1e-10 or max(state_error) > 1e-9:
        raise ValueError("Matched-backend rating comparison failed")

    lobby = tuple(rng.sample(players, 10))
    candidates = list(constrained_assignments(lobby, (5, 5)))
    acquisition = LikeliestDrawAF()
    author_costs = []
    for assignment in candidates:
        matchup, _, _ = pair(assignment.teams, rng)
        author_costs.append((1 - float(acquisition(author, matchup))) / 2)
    choice = minimize_assignments(
        candidates,
        lambda batch: np.array([expit(ours.features(a.teams)) for a in batch]),
        lambda p: np.abs(p - 0.5),
        batch_size=37,
    )
    optimality_error = abs(choice["cost"] - min(author_costs))
    selected_author_gap = author_costs[choice["candidate_index"]] - min(author_costs)
    if optimality_error > 1e-10 or selected_author_gap > 1e-10:
        raise ValueError("Matched candidate-objective comparison failed")

    # Exercise the actual author simulator on a fresh generated MatchDataset.
    matches = []
    for i, (teams, outcome) in enumerate(fixtures[:12]):
        matchup, first, second = pair(teams, rng)
        winner = first if outcome == 2 else second if outcome == 0 else None
        matches.append(Match(i, i * 60, matchup, winner))
    simulator = Simulator(MatchDataset(matches), random.Random(42))
    emulated = TrueSkillPlayersEmulator()
    simulator.fit_emulator(emulated, 12, acquisition, bar=False)
    fit_count = sum(emulated.matchup_fit_count.values())
    smoke_accuracy = float(simulator.evaluate_emulator(emulated))
    if fit_count != 12 or simulator.matchups_left or not 0 <= smoke_accuracy <= 1:
        raise ValueError("Author simulator smoke failed")

    # Explicitly add the same external feedback queue to the author emulator.
    events = [
        MatchEvent(str(i), i * 60, i * 60 + 180, teams, outcome, 0, duration=180)
        for i, (teams, outcome) in enumerate(fixtures[:80])
    ]
    RATING_MODELS["matched_skillbench"] = gm_model
    scores, _, _ = replay(events, "matched_skillbench")
    delayed, immediate = TrueSkillPlayersEmulator(), TrueSkillPlayersEmulator()
    pending, delayed_probabilities, immediate_probabilities = [], [], []

    def update(model, event):
        matchup, first, second = pair(event.teams, rng)
        model.fit_one_match(
            matchup, first if event.outcome == 2 else second if event.outcome == 0 else None
        )

    for event in events:
        while pending and pending[0][0] < event.time:
            update(delayed, heapq.heappop(pending)[2])
        _, first, second = pair(event.teams, rng)
        delayed_probabilities.append(float(delayed.emulate(first, second)))
        immediate_probabilities.append(float(immediate.emulate(first, second)))
        update(immediate, event)
        heapq.heappush(pending, (event.available_at, event.row_id, event))
    delayed_error = float(np.max(np.abs(expit(scores) - delayed_probabilities)))
    timing_difference = float(
        np.max(np.abs(np.asarray(immediate_probabilities) - delayed_probabilities))
    )
    if delayed_error > 1e-10 or timing_difference <= 1e-6:
        raise ValueError("Feedback-availability contract probe failed")
    report = {
        "status": "passed",
        "author_commit": manifest["commit"],
        "source": str(SOURCE.relative_to(ROOT)),
        "runtime": runtime_identity(),
        "python_executable": sys.executable,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "platform": platform.platform(),
        "execution": "Unmodified pinned source modules loaded from local cache; not a pip installation or the authors' original dependency lock. Python 3.10 preserves the set-sampling API used by this commit. Locally available tqdm supplied on an isolated search path.",
        "matched_parameters": {
            "mu": 25,
            "sigma": 25 / 3,
            "beta": 25 / 6,
            "tau": 25 / 300,
            "draw_probability": 0.1,
            "backend": "mpmath",
        },
        "sequential_matches": 200,
        "max_probability_error": max(probability_error),
        "max_rating_state_error": max(state_error),
        "candidate_count": len(candidates),
        "candidate_objective": "abs(p(left win)-0.5); equivalent to maximizing Skillbench LikeliestDrawAF for complementary win probabilities",
        "optimality_cost_error": optimality_error,
        "selected_author_optimality_gap": selected_author_gap,
        "simulator_smoke": {
            "generated_matches": 12,
            "fit_calls": fit_count,
            "remaining_matchups": len(simulator.matchups_left),
            "in_sample_accuracy_not_a_benchmark": smoke_accuracy,
        },
        "delayed_feedback": {
            "matches": 80,
            "duration_seconds": 180,
            "spacing_seconds": 60,
            "max_queued_prediction_error": delayed_error,
            "max_difference_if_updated_immediately": timing_difference,
        },
        "scope": "Function and interface correspondence on generated fixtures. Same TrueSkill library/backend, explicitly matched parameters. No CS:GO paper-result reproduction, speed ranking, held-out model-quality comparison or deployment gain. Skillbench's historical acquisition protocol and GameMatch's delayed replay/new-assignment evaluation remain different tasks.",
    }
    (ROOT / "reports/skillbench-runtime-probe.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (ROOT / "reports/skillbench-runtime-trace.json").write_text(
        json.dumps(trace, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
