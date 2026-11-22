# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Reload a hero-outcome model and score hypothetical character compositions.

This demonstrates prediction/search composition. The entities are characters,
not persistent people, and no counterfactual outcome is available for evaluation.
"""

import argparse
import json

import numpy as np

from gamematch.inference import HeroPredictor
from gamematch.matching.constraints import constrained_assignments
from gamematch.matching.search import minimize_assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--heroes", nargs=10, type=int)
    args = parser.parse_args()
    predictor = HeroPredictor(args.artifact, batch_size=64)
    heroes = args.heroes or predictor.vocabulary.values[:10].tolist()
    if len(heroes) != 10 or len(set(heroes)) != 10:
        raise ValueError("This example requires ten distinct character identifiers")
    if not set(heroes) <= set(predictor.vocabulary.values.tolist()):
        raise ValueError("Use training-observed hero identifiers for this example")

    def score(assignments):
        rosters = np.asarray([assignment.teams for assignment in assignments], dtype=np.int64)
        return predictor.predict(rosters)

    original = predictor.predict(np.asarray(heroes).reshape(1, 2, 5))[0]
    result = minimize_assignments(
        constrained_assignments(heroes, (5, 5)),
        score,
        objective=lambda probability: np.abs(probability - 0.5),
        batch_size=64,
    )
    chosen = result.pop("assignment")
    print(
        json.dumps(
            {
                **result,
                "entity_kind": "hero_character",
                "selected_teams": chosen.teams,
                "input_order_positive_probability": float(original),
                "selected_positive_probability": float(score([chosen])[0]),
                "scope": "Hypothetical character compositions scored by a saved predictor. No observed reassignment outcome or human-matchmaking effectiveness is inferred.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
