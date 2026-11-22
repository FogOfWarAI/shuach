# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""A 1v4 interface example with a five-outcome distribution and explicit target.

The hand-specified role abilities and conditionally independent eliminations are
illustrative assumptions. They are not fitted DBD predictions or observed gains.
"""

import argparse
import json

import numpy as np

from gamematch.matching.constraints import constrained_assignments
from gamematch.matching.search import minimize_assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-eliminations", type=float, default=2.0)
    args = parser.parse_args()
    if not np.isfinite(args.target_eliminations) or not 0 <= args.target_eliminations <= 4:
        parser.error("target-eliminations must be finite and in [0, 4]")
    predator_ability = np.array([0.1, 1.2, -0.2, 0.5, 0.6])
    prey_ability = np.array([-0.3, 1.0, 0.2, 0.4, 0.7])

    def scorer(batch):
        distributions = []
        for assignment in batch:
            predator, prey = assignment.teams
            gap = predator_ability[predator[0]] - prey_ability[list(prey)]
            chance = 1 / (1 + np.exp(-gap))
            distribution = np.array([1.0])
            for probability in chance:
                distribution = np.convolve(distribution, [1 - probability, probability])
            distributions.append(distribution)
        return np.stack(distributions)

    result = minimize_assignments(
        constrained_assignments(
            range(5),
            (1, 4),
            parties=[(2, 3)],
            roles=(("predator",), ("prey",) * 4),
        ),
        scorer,
        objective=lambda probabilities: np.abs(
            probabilities @ np.arange(5) - args.target_eliminations
        ),
        batch_size=2,
    )
    assignment = result.pop("assignment")
    probability = scorer([assignment])[0]
    print(
        json.dumps(
            {
                **result,
                "teams": assignment.teams,
                "roles": assignment.roles,
                "elimination_probabilities_0_to_4": probability.tolist(),
                "expected_eliminations": float(probability @ np.arange(5)),
                "target_eliminations": args.target_eliminations,
                "scope": "Hand-specified synthetic probabilities and a caller-chosen target; this demonstrates the API, not DBD reassignment validity or observed human outcomes. The party (2,3) remains together.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
