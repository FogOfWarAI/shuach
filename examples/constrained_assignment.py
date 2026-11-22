# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""A data-free symmetric party/role assignment example."""

import json

import numpy as np

from gamematch.matching.constraints import constrained_assignments
from gamematch.matching.search import minimize_assignments


def main():
    skills = np.array([0.2, 1.1, -0.4, 0.7, 0.5, -0.1])
    candidates = constrained_assignments(
        range(6),
        (3, 3),
        parties=[(0, 1)],
        roles=(("support", "damage", "damage"), ("support", "damage", "damage")),
        allowed_roles={0: {"support"}, 3: {"support"}, 2: {"damage"}},
    )

    def scorer(batch):
        gap = np.asarray(
            [skills[list(a.teams[0])].sum() - skills[list(a.teams[1])].sum() for a in batch]
        )
        return 1 / (1 + np.exp(-gap))

    result = minimize_assignments(candidates, scorer, lambda p: np.abs(p - 0.5), batch_size=4)
    assignment = result.pop("assignment")
    print(
        json.dumps(
            {
                **result,
                "teams": assignment.teams,
                "roles": assignment.roles,
                "scope": "Synthetic scores illustrate the API; no observed or causal human outcome",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
