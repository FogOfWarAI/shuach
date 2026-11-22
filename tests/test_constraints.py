# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import unittest
from itertools import combinations, permutations

import numpy as np

from gamematch.matching.constraints import constrained_assignments
from gamematch.matching.lobby import partitions


class ConstraintTests(unittest.TestCase):
    def test_one_shot_role_choices_match_reusable_choices_for_every_branch(self):
        slots = (("front", "back"), ("front", "back"))
        expected = list(constrained_assignments(range(4), (2, 2), roles=slots))
        actual = list(
            constrained_assignments(
                range(4),
                (2, 2),
                roles=(iter(side) for side in slots),
                allowed_roles={0: iter(("front", "back"))},
            )
        )
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 24)

    def test_single_role_string_matches_only_the_complete_name(self):
        slots = (("support", "port"),) * 2
        as_string = list(
            constrained_assignments(range(4), (2, 2), roles=slots, allowed_roles={0: "support"})
        )
        as_set = list(
            constrained_assignments(range(4), (2, 2), roles=slots, allowed_roles={0: {"support"}})
        )
        self.assertEqual(as_string, as_set)
        self.assertEqual(len(as_string), 12)

    def test_partitions_requires_an_integral_team_size(self):
        for size in (True, 1.0):
            with self.subTest(size=size), self.assertRaises(ValueError):
                list(partitions(range(2), size))
        self.assertEqual(len(list(partitions(range(2), np.int64(1)))), 2)

    def test_repeated_role_pruning_preserves_candidate_and_tie_order(self):
        players = (10, 2, 8, 4, 6, 0)
        index = {p: i for i, p in enumerate(players)}
        slots = (("a", "b", "a"), ("b", "a", "b"))
        expected = []
        for left in combinations(players, 3):
            right = tuple(p for p in players if p not in left)
            orders = []
            for team, roles in zip((left, right), slots):
                orders.append(
                    [
                        order
                        for order in permutations(team)
                        if not any(
                            roles[i] == roles[j] and index[order[i]] > index[order[j]]
                            for i in range(3)
                            for j in range(i + 1, 3)
                        )
                    ]
                )
            expected.extend((a, b) for a in orders[0] for b in orders[1])
        actual = list(constrained_assignments(players, (3, 3), roles=slots))
        self.assertEqual([a.teams for a in actual], expected)
        repeated = list(constrained_assignments(range(10), (5, 5), roles=(("a",) * 5,) * 2))
        self.assertEqual(len(repeated), 252)

    def test_small_role_and_party_spaces_match_independent_permutation_enumeration(self):
        rng = np.random.default_rng(15)
        players = tuple(range(6))
        for sizes in [(2, 4), (3, 3)]:
            slots = (("a",) + ("b",) * (sizes[0] - 1), ("a",) + ("b",) * (sizes[1] - 1))
            for _ in range(8):
                allowed = {p: {r for r in ["a", "b"] if rng.random() < 0.7} for p in players}

                def key(order):
                    return tuple(
                        sorted(
                            (player, side, role)
                            for side, (team, roles) in enumerate(zip(order, slots))
                            for player, role in zip(team, roles)
                        )
                    )

                expected = set()
                for order in permutations(players):
                    teams = (order[: sizes[0]], order[sizes[0] :])
                    if (0 in teams[0]) != (1 in teams[0]):
                        continue
                    if all(
                        role in allowed[player]
                        for team, roles in zip(teams, slots)
                        for player, role in zip(team, roles)
                    ):
                        expected.add(key(teams))
                actual = list(
                    constrained_assignments(
                        players, sizes, parties=[(0, 1)], roles=slots, allowed_roles=allowed
                    )
                )
                keys = [key(item.teams) for item in actual]
                self.assertEqual(set(keys), expected)
                self.assertEqual(len(keys), len(set(keys)))

    def test_incomplete_role_configuration_and_noninteger_sizes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit role slots"):
            list(constrained_assignments(range(4), (2, 2), allowed_roles={0: {"support"}}))
        for sizes in [(2.0, 2.0), (True, 3)]:
            with self.assertRaises(ValueError):
                list(constrained_assignments(range(4), sizes))

    def test_asymmetric_party_and_roles(self):
        result = list(
            constrained_assignments(
                range(5),
                (1, 4),
                parties=[(1, 2)],
                roles=[("predator",), ("prey",) * 4],
                allowed_roles={0: {"predator"}, 1: {"prey"}, 2: {"prey"}, 3: {"prey"}, 4: {"prey"}},
            )
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].teams, ((0,), (1, 2, 3, 4)))

    def test_exact_counts_and_no_party_split(self):
        self.assertEqual(len(list(constrained_assignments(range(10), (5, 5)))), 252)
        self.assertEqual(
            len(list(constrained_assignments(range(10), (5, 5), exchangeable_sides=True))), 126
        )
        result = list(constrained_assignments(range(4), (2, 2), parties=[(0, 1)]))
        self.assertEqual(len(result), 2)
        for c in result:
            self.assertEqual(0 in c.teams[0], 1 in c.teams[0])

    def test_role_permutations_are_not_silently_dropped(self):
        result = list(constrained_assignments(range(4), (2, 2), roles=[("a", "b"), ("a", "b")]))
        self.assertEqual(len(result), 24)
        with self.assertRaises(ValueError):
            list(constrained_assignments(range(5), (1, 4), exchangeable_sides=True))
        with self.assertRaises(ValueError):
            list(constrained_assignments(range(4), (2, 2), parties=[(0, 1), (1, 2)]))


if __name__ == "__main__":
    unittest.main()
