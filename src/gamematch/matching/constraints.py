# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Small-lobby candidate generation with asymmetric sides, parties and roles."""

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Assignment:
    """One feasible assignment, with players aligned to any supplied role slots.

    Attributes:
        teams: Left and right player tuples.
        roles: Left and right role tuples in slot order, or None for untyped teams.
            When present, `teams[side][slot]` is assigned `roles[side][slot]`.
    """

    teams: tuple[tuple, tuple]
    roles: tuple[tuple, tuple] | None = None


def constrained_assignments(
    players, sizes, parties=(), roles=None, allowed_roles=None, exchangeable_sides=False
):
    """Enumerate assignments satisfying team, party and role constraints.

    Args:
        players: Finite iterable of distinct, hashable player IDs. Input order
            determines candidate order and breaks ties in the search functions.
        sizes: Two positive Python integers summing to the number of players.
        parties: Nonempty, disjoint groups of known players who must stay together.
        roles: Two ordered collections of role slots matching the team sizes.
            Repeated role names within a side denote interchangeable slots.
        allowed_roles: Optional mapping from player ID to permitted role names.
            A single string denotes one role. Omitted players may fill any slot.
        exchangeable_sides: Keep the first input player on the left. Requires
            equal team sizes, identical slot layouts and a side-invariant objective.

    Yields:
        An Assignment for each feasible team and slot allocation. Infeasible
        constraints produce an empty iterator.

    Raises:
        ValueError: On iteration, for invalid team sizes, player identities,
            party memberships, slot counts or incompatible side exchangeability.
    """
    players, sizes = tuple(players), tuple(sizes)
    if (
        len(sizes) != 2
        or any(isinstance(size, bool) or not isinstance(size, int) for size in sizes)
        or min(sizes) < 1
        or sum(sizes) != len(players)
        or len(set(players)) != len(players)
    ):
        raise ValueError("Two complete disjoint teams with positive sizes are required")
    if roles is not None:
        roles = tuple(tuple(slots) for slots in roles)
        if tuple(map(len, roles)) != sizes:
            raise ValueError("Role slots must match team sizes")
    if allowed_roles and roles is None:
        raise ValueError("Role restrictions require explicit role slots")
    if exchangeable_sides and (
        sizes[0] != sizes[1] or (roles is not None and roles[0] != roles[1])
    ):
        raise ValueError("Different team sizes or role layouts cannot be exchangeable")
    known = set(players)
    party_sets = [set(p) for p in parties]
    union = set()
    for group in party_sets:
        if not group or not group <= known or group & union:
            raise ValueError("Party memberships must be nonempty, known and nonoverlapping")
        union |= group
    allowed_roles = allowed_roles or {}
    if set(allowed_roles) - known:
        raise ValueError("Role restrictions name unknown players")
    # Every branch must see the same options, even when a caller passes an iterator.
    allowed_roles = {
        player: (choices,) if isinstance(choices, str) else tuple(choices)
        for player, choices in allowed_roles.items()
    }
    index = {p: i for i, p in enumerate(players)}

    def role_orders(team, slots):
        # Prune equivalent equal-role permutations before constructing them.
        # Input-order DFS retains the old filtered-permutations ordering and ties.
        previous = [
            next((j for j in range(i - 1, -1, -1) if slots[j] == role), None)
            for i, role in enumerate(slots)
        ]
        order, used = [], set()

        def extend():
            position = len(order)
            if position == len(slots):
                yield tuple(order)
                return
            role, prior = slots[position], previous[position]
            for player in team:
                if player in used or (
                    player in allowed_roles and role not in allowed_roles[player]
                ):
                    continue
                if prior is not None and index[player] < index[order[prior]]:
                    continue
                order.append(player)
                used.add(player)
                yield from extend()
                used.remove(player)
                order.pop()

        yield from extend()

    for chosen in combinations(range(len(players)), sizes[0]):
        if exchangeable_sides and 0 not in chosen:
            continue
        chosen = set(chosen)
        a = tuple(p for i, p in enumerate(players) if i in chosen)
        b = tuple(p for i, p in enumerate(players) if i not in chosen)
        aset = set(a)
        if any(group & aset and not group <= aset for group in party_sets):
            continue
        if roles is None:
            yield Assignment((a, b))
        else:
            for left in role_orders(a, roles[0]):
                for right in role_orders(b, roles[1]):
                    yield Assignment((left, right), roles)
