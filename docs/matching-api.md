# Team assignment API

Use `exact_balance` for two equal-size teams and a binary win predictor. Use
`constrained_assignments` with `minimize_assignments` for parties, role slots,
unequal team sizes or a custom objective. Both solvers evaluate a finite lobby.

## Work through a four-player lobby

Ava and Bo can each captain a team. Each side needs one captain and one flex
player. The example chooses the assignment with the smallest difference in total
rating. Swapping the sides leaves this objective unchanged, so we can keep Ava
on the left.

```python
from gamematch.matching.constraints import constrained_assignments
from gamematch.matching.search import minimize_assignments

ratings = {"Ava": 6, "Bo": 4, "Cy": 3, "Di": 1}
assignments = constrained_assignments(
    players=ratings.keys(),
    sizes=(2, 2),
    roles=(("captain", "flex"), ("captain", "flex")),
    allowed_roles={"Ava": {"captain"}, "Bo": {"captain"}},
    exchangeable_sides=True,
)

def score(batch):
    totals = []
    for assignment in batch:
        left, right = assignment.teams
        totals.append((sum(ratings[p] for p in left), sum(ratings[p] for p in right)))
    return totals

result = minimize_assignments(assignments, score, lambda totals: [abs(a - b) for a, b in totals])
assert result["assignment"].teams == (("Ava", "Di"), ("Bo", "Cy"))
assert result["cost"] == 0
assert result["candidate_count"] == 2
assert result["candidate_index"] == 1
```

## Generate candidates

Import `Assignment` and `constrained_assignments` from
`gamematch.matching.constraints`.

```text
constrained_assignments(players, sizes, parties=(), roles=None, allowed_roles=None,
                        exchangeable_sides=False)
```

| Parameter | Contract |
| --- | --- |
| `players` | Finite iterable of distinct, hashable IDs. Supply a reproducibly ordered iterable if tied solutions should be reproducible. |
| `sizes` | Two positive Python integers in left/right order. Their sum must equal the player count. |
| `parties` | Iterable of disjoint, nonempty groups of known players. Every member of a group must land on the same side. |
| `roles` | Two ordered collections of role slots, one per player on each side. Repeating a role makes those slots interchangeable. The default leaves roles unspecified. |
| `allowed_roles` | Mapping from player to permitted role names. A single string means one complete role name. Omitted players may take any role. An empty collection prevents that player from filling any slot. Restrictions require explicit slots. |
| `exchangeable_sides` | With `True`, the first input player stays on the left. Both sides must have equal sizes and identical role layouts. The caller must also establish that the objective is invariant under swapping sides. |

The generator yields frozen `Assignment(teams, roles)` instances.
`teams[side][i]` fills `roles[side][i]` when roles are present. Without
slots, `roles` is `None` and each team retains the input order of its members.
Invalid configurations such as duplicate players, unknown party members and
incompatible slot counts raise `ValueError` when iteration starts. A valid set
of constraints can also produce no candidates, for example if a party is too
large for either side.

## Score candidates in batches

Import `minimize_assignments` from `gamematch.matching.search`.

```text
minimize_assignments(candidates, score_candidates, objective, batch_size=256)
```

`score_candidates` receives an ordered list of up to `batch_size` items.
`objective` receives its result and returns one finite real cost per item, in a
vector of shape `(len(batch),)`. For a NumPy vector of binary win probabilities
use `lambda p: abs(p - 0.5)`. For categorical predictions choose an objective
that reflects the meaning and ordering of the classes. A scorer should evaluate
each candidate consistently across different batch sizes.

| Returned field | Meaning |
| --- | --- |
| `assignment` | Original item from `candidates` with the lowest cost. |
| `cost` | Selected objective value as a float. |
| `candidate_index` | Zero-based position of the winner in the complete input stream. Equal costs select the earliest position, including across batch boundaries. |
| `candidate_count` | Total number of candidates evaluated. |
| `status` | `"optimal_over_enumerated_candidates"` refers to the supplied candidate set, scores and objective. |
| `batch_size` | Requested maximum number of candidates per call. |

An empty stream, an invalid batch size, a cost vector of the wrong shape or
nonfinite costs raise `ValueError`. Every candidate is scored. Batching limits
the number of candidate objects held by the solver at once, while the total
work still grows with the number of possible teams and role assignments.

## Equal-size teams and a binary predictor

Import `partitions` and `exact_balance` from `gamematch.matching.lobby`.

```text
partitions(players, team_size, exchangeable_sides=False)
exact_balance(players, team_size, predict_win, exchangeable_sides=False)
```

`partitions` yields plain `(left, right)` tuple pairs. With ten distinct players
and five per team it yields 252 ordered partitions, or 126 when sides are
exchangeable. `exact_balance` passes a list of **all** pairs to `predict_win` in
one call. Return one finite left-win probability in [0, 1] for each pair, with
shape `(candidate_count,)`. A matrix of two class probabilities per row must be
reduced to the left-win column by the scorer.

The result contains `teams`, `predicted_win`, `predicted_imbalance`
(`abs(predicted_win - 0.5)`), `candidate_index`, `candidate_count` and
`status`. Ties select the first partition. The optimum refers to predicted
imbalance. Measuring the effect of deploying the selected teams requires a
separate evaluation.
